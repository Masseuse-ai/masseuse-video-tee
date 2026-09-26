"""Regional motion descriptors: the last code that reads pixels.

Everything in this module is computed from the decoded grey frame and the
keypoints of the frame's pose row. Its output is a small dictionary of
numbers per frame pair (`StripDescriptor`, `MotionSample.as_json`) which the
producer hands to the analysis process over a local socket
(`analysis/protocol.md`). The analysis process never sees a frame; this
module is where the pixels stop.

Three pieces:

  RowAssembler       one pose row per streamed frame, interpolated between
                     the model's pose-cadence rows and decided as late as it
                     must be so that a stream and an offline pass over the
                     same rows agree bit for bit. Keypoints only, no pixels.
  pelvis_frame       a body-carried coordinate frame from the two hips and
                     the two knees. Everything below is measured in it, so
                     translation, rotation and scale of the body in the
                     picture leave before anything is measured and what
                     remains is deformation of the surface.
  DescriptorWorker   per frame pair: the hip region and a longer strip along
                     the body axis are resampled into canonical patches, a
                     laterally displaced copy of the strip lands on the
                     support surface beside the body as a registration
                     control, dense optical flow (Farneback) runs between
                     consecutive strips, and the flow and brightness fields
                     are reduced to per-window statistics. Two cadences,
                     30 fps pairs and 6 fps pairs, because a statistic over a
                     0.033 s pair and one over a 0.167 s pair are different
                     measurements and the analysis wants both.

Units: lengths are in hip widths (the distance between the hip keypoints,
so the descriptors are independent of camera distance), flow in canonical
pixels per frame, brightness in the source's grey levels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pose rows at frame cadence
# ---------------------------------------------------------------------------

# 30 fps frames per 6 fps pose interval.
FRAMES_PER_POSE = 5

# A pose gap wider than this is not bridged by interpolation: the frames in
# between get no keypoints and the descriptors reset across them.
MAX_POSE_BRIDGE_S = 0.35

# A streamed frame this close to a pose row's own timestamp reuses that pose
# outright. Half a 30 fps step, so the 6 fps grid points, which land exactly
# on every fifth 30 fps frame, reproduce their pose bit for bit.
POSE_SNAP_S = 1.0 / 60.0


def pose_between(before: dict, after: dict, at_s: float) -> dict:
    """Keypoints lerped between two pose rows, scores taken pessimistically.

    Linear interpolation of the joints, not of any derived frame: everything
    downstream is built from joints, so interpolating anything else would let
    two paths disagree about what a pose means. The score of an interpolated
    point is the smaller of its endpoints: an endpoint the model was unsure of
    contaminates every pose between it and its neighbour.
    """
    weight = (at_s - before["atS"]) / (after["atS"] - before["atS"])
    keypoints = {}
    for name, a in before["keypoints"].items():
        b = after["keypoints"].get(name)
        if b is None:
            continue
        keypoints[name] = [
            a[0] + (b[0] - a[0]) * weight,
            a[1] + (b[1] - a[1]) * weight,
            min(a[2], b[2]),
        ]
    return keypoints


class RowAssembler:
    """One pose row per streamed frame, decided as late as it must be.

    A 30 fps frame's pose decision is final once the pose clock has advanced
    past `atS + POSE_SNAP_S`: no future pose row can snap to it, and whether
    it bridges is decided by the two rows already seen. Pose rows without
    keypoints still advance the clock, so a model that loses the person
    stalls nothing: the affected frames emit keypoint-less rows and the
    descriptors reset, exactly as an offline pass does across a dropped
    frame.

    `bridge_s` widens `MAX_POSE_BRIDGE_S` for pose cadences below 6 fps; a
    braced, mostly still body's registration tolerates second-scale gaps.
    """

    def __init__(self, fps: float = 30.0, bridge_s: float | None = None):
        self.fps = fps
        self.bridge_s = MAX_POSE_BRIDGE_S if bridge_s is None else bridge_s
        self.tracked: deque[dict] = deque(maxlen=8)
        self.clock_s: float | None = None
        self._next_index = 0

    def push_pose(self, row: dict) -> None:
        """A pose-cadence row, in time order. Keypoint-less rows advance the
        clock without becoming interpolation endpoints."""
        self.clock_s = float(row["atS"])
        if row.get("keypoints"):
            self.tracked.append(row)

    def _final(self, at_s: float) -> bool:
        """Whether no pose row still to come could change this frame's
        decision.

        The clock passing `at_s + POSE_SNAP_S` rules out a later snap, but
        not a later bridge: a dropped pose row advances the clock while the
        row after it, the bridge's far endpoint, is still to come. So a frame
        with a predecessor and no successor yet waits until either the
        successor arrives or the clock is past the predecessor's bridge
        horizon, at which point no successor can bridge and the answer is
        final either way.
        """
        if self.clock_s is None or at_s + POSE_SNAP_S >= self.clock_s:
            return False
        rows = list(self.tracked)
        if any(abs(r["atS"] - at_s) <= POSE_SNAP_S or r["atS"] > at_s
               for r in rows):
            return True
        before = [r for r in rows if r["atS"] < at_s]
        if not before:
            return True
        return self.clock_s > before[-1]["atS"] + self.bridge_s + 1e-6

    def _decide(self, index: int, at_s: float) -> dict:
        row = {"frame": index, "atS": round(at_s, 4)}
        rows = list(self.tracked)
        if not rows:
            row["keypoints"] = None
            return row
        nearest = min(rows, key=lambda r: abs(r["atS"] - at_s))
        if abs(nearest["atS"] - at_s) <= POSE_SNAP_S:
            row["keypoints"] = nearest["keypoints"]
            return row
        before = [r for r in rows if r["atS"] < at_s]
        after = [r for r in rows if r["atS"] > at_s]
        if (before and after
                and after[0]["atS"] - before[-1]["atS"]
                <= self.bridge_s + 1e-6):
            row["keypoints"] = pose_between(before[-1], after[0], at_s)
            return row
        row["keypoints"] = None
        return row

    def flush(self) -> list[dict]:
        """Rows whose decision is final under the current pose clock."""
        if self.clock_s is None:
            return []
        out = []
        while True:
            at_s = self._next_index / self.fps
            if not self._final(at_s):
                break
            out.append(self._decide(self._next_index, at_s))
            self._next_index += 1
        return out

    def drain(self, last_index: int) -> list[dict]:
        """End of stream: decide every frame up to and including the last
        one decoded, with the poses that exist. The assembler cannot know
        where a stream ended, only the decoder does, so the bound comes in
        from the caller rather than from an unbounded clock."""
        out = []
        while self._next_index <= last_index:
            out.append(self._decide(self._next_index,
                                    self._next_index / self.fps))
            self._next_index += 1
        return out


# ---------------------------------------------------------------------------
# The pelvis frame
# ---------------------------------------------------------------------------

# A keypoint below this score is treated as unplaced.
MIN_SCORE = 0.3

# Below these the frame would be defined by keypoint noise rather than by the
# body.
MIN_HIP_WIDTH_PX = 8.0
MIN_THIGH_PX = 20.0


@dataclass(frozen=True)
class PelvisFrame:
    """The body-carried coordinate frame for one video frame.

    `origin` is the hip midpoint in source pixels, `lateral` the unit vector
    along the hip line, `axial` the unit vector along the thighs (positive
    toward the knees) orthogonalised against `lateral`, `hip_width` the
    length unit, `thigh` the mean projected femur length.
    """

    origin: np.ndarray
    lateral: np.ndarray
    axial: np.ndarray
    hip_width: float
    thigh: float

    def point(self, lateral_offset: float, axial_offset: float) -> np.ndarray:
        """Source pixel at an offset from the hip midpoint, in hip widths."""
        return (
            self.origin
            + self.lateral * (lateral_offset * self.hip_width)
            + self.axial * (axial_offset * self.hip_width)
        )


def pelvis_frame(hip_left, hip_right, knee_left, knee_right
                 ) -> PelvisFrame | None:
    """Pelvis frame from the two hips and both knees.

    Lateral comes from the hip line. Axial comes from the mean of the two
    unit femurs, orthogonalised against lateral, so an asymmetric leg
    position does not tip the axis toward whichever leg is more extended.
    None when the landmarks are too close together or too nearly collinear
    for a direction to be defined.
    """
    hip_left = np.asarray(hip_left, dtype=float)
    hip_right = np.asarray(hip_right, dtype=float)
    knee_left = np.asarray(knee_left, dtype=float)
    knee_right = np.asarray(knee_right, dtype=float)

    span = hip_right - hip_left
    hip_width = float(np.linalg.norm(span))
    if hip_width < MIN_HIP_WIDTH_PX:
        return None
    lateral = span / hip_width

    origin = (hip_left + hip_right) / 2
    femur_left, femur_right = knee_left - hip_left, knee_right - hip_right
    length_left = float(np.linalg.norm(femur_left))
    length_right = float(np.linalg.norm(femur_right))
    thigh = (length_left + length_right) / 2
    if thigh < MIN_THIGH_PX:
        return None
    if min(length_left, length_right) < MIN_HIP_WIDTH_PX:
        # One femur has collapsed to nothing in projection, so its direction
        # is noise and averaging it in would corrupt the axis.
        return None

    femur = femur_left / length_left + femur_right / length_right
    axial = femur - lateral * float(np.dot(femur, lateral))
    axial_norm = float(np.linalg.norm(axial))
    if axial_norm < 0.1:
        # The femurs project almost along the hip line: the knees are side-on
        # to the camera and there is no usable axial direction.
        return None

    return PelvisFrame(
        origin=origin, lateral=lateral, axial=axial / axial_norm,
        hip_width=hip_width, thigh=thigh,
    )


def frame_of(row: dict) -> PelvisFrame | None:
    """The pelvis frame of one pose row, or None if any of the four
    landmarks it needs is missing or unplaced."""
    keypoints = row.get("keypoints")
    if not keypoints:
        return None
    names = ("left_hip", "right_hip", "left_knee", "right_knee")
    points = [keypoints.get(name) for name in names]
    if any(point is None or point[2] < MIN_SCORE for point in points):
        return None
    left_hip, right_hip, left_knee, right_knee = [
        np.asarray(point[:2], dtype=float) for point in points
    ]
    return pelvis_frame(left_hip, right_hip, left_knee, right_knee)


# ---------------------------------------------------------------------------
# The hip-region patch
# ---------------------------------------------------------------------------

# The hip region as a square canonical patch: centre and half-extents as
# offsets from the hip midpoint in hip widths, axial positive toward the
# knees. Its mean axial flow between consecutive frames is the coarsest
# descriptor (`MotionSample.region_axial_flow`).
PATCH = 128
REGION_LATERAL_HALF = 0.961
REGION_AXIAL_CENTRE = 0.157
REGION_AXIAL_HALF = 0.333


def _dest(lateral: int, axial: int) -> np.ndarray:
    return np.float32([
        [lateral / 2, axial / 2],
        [lateral, axial / 2],
        [lateral / 2, axial],
    ])


def region_patch(image: np.ndarray, frame: PelvisFrame) -> np.ndarray:
    """Resample the hip region into the canonical PATCH x PATCH square.

    Three source points (centre, lateral edge, axial edge) define the affine
    map, so the patch reads head-up, knees-down however the body lies.
    """
    centre = frame.point(0.0, REGION_AXIAL_CENTRE)
    source = np.float32([
        centre,
        centre + frame.lateral * (REGION_LATERAL_HALF * frame.hip_width),
        centre + frame.axial * (REGION_AXIAL_HALF * frame.hip_width),
    ])
    matrix = cv2.getAffineTransform(source, _dest(PATCH, PATCH))
    return cv2.warpAffine(
        image, matrix, (PATCH, PATCH),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# The axial strip and its windows
# ---------------------------------------------------------------------------

# Canonical strip as (columns, rows), square canonical pixels: one blur
# sigma has to mean the same physical distance across and along the body,
# and the axial gradient in `axial_strain` has to be comparable to the
# lateral one.
COLUMNS = 256

# Axial span of the strip in hip widths from the hip midpoint, lower back to
# upper thigh. Wide enough to hold every candidate window plus the upper
# control windows, so one warp per frame serves the whole sweep.
STRIP_AXIAL = (-1.45, 0.75)

SCALE = COLUMNS / (2 * REGION_LATERAL_HALF)          # canonical px per hip width
ROWS = int(round((STRIP_AXIAL[1] - STRIP_AXIAL[0]) * SCALE))

# Blur applied before anything is measured, in hip widths so it is a
# physical distance rather than a pixel count: coarser than skin texture,
# finer than the surface relief being measured.
SMOOTH_HIP_WIDTHS = 0.045

# Where the strip is split into its left and right halves, as a lateral
# offset from the hip midpoint in hip widths.
MIDLINE_OFFSET = 0.158

# Candidate axial windows, as centres in hip widths with a fixed half-extent,
# stepping across the whole hip region and past both its boundaries. Which
# window carries the cleanest signal depends on the camera angle, so the
# strip is reported for all of them and the analysis chooses.
WINDOW_HALF = 0.15
WINDOW_CENTRES = tuple(round(-0.45 + 0.10 * step, 2) for step in range(11))

# The rows the flow statistics are taken over: the hip region proper.
STRAIN_WINDOW = (-0.236, 0.534)

# The control strip: the same warp slid sideways off the body onto the
# support surface. It inherits every registration error the real strip has,
# under the same lighting, on a surface that cannot deform, so whatever
# rhythm it shows is registration artifact rather than movement.
CONTROL_LATERAL = -1.9

# Two windows above the hip region, on the lower back. Reported for
# inspection alongside the control strip.
BACK_CENTRES = (-0.95, -1.30)

CONTROL_PREFIX = "bed"
BACK_PREFIX = "back"


@dataclass(frozen=True)
class Window:
    """An axial window of the strip, in hip widths."""

    low: float
    high: float

    @property
    def key(self) -> str:
        return f"{(self.low + self.high) / 2:+.2f}".replace("+", "p").replace(
            "-", "m").replace(".", "")

    def rows(self) -> slice:
        first = int(round(row_for(self.low)))
        last = int(round(row_for(self.high)))
        return slice(max(0, min(first, ROWS - 1)), max(1, min(last, ROWS)))


CANDIDATES = tuple(
    Window(centre - WINDOW_HALF, centre + WINDOW_HALF)
    for centre in WINDOW_CENTRES
)

BACKS = tuple(
    Window(centre - WINDOW_HALF, centre + WINDOW_HALF)
    for centre in BACK_CENTRES
)


def row_for(axial: float) -> float:
    """Canonical row holding a given axial offset, in hip widths."""
    low, high = STRIP_AXIAL
    return (axial - low) / (high - low) * (ROWS - 1)


def axial_for(row: float) -> float:
    low, high = STRIP_AXIAL
    return low + row / (ROWS - 1) * (high - low)


def strip(image: np.ndarray, frame: PelvisFrame,
          lateral_offset: float = 0.0,
          smooth_hip_widths: float = SMOOTH_HIP_WIDTHS) -> np.ndarray:
    """The whole axial span, warped into the pelvis frame and blurred.

    `lateral_offset` slides the strip sideways in hip widths; the control
    strip is this function at `CONTROL_LATERAL`.
    """
    low, high = STRIP_AXIAL
    centre = (
        frame.origin
        + frame.axial * (((low + high) / 2) * frame.hip_width)
        + frame.lateral * (lateral_offset * frame.hip_width)
    )
    source = np.float32([
        centre,
        centre + frame.lateral * (REGION_LATERAL_HALF * frame.hip_width),
        centre + frame.axial * (((high - low) / 2) * frame.hip_width),
    ])
    destination = np.float32([
        [COLUMNS / 2, ROWS / 2], [COLUMNS, ROWS / 2], [COLUMNS / 2, ROWS],
    ])
    patch = cv2.warpAffine(
        image, cv2.getAffineTransform(source, destination), (COLUMNS, ROWS),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    sigma = smooth_hip_widths * SCALE
    if sigma <= 0:
        return patch
    return cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)


def halves(midline_offset: float = MIDLINE_OFFSET) -> tuple[slice, slice]:
    """Column ranges of the left and right halves of the strip. The lateral
    axis runs left hip to right hip, so the first slice is the left."""
    split = int(round(COLUMNS / 2 + midline_offset * SCALE))
    split = max(1, min(split, COLUMNS - 1))
    return slice(0, split), slice(split, COLUMNS)


def profile(patch: np.ndarray, window: Window, columns: slice) -> np.ndarray:
    """Mean brightness along the body axis, over one half's columns."""
    return patch[window.rows(), columns].mean(axis=1)


def profile_range(patch: np.ndarray, window: Window, columns: slice) -> float:
    """Peak-to-trough range of the axial brightness profile: how much axial
    shading structure the window holds. A range rather than a value at a
    fixed row because it is invariant to the sub-pixel registration wobble
    that shifts the whole profile bodily."""
    values = profile(patch, window, columns)
    if not values.size:
        return float("nan")
    return float(values.max() - values.min())


def axial_strain(flow: np.ndarray, window: Window, columns: slice) -> float:
    """Mean axial gradient of the axial flow: local compression or stretch
    of the surface along the body axis. Registration has already removed
    rigid motion; differentiating removes any residual bulk drift of the
    patch as well."""
    if flow.ndim != 3 or flow.shape[2] < 2:
        return float("nan")
    axial = flow[window.rows(), columns, 1]
    if axial.size < 2 or axial.shape[0] < 2:
        return float("nan")
    return float(np.gradient(axial, axis=0).mean())


def half_means(flow: np.ndarray, window: Window,
               midline_offset: float = MIDLINE_OFFSET
               ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Mean (lateral, axial) flow per half over the window's rows, left then
    right, in canonical pixels per frame. Their differences are what a
    whole-window mean throws away: movement of the two halves toward or
    away from each other, and left-right asymmetry under an oblique camera.
    """
    if flow is None or flow.ndim != 3 or flow.shape[2] < 2:
        return None
    region = flow[window.rows(), :, :]
    if region.shape[0] < 1 or region.shape[1] < 2:
        return None
    left, right = halves(midline_offset)
    return (
        (float(region[:, left, 0].mean()), float(region[:, left, 1].mean())),
        (float(region[:, right, 0].mean()),
         float(region[:, right, 1].mean())),
    )


@dataclass(frozen=True)
class RigidMotion:
    """A flow field split into what a rigid patch could do and what is left.

    Fitted by least squares over the `STRAIN_WINDOW` rows and all columns.
    On a full rectangular grid centred on its own centroid the normal
    equations decouple, so the fit is exact and closed-form: translation is
    the mean flow, rotation and scale are projections onto the perpendicular
    and radial coordinate fields. The residual, reported per half as an RMS,
    is the part that is shape change. Units are canonical pixels per frame
    (radians per frame for `omega`, per-frame fraction for `scale`).
    """

    translation: tuple[float, float]   # (lateral, axial), px/frame
    omega: float                       # rad/frame, about the window centroid
    scale: float                       # fractional expansion per frame
    deform: tuple[float, float]        # per-half RMS residual, px/frame

    def as_json(self) -> dict:
        return {
            "translation": [self.translation[0], self.translation[1]],
            "omega": self.omega,
            "scale": self.scale,
            "deform": [self.deform[0], self.deform[1]],
        }


def rigid_split(flow: np.ndarray,
                midline_offset: float = MIDLINE_OFFSET) -> RigidMotion | None:
    """Fit the similarity motion of the hip region and measure what escapes
    it, per half."""
    if flow is None or flow.ndim != 3 or flow.shape[2] < 2:
        return None
    window = Window(*STRAIN_WINDOW)
    region = flow[window.rows(), :, :]
    if region.shape[0] < 2 or region.shape[1] < 2:
        return None
    height, width = region.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    xs -= xs.mean()
    ys -= ys.mean()
    u = region[..., 0].astype(np.float64)
    v = region[..., 1].astype(np.float64)

    radius_sq = float((xs * xs + ys * ys).sum())
    tx = float(u.mean())
    ty = float(v.mean())
    scale = float((xs * u + ys * v).sum() / radius_sq)
    omega = float((xs * v - ys * u).sum() / radius_sq)

    residual_u = u - (tx + scale * xs - omega * ys)
    residual_v = v - (ty + omega * xs + scale * ys)
    magnitude_sq = residual_u ** 2 + residual_v ** 2
    left, right = halves(midline_offset)
    return RigidMotion(
        translation=(tx, ty),
        omega=omega,
        scale=scale,
        deform=(
            float(np.sqrt(magnitude_sq[:, left].mean())),
            float(np.sqrt(magnitude_sq[:, right].mean())),
        ),
    )


# Farneback parameters, shared by every flow field measured here.
FLOW_PARAMS = (0.5, 3, 15, 3, 5, 1.2, 0)


def flow_between(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Dense optical flow from `previous` to `current` (both single-channel),
    as an (rows, columns, 2) field of (lateral, axial) displacements."""
    return cv2.calcOpticalFlowFarneback(
        previous.astype(np.uint8), current.astype(np.uint8), None,
        *FLOW_PARAMS,
    )


@dataclass(frozen=True)
class StripDescriptor:
    """One frame's strip statistics: brightness profile ranges over every
    candidate window and the control windows, and, when the previous strip
    was one step earlier, the flow statistics between the two."""

    at_s: float
    profile_range: dict[str, tuple[float, float]]
    control: dict[str, tuple[float, float]]
    axial_strain: tuple[float, float] | None
    rigid: RigidMotion | None
    control_rigid: RigidMotion | None
    half_flow: tuple[tuple[float, float], tuple[float, float]] | None

    def as_json(self) -> dict:
        """Full precision: the wire carries exactly what was measured."""
        return {
            "atS": self.at_s,
            "profileRange": {
                key: [left, right]
                for key, (left, right) in self.profile_range.items()
            },
            "control": {
                key: [left, right]
                for key, (left, right) in self.control.items()
            },
            "axialStrain": (list(self.axial_strain)
                            if self.axial_strain is not None else None),
            "rigid": self.rigid.as_json() if self.rigid else None,
            "controlRigid": (self.control_rigid.as_json()
                             if self.control_rigid else None),
            "halfFlow": ([list(half) for half in self.half_flow]
                         if self.half_flow is not None else None),
        }


def describe(patch: np.ndarray, previous: np.ndarray | None,
             control: np.ndarray | None = None,
             previous_control: np.ndarray | None = None,
             midline_offset: float = MIDLINE_OFFSET,
             at_s: float = 0.0) -> StripDescriptor:
    """Every strip statistic for one frame, from one warp.

    `previous` is the preceding frame's strip, or None when the pair is not
    consecutive: a dropped frame makes a flow field between two moments that
    are not one step apart, and its gradient would be scaled wrongly rather
    than merely noisy. `previous_control` exists so the rigid split has its
    control computed through the identical arithmetic.
    """
    left, right = halves(midline_offset)
    values = {
        window.key: (profile_range(patch, window, left),
                     profile_range(patch, window, right))
        for window in CANDIDATES
    }
    controls = {
        f"{BACK_PREFIX}{window.key}": (
            profile_range(patch, window, left),
            profile_range(patch, window, right),
        )
        for window in BACKS
    }
    if control is not None:
        for window in CANDIDATES:
            controls[f"{CONTROL_PREFIX}{window.key}"] = (
                profile_range(control, window, left),
                profile_range(control, window, right),
            )
    strain = None
    rigid = None
    control_rigid = None
    half_flow = None
    if previous is not None and previous.shape == patch.shape:
        flow = flow_between(previous, patch)
        window = Window(*STRAIN_WINDOW)
        strain = (
            axial_strain(flow, window, left),
            axial_strain(flow, window, right),
        )
        rigid = rigid_split(flow, midline_offset)
        half_flow = half_means(flow, window, midline_offset)
        if (control is not None and previous_control is not None
                and previous_control.shape == control.shape):
            control_rigid = rigid_split(flow_between(previous_control, control),
                                        midline_offset)
    return StripDescriptor(at_s=at_s, profile_range=values, control=controls,
                           axial_strain=strain, rigid=rigid,
                           control_rigid=control_rigid, half_flow=half_flow)


# ---------------------------------------------------------------------------
# Per frame pair, at two cadences
# ---------------------------------------------------------------------------

@dataclass
class MotionSample:
    """One cadence's descriptors for one frame pair."""

    at_s: float
    strip: StripDescriptor
    region_axial_flow: float

    def as_json(self) -> dict:
        body = self.strip.as_json()
        body["regionAxialFlow"] = self.region_axial_flow
        return body


class _PairState:
    def __init__(self):
        self.frame: int = -(10 ** 9)
        self.patch = None
        self.strip = None
        self.control = None


class DescriptorWorker:
    """The per-pair arithmetic, held open across a stream.

    `step` consumes each 30 fps (row, grey image) in order and returns the
    30 fps sample (or None while the pair is not consecutive) plus, on the
    6 fps grid, the 6 fps-pair sample. Consecutiveness is judged in each
    cadence's own frame numbering.
    """

    def __init__(self):
        self.fast = _PairState()
        self.slow = _PairState()

    @staticmethod
    def _measure(state: _PairState, row: dict, image: np.ndarray,
                 stride: int) -> MotionSample | None:
        frame = frame_of(row)
        if frame is None or image is None:
            state.frame = -(10 ** 9)
            state.patch = state.strip = state.control = None
            return None
        patch = region_patch(image, frame)
        current = strip(image, frame)
        control = strip(image, frame, lateral_offset=CONTROL_LATERAL)
        sample = None
        if state.patch is not None and row["frame"] == state.frame + stride:
            registered = cv2.calcOpticalFlowFarneback(
                state.patch, patch, None, *FLOW_PARAMS,
            )
            sample = MotionSample(
                at_s=float(row["atS"]),
                strip=describe(
                    current, state.strip, control=control,
                    previous_control=state.control,
                    at_s=float(row["atS"]),
                ),
                region_axial_flow=float(registered[..., 1].mean()),
            )
        state.frame = row["frame"]
        state.patch, state.strip, state.control = patch, current, control
        return sample

    def step(self, row: dict, image: np.ndarray
             ) -> tuple[MotionSample | None, MotionSample | None]:
        fast = self._measure(self.fast, row, image, stride=1)
        slow = None
        if row["frame"] % FRAMES_PER_POSE == 0:
            slow = self._measure(self.slow, row, image,
                                 stride=FRAMES_PER_POSE)
        return fast, slow
