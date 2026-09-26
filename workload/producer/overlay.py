"""The live annotated view: every Sapiens2 keypoint drawn on the frame it
was detected on, encoded and published in real time.

The producer's rows keep 21 body points (`pose_rows.row_for`); the model
emits 308. The renderer taps the full result through `GpuPose.on_full`,
before the row is built, so the row schema - and the equivalence gate on
`poses.jsonl` - is untouched.

Three real-time decisions live here:

  aligned, not stale   The view runs `delay_s` behind the decode head. Pose
                       for a frame lands queue-wait plus inference after
                       the frame (up to ~0.8 s at queue depth 4), and the
                       *next* detection is needed to interpolate, so a
                       frame is drawn only once both brackets exist. A
                       skeleton drawn on the frame it was detected on is
                       what makes the video read as "this is what Sapiens2
                       sees"; the latest pose over a live frame misaligns
                       by up to that much motion.
  fixed cadence        Frames go to the encoder on a fixed tick whatever
                       the source does: a decode stall repeats the last
                       picture, a burst is skipped through (the target is
                       always `head - delay`), so the RTP clock never
                       drifts and the view never accumulates delay.
  never the pose budget  `offer_frame` is an O(1) append of a reference on
                       the decode thread; downscale, drawing and the pipe
                       write happen here, and overflow drops the oldest
                       frame, counted, rather than ever blocking.

Interpolation follows `body_motion.pose_between`: joints lerped, scores
the pessimistic minimum of the endpoints, so an endpoint the model was
unsure of cannot be laundered past MIN_KEYPOINT_SCORE by its neighbour.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from keypoints import (
    BODY, LEFT_ANKLE, LEFT_BIG_TOE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE,
    LEFT_HAND, LEFT_HEEL, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_SMALL_TOE,
    MIN_KEYPOINT_SCORE, NOSE, RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_EAR,
    RIGHT_ELBOW, RIGHT_EYE, RIGHT_HAND, RIGHT_HEEL, RIGHT_HIP, RIGHT_KNEE,
    RIGHT_SHOULDER, RIGHT_SMALL_TOE,
)

SAPIENS2_KEYPOINTS = 308
# The body block has no wrists (keypoints.py): the arm ends at the elbow.
BODY_EDGES = (
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_BIG_TOE),
    (LEFT_HEEL, LEFT_SMALL_TOE),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_BIG_TOE),
    (RIGHT_HEEL, RIGHT_SMALL_TOE),
)
FEET = tuple(range(LEFT_BIG_TOE, RIGHT_HEEL + 1))
FACE_START = RIGHT_HAND[-1] + 1

# BGR. One colour per block so the viewer can tell a hand guess from a
# face guess at a glance; the 21 rows the producer shares get a white ring.
COLOR_BODY = (255, 200, 0)
COLOR_FEET = (80, 220, 80)
COLOR_LEFT_HAND = (0, 160, 255)
COLOR_RIGHT_HAND = (255, 80, 255)
COLOR_FACE = (60, 230, 255)
COLOR_FAINT = (120, 120, 120)
COLOR_BOX = (255, 255, 255)
COLOR_TEXT = (240, 240, 240)
COLOR_WARN = (0, 165, 255)
COLOR_BAD = (60, 60, 255)

FAINT_SCORE = 0.15
# Two detections further apart than this are not bridged; the frames
# between them show the earlier one as stale. Matches the assembler's
# order of magnitude (the pose interval at 9 fps is 0.11 s, its bridge
# floor 0.35 s; a drop or two widens it) without ever bridging a whole
# lost second.
BRIDGE_S = 1.0
SIZE_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
# The view's cadence and bit rate when the args name none
# (producer --overlay-fps / --overlay-bitrate): every frame of the 30 fps
# decode grid, at the 200 kbit a frame that 3M at 15 fps was.
DEFAULT_FPS = 30.0
DEFAULT_BITRATE = "6M"


@dataclass(frozen=True)
class Rendition:
    """One encoding of the annotated view (RENDITIONS).

    `scale` is of the canvas; `fps` and `bitrate` None mean the canvas
    cadence and the publisher's own bit rate (the `hi` rung, today's
    view). `suffix` is appended to the relay path: `overlay-half`.
    """

    name: str
    scale: float
    fps: float | None
    bitrate: str | None
    suffix: str

    def size(self, canvas: tuple[int, int]) -> tuple[int, int]:
        """The rendition's frame size: the canvas scaled, each side rounded
        to a multiple of 4 (yuv420 needs even; 4 keeps x264 happy)."""
        if self.scale == 1.0:
            return canvas
        return tuple(max(4, int(round(side * self.scale / 4)) * 4)
                     for side in canvas)

    def rate(self, canvas_fps: float) -> float:
        """The rendition's frame rate: its own, never above the canvas's."""
        return canvas_fps if self.fps is None else min(self.fps, canvas_fps)

    def bits(self, canvas_bitrate: str) -> str:
        return canvas_bitrate if self.bitrate is None else self.bitrate

    def path(self, base: str = "overlay") -> str:
        """The relay path: the canvas's path with the suffix."""
        return f"{base}{self.suffix}"


# The renditions the view is published in, one encoder process for all
# of them (OverlayPublisher.argv). In the order a phone that cannot keep
# up steps down: frame rate first (`half`: every other frame, the same
# 200 kbit a frame), then resolution (`small`: three quarters of the
# canvas on each side, the same bits per pixel), then bits (`lean`: half
# of those). The phone picks one by its relay path (relay_proxy.route),
# the trainer's snapshot lists them (OverlayRenderer.snapshot), and the
# recording, when asked for, is of `hi` only.
RENDITIONS: dict[str, Rendition] = {
    "hi": Rendition("hi", 1.0, None, None, ""),
    "half": Rendition("half", 1.0, 15.0, "3M", "-half"),
    "small": Rendition("small", 0.75, 15.0, "1700k", "-small"),
    "lean": Rendition("lean", 0.75, 15.0, "850k", "-lean"),
}
# The frame rate of every rendition below `hi`, shared by one `fps` filter.
RENDITION_FPS = 15.0
DEFAULT_RENDITIONS = ("hi",)


def parse_renditions(text: str | None) -> tuple[str, ...]:
    """`--overlay-renditions hi,half,small,lean` as the tuple of names, in
    RENDITIONS' order, `hi` always among them (it is the view's canvas
    and the one recorded); empty or None is `hi` alone."""
    names = {part.strip() for part in (text or "").split(",") if part.strip()}
    unknown = names - set(RENDITIONS)
    if unknown:
        raise ValueError(f"unknown overlay rendition(s): {sorted(unknown)}; "
                         f"known: {list(RENDITIONS)}")
    names.add("hi")
    return tuple(name for name in RENDITIONS if name in names)


# The face inset: the phone's view drawn over the fixed camera's, in the
# top-right corner, portrait 3:4 at this fraction of the canvas height,
# this far from the edges, with a border.
INSET_ASPECT = (3, 4)
INSET_HEIGHT_FRACTION = 0.40
INSET_MARGIN = 16
INSET_BORDER = 2
COLOR_INSET_BORDER = (255, 255, 255)
# The inset's crop of the phone's frame follows the face: the box around
# the confident face keypoints, grown to this many times its size, never
# smaller than this fraction of the frame's height, moved this far toward
# each new position per frame (an exponential smoothing, so the crop does
# not shake with the keypoints). Without a face it is the middle of the
# frame.
INSET_FACE_GROW = 2.6
INSET_MIN_HEIGHT_FRACTION = 0.3
INSET_SMOOTH = 0.15
# The head's keypoints: the face block, and the body's nose, eyes and ears.
HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR) + tuple(
    range(FACE_START, SAPIENS2_KEYPOINTS))


def parse_size(text: str) -> tuple[int, int]:
    """`WxH`, both even and at least 64: what yuv420p encoders accept."""
    match = SIZE_RE.match(text.strip())
    if not match:
        raise ValueError(f"size must be WxH, got {text!r}")
    width, height = int(match.group(1)), int(match.group(2))
    if width % 2 or height % 2 or width < 64 or height < 64:
        raise ValueError(f"size must be even and >= 64x64, got {text!r}")
    return width, height


def to_array(values, width: int) -> np.ndarray | None:
    """Keypoints or scores as a float32 numpy array, whatever the model
    handed back (a torch tensor on the GPU, a list, an array)."""
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float32)
    if width == 1:
        return array.reshape(-1)
    return array.reshape(-1, width)


# -- what the renderer is given ------------------------------------------------


@dataclass
class Detection:
    """One real pose result, in full-frame pixels. `points` is None when
    the detector found nobody (the row was `missing_row`)."""

    index: int
    at_s: float
    points: np.ndarray | None
    scores: np.ndarray | None
    box: list[float] | None  # COCO xywh, like the row
    box_score: float | None
    people: int
    unresolved: bool
    wall: float


@dataclass
class Frame:
    index: int
    at_s: float
    yuv: np.ndarray
    wall: float


@dataclass
class PoseAt:
    """What to draw on one frame: the state names how it was obtained."""

    state: str  # exact | interpolated | stale | missing | none
    points: np.ndarray | None = None
    scores: np.ndarray | None = None
    box: list[float] | None = None
    box_score: float | None = None
    people: int = 0
    unresolved: bool = False
    age_s: float = 0.0


class FrameStore:
    """The last `capacity` decoded frames, oldest first."""

    def __init__(self, capacity: int):
        self.capacity = max(2, int(capacity))
        self.frames: deque[Frame] = deque()

    def offer(self, frame: Frame) -> int:
        """Keep the frame; returns how many were evicted to make room."""
        self.frames.append(frame)
        evicted = 0
        while len(self.frames) > self.capacity:
            self.frames.popleft()
            evicted += 1
        return evicted

    def head_at_s(self) -> float | None:
        return self.frames[-1].at_s if self.frames else None

    def nearest(self, at_s: float) -> Frame | None:
        """The stored frame closest to `at_s` in media time."""
        best = None
        for frame in self.frames:
            if best is None or abs(frame.at_s - at_s) < abs(best.at_s - at_s):
                best = frame
            elif frame.at_s > at_s:
                break
        return best

    def __len__(self) -> int:
        return len(self.frames)


class PoseTrack:
    """Recent detections and the bracket decision for any frame between them."""

    def __init__(self, keep: int = 24, bridge_s: float = BRIDGE_S,
                 frame_s: float = 1 / 30.0):
        self.detections: deque[Detection] = deque(maxlen=keep)
        self.bridge_s = bridge_s
        self.frame_s = frame_s

    def offer(self, detection: Detection) -> None:
        # Slot order is inference order (one worker thread drains a FIFO),
        # but a reconnect or a re-detect could in principle hand an older
        # index; keep the deque sorted so the bracket search stays simple.
        if self.detections and detection.index < self.detections[-1].index:
            items = sorted([*self.detections, detection], key=lambda d: d.index)
            self.detections.clear()
            self.detections.extend(items[-self.detections.maxlen:])
            return
        self.detections.append(detection)

    def at(self, at_s: float, index: int) -> PoseAt:
        before = after = None
        for detection in self.detections:
            if detection.at_s <= at_s + self.frame_s / 2:
                before = detection
            else:
                after = detection
                break
        if before is None:
            return PoseAt(state="none")
        age = max(0.0, at_s - before.at_s)
        if before.points is None:
            return PoseAt(state="missing", people=before.people,
                          unresolved=before.unresolved, age_s=age)
        if before.index == index or age < self.frame_s / 2:
            return PoseAt("exact", before.points, before.scores, before.box,
                          before.box_score, before.people, before.unresolved)
        if (after is not None and after.points is not None
                and after.at_s - before.at_s <= self.bridge_s + 1e-6):
            weight = (at_s - before.at_s) / (after.at_s - before.at_s)
            count = min(len(before.points), len(after.points))
            points = (before.points[:count]
                      + (after.points[:count] - before.points[:count]) * weight)
            scores = np.minimum(before.scores[:count], after.scores[:count])
            box = None
            if before.box is not None and after.box is not None:
                box = [a + (b - a) * weight
                       for a, b in zip(before.box, after.box)]
            return PoseAt("interpolated", points, scores, box,
                          before.box_score, before.people, before.unresolved)
        return PoseAt("stale", before.points, before.scores, before.box,
                      before.box_score, before.people, before.unresolved,
                      age_s=age)

    def rate_per_s(self, now_wall: float, window_s: float = 3.0) -> float:
        recent = sum(1 for d in self.detections if now_wall - d.wall <= window_s)
        return recent / window_s

    def __len__(self) -> int:
        return len(self.detections)


# -- the picture ---------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Where a source frame lands on the output canvas (fit, letterboxed),
    and the per-axis scale that carries keypoints along with it."""

    src_w: int
    src_h: int
    out_w: int
    out_h: int
    fit_w: int
    fit_h: int
    ox: int
    oy: int

    @property
    def sx(self) -> float:
        return self.fit_w / self.src_w

    @property
    def sy(self) -> float:
        return self.fit_h / self.src_h

    def point(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x * self.sx + self.ox)), int(round(y * self.sy + self.oy))


def fit_geometry(src_w: int, src_h: int, out_w: int, out_h: int) -> Geometry:
    scale = min(out_w / src_w, out_h / src_h)
    # Multiples of four: the fitted region is rebuilt as I420 before the
    # colour conversion, and its chroma planes need whole rows.
    fit_w = max(4, (int(src_w * scale) // 4) * 4)
    fit_h = max(4, (int(src_h * scale) // 4) * 4)
    return Geometry(src_w, src_h, out_w, out_h, fit_w, fit_h,
                    (out_w - fit_w) // 2, (out_h - fit_h) // 2)


def downscale_i420(yuv: np.ndarray, geometry: Geometry) -> np.ndarray:
    """A BGR canvas of the output size with the frame fitted into it.

    The planes are resized before the colour conversion: at 4K that is a
    ~3 ms operation against ~10 ms for converting the whole frame first,
    and this runs for every emitted frame.
    """
    width, height = geometry.src_w, geometry.src_h
    fit_w, fit_h = geometry.fit_w, geometry.fit_h
    canvas = np.zeros((geometry.out_h, geometry.out_w, 3), np.uint8)
    interp = cv2.INTER_AREA if fit_w < width else cv2.INTER_LINEAR
    if height % 4 == 0 and width % 2 == 0 and yuv.shape[0] == height * 3 // 2:
        y = yuv[:height]
        u = yuv[height:height + height // 4].reshape(height // 2, width // 2)
        v = yuv[height + height // 4:].reshape(height // 2, width // 2)
        small = np.empty((fit_h * 3 // 2, fit_w), np.uint8)
        small[:fit_h] = cv2.resize(y, (fit_w, fit_h), interpolation=interp)
        small[fit_h:fit_h + fit_h // 4] = cv2.resize(
            u, (fit_w // 2, fit_h // 2), interpolation=interp
        ).reshape(fit_h // 4, fit_w)
        small[fit_h + fit_h // 4:] = cv2.resize(
            v, (fit_w // 2, fit_h // 2), interpolation=interp
        ).reshape(fit_h // 4, fit_w)
        bgr = cv2.cvtColor(small, cv2.COLOR_YUV2BGR_I420)
    else:
        # An odd geometry: convert whole, then resize. Correct, just slower.
        bgr = cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420),
                         (fit_w, fit_h), interpolation=interp)
    canvas[geometry.oy:geometry.oy + fit_h,
           geometry.ox:geometry.ox + fit_w] = bgr
    return canvas


def inset_rect(out_w: int, out_h: int) -> tuple[int, int, int, int]:
    """Where the face inset sits on a canvas of this size: x, y, w, h, the
    sides even (an I420 fit lands in it), in the top-right corner."""
    height = max(8, int(out_h * INSET_HEIGHT_FRACTION) // 4 * 4)
    width = max(8, int(height * INSET_ASPECT[0] / INSET_ASPECT[1]) // 4 * 4)
    return out_w - width - INSET_MARGIN, INSET_MARGIN, width, height


def face_crop_target(points: np.ndarray | None, scores: np.ndarray | None,
                     frame_w: int, frame_h: int,
                     threshold: float = MIN_KEYPOINT_SCORE
                     ) -> tuple[float, float, float]:
    """The crop of a frame the inset should show, as centre x, centre y and
    height in frame pixels (the width follows INSET_ASPECT): around the
    confident head keypoints when there are any, else the frame's middle.
    Clamped so the crop lies within the frame."""
    aspect = INSET_ASPECT[0] / INSET_ASPECT[1]
    max_h = min(float(frame_h), frame_w / aspect)
    cx, cy, height = frame_w / 2.0, frame_h / 2.0, max_h
    if points is not None and scores is not None:
        head = [i for i in HEAD_KEYPOINTS
                if i < len(points) and scores[i] >= threshold]
        if len(head) >= 3:
            xs, ys = points[head, 0], points[head, 1]
            box_w = float(xs.max() - xs.min())
            box_h = float(ys.max() - ys.min())
            cx = float((xs.max() + xs.min()) / 2)
            cy = float((ys.max() + ys.min()) / 2)
            height = max(box_h, box_w / aspect) * INSET_FACE_GROW
            height = max(height, frame_h * INSET_MIN_HEIGHT_FRACTION)
            height = min(height, max_h)
    width = height * aspect
    cx = min(max(cx, width / 2), frame_w - width / 2)
    cy = min(max(cy, height / 2), frame_h - height / 2)
    return cx, cy, height


def crop_box(cx: float, cy: float, height: float, frame_w: int,
             frame_h: int) -> tuple[int, int, int, int]:
    """A centre-and-height crop as integer x, y, w, h: origin even and sides
    multiples of four (whole I420 chroma rows), inside the frame."""
    aspect = INSET_ASPECT[0] / INSET_ASPECT[1]
    h = max(8, min(int(height) // 4 * 4, frame_h // 4 * 4))
    w = max(8, min(int(h * aspect) // 4 * 4, frame_w // 4 * 4))
    x = int(round(cx - w / 2)) // 2 * 2
    y = int(round(cy - h / 2)) // 2 * 2
    x = min(max(0, x), frame_w - w)
    y = min(max(0, y), frame_h - h)
    return x, y, w, h


def crop_i420(yuv: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The I420 frame's `box` (x, y, w, h as crop_box makes them) as an
    I420 frame of its own, without converting the rest."""
    x, y, w, h = box
    height = yuv.shape[0] * 2 // 3
    width = yuv.shape[1]
    y_plane = yuv[:height]
    u_plane = yuv[height:height + height // 4].reshape(height // 2, width // 2)
    v_plane = yuv[height + height // 4:].reshape(height // 2, width // 2)
    out = np.empty((h * 3 // 2, w), np.uint8)
    out[:h] = y_plane[y:y + h, x:x + w]
    out[h:h + h // 4] = u_plane[y // 2:y // 2 + h // 2,
                                x // 2:x // 2 + w // 2].reshape(h // 4, w)
    out[h + h // 4:] = v_plane[y // 2:y // 2 + h // 2,
                               x // 2:x // 2 + w // 2].reshape(h // 4, w)
    return out


class CropGeometry:
    """A Geometry for keypoints in a frame of which only a crop is shown:
    the crop's origin is taken off, then the fit applies."""

    def __init__(self, fit: Geometry, x0: int, y0: int):
        self.fit = fit
        self.x0 = x0
        self.y0 = y0

    def point(self, x: float, y: float) -> tuple[int, int]:
        return self.fit.point(x - self.x0, y - self.y0)


def group_color(index: int) -> tuple[int, int, int]:
    if index in FEET:
        return COLOR_FEET
    if index < FEET[0]:
        return COLOR_BODY
    if index in LEFT_HAND:
        return COLOR_LEFT_HAND
    if index in RIGHT_HAND:
        return COLOR_RIGHT_HAND
    return COLOR_FACE


def group_radius(index: int) -> int:
    if index <= FEET[-1]:
        return 4  # body and feet: the points the producer shares
    if index < FACE_START:
        return 2  # hands
    return 1  # face


class KeypointPainter:
    """Draws one PoseAt and the HUD onto a canvas, in place.

    With mirror=True the view is a selfie: picture and skeleton are flipped
    left-to-right (the phone shows its own camera that way, so the trainer's
    view takes over without the body jumping sides) and the lettering is
    drawn after the flip so it still reads left to right."""

    def __init__(self, geometry: Geometry, threshold: float = MIN_KEYPOINT_SCORE,
                 faint: float = FAINT_SCORE, mirror: bool = False):
        self.geometry = geometry
        self.threshold = threshold
        self.faint = faint
        self.mirror = mirror

    def paint(self, canvas: np.ndarray, pose: PoseAt, hud: list[tuple[str, tuple]]) -> None:
        if pose.box is not None:
            self._box(canvas, pose)
        if pose.points is not None and pose.scores is not None:
            self._edges(canvas, pose)
            self._points(canvas, pose)
        if self.mirror:
            canvas[:] = canvas[:, ::-1]  # numpy copies first: overlap-safe
        if pose.box is not None and pose.box_score is not None:
            self._box_label(canvas, pose)
        self._hud(canvas, hud)

    def _box(self, canvas, pose: PoseAt) -> None:
        x, y, w, h = pose.box
        p0 = self.geometry.point(x, y)
        p1 = self.geometry.point(x + w, y + h)
        cv2.rectangle(canvas, p0, p1, COLOR_BOX, 1, cv2.LINE_AA)

    def _box_label(self, canvas, pose: PoseAt) -> None:
        # Above the box's top-left corner as displayed: after the flip that
        # corner is where the top-right one was.
        x, y, w, h = pose.box
        p0 = self.geometry.point(x, y)
        left = p0[0]
        if self.mirror:
            left = canvas.shape[1] - 1 - self.geometry.point(x + w, y + h)[0]
        cv2.putText(canvas, f"RT-DETRv4 {pose.box_score:.2f}",
                    (max(0, left), max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOR_BOX, 1, cv2.LINE_AA)

    def _edges(self, canvas, pose: PoseAt) -> None:
        points, scores = pose.points, pose.scores
        for a, b in BODY_EDGES:
            if a >= len(points) or b >= len(points):
                continue
            if scores[a] < self.threshold or scores[b] < self.threshold:
                continue
            cv2.line(canvas, self.geometry.point(*points[a]),
                     self.geometry.point(*points[b]), group_color(a), 2,
                     cv2.LINE_AA)

    def _points(self, canvas, pose: PoseAt) -> None:
        points, scores = pose.points, pose.scores
        hollow = pose.state == "interpolated"
        for index in range(len(points)):
            score = float(scores[index])
            if score < self.faint:
                continue
            center = self.geometry.point(*points[index])
            if score < self.threshold:
                cv2.circle(canvas, center, 1, COLOR_FAINT, -1, cv2.LINE_AA)
                continue
            radius = group_radius(index)
            if index in BODY:
                cv2.circle(canvas, center, radius + 2, COLOR_BOX, 1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, group_color(index),
                       1 if (hollow and index in BODY) else -1, cv2.LINE_AA)

    def _hud(self, canvas, lines: list[tuple[str, tuple]]) -> None:
        if not lines:
            return
        scale, line_h = 0.52, 21
        height = 9 + line_h * len(lines)
        width = min(canvas.shape[1], 12 + max(
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            for text, _ in lines))
        roi = canvas[0:height, 0:width]
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
        for row, (text, color) in enumerate(lines):
            cv2.putText(canvas, text, (6, 16 + row * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# -- the encoder ---------------------------------------------------------------


def probe_nvenc(run=subprocess.run) -> bool:
    """Whether this ffmpeg can open h264_nvenc here - the encoder library
    rides the host driver, which Cloud Run's GPU images may not mount."""
    try:
        result = run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=black:s=256x256:r=15", "-frames:v", "3",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class OverlayPublisher:
    """Frames in over a pipe, H.264 out to the relay over RTSP.

    x264 ultrafast/zerolatency at 720p30 is about one core (720p15 was
    well under one) and needs nothing from the driver; NVENC is opt-in
    behind `probe_nvenc`. The bit rate is CBR with a one-second buffer:
    DEFAULT_BITRATE at DEFAULT_FPS is 200 kbit a frame, what 3M was at 15
    fps, so a frame is coded as well as it was when the view showed half
    as many. A publisher that dies (relay restart, encoder crash) is
    respawned with backoff on the next write; frames offered while it is
    down are dropped and counted. `close` sends EOF and waits so a
    recording finalises.

    With more `renditions` than `hi` (RENDITIONS) the one process encodes
    each from the same frames through one filter graph - a `split`, one
    `fps` for the slower rungs, one `scale` for the smaller ones - and
    publishes each to its own relay path (`overlay-half`, ...), so the
    renderer still writes one frame a tick and a crash is still one
    respawn. About 2.1 cores for the four: 720p30 ~1.0, 720p15 ~0.5, two
    540p15 ~0.3 each. The recording tee stays on `hi`.
    """

    BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 10.0)

    def __init__(self, url: str, size: tuple[int, int], fps: float,
                 encoder: str = "x264", bitrate: str = DEFAULT_BITRATE,
                 record_path: str | Path | None = None, telemetry=None,
                 popen=subprocess.Popen, clock=time.monotonic,
                 renditions: tuple[str, ...] = DEFAULT_RENDITIONS):
        self.url = url
        self.size = size
        self.fps = fps
        self.encoder = encoder
        self.bitrate = bitrate
        self.record_path = Path(record_path) if record_path else None
        self.telemetry = telemetry
        self.popen = popen
        self.clock = clock
        self.renditions = parse_renditions(",".join(renditions))
        self.proc = None
        self.spawns = 0
        self.failures = 0
        self._retry_at = 0.0
        self._lock = threading.Lock()

    # -- the command --------------------------------------------------------

    def record_file(self) -> Path | None:
        if self.record_path is None:
            return None
        if self.spawns <= 1:
            return self.record_path
        # A restart must not truncate what the first process wrote.
        return self.record_path.with_name(
            f"{self.record_path.stem}.{self.spawns}{self.record_path.suffix}")

    def rendition_url(self, name: str) -> str:
        """Where a rendition is published: the canvas URL with the suffix."""
        return self.url + RENDITIONS[name].suffix

    def rendition_table(self) -> list[dict]:
        """The renditions as the status and the trainer's snapshot list
        them: name, size, fps, bitrate and the relay path the phone dials
        (`overlay-half`, from the publish URL's last segment)."""
        base = self.url.rstrip("/").rsplit("/", 1)[-1] or "overlay"
        table = []
        for name in self.renditions:
            rendition = RENDITIONS[name]
            width, height = rendition.size(self.size)
            table.append({"name": name, "size": f"{width}x{height}",
                          "fps": rendition.rate(self.fps),
                          "bitrate": rendition.bits(self.bitrate),
                          "path": rendition.path(base)})
        return table

    def filter_graph(self) -> str:
        """The `-filter_complex` that feeds every rendition from the one
        input: `[0:v]split=2[hi][rest];[rest]fps=15,split=2[half][sm];
        [sm]scale=W:H:flags=area,split=2[small][lean]` for all four,
        the same graph with the unused branches left out for fewer."""
        slower = [name for name in self.renditions if name != "hi"]
        if not slower:
            return "[0:v]null[hi]"
        graph = ["[0:v]split=2[hi][rest]"]
        smaller = [name for name in slower if RENDITIONS[name].scale != 1.0]
        full = [name for name in slower if RENDITIONS[name].scale == 1.0]
        outs = [f"[{name}]" for name in full] + (["[sm]"] if smaller else [])
        graph.append(f"[rest]fps={RENDITION_FPS:g}"
                     + (f",split={len(outs)}" if len(outs) > 1 else "")
                     + "".join(outs))
        if smaller:
            width, height = RENDITIONS[smaller[0]].size(self.size)
            outs = [f"[{name}]" for name in smaller]
            graph.append(f"[sm]scale={width}:{height}:flags=area"
                         + (f",split={len(outs)}" if len(outs) > 1 else "")
                         + "".join(outs))
        return ";".join(graph)

    def _codec(self, gop: int, bitrate: str) -> list[str]:
        """One output's encoder block: the codec, its keyframe interval and
        its CBR bit rate."""
        if self.encoder == "nvenc":
            argv = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                    "-zerolatency", "1", "-rc", "cbr"]
        else:
            argv = ["-c:v", "libx264", "-preset", "ultrafast",
                    "-tune", "zerolatency", "-x264-params",
                    f"keyint={gop}:min-keyint={gop}:scenecut=0:repeat-headers=1"]
        # global_header is load-bearing for the recording: without extradata
        # the mp4 muxer writes Annex-B samples into an avc1 track and the
        # file is unreadable (seen 2026-09-04). The RTSP side still carries
        # SPS/PPS in-band (repeat-headers) and in the SDP. 1200-byte RTP
        # packets fit the relay's WebRTC MTU without remuxing.
        argv += ["-profile:v", "baseline", "-pix_fmt", "yuv420p",
                 "-g", str(gop), "-bf", "0", "-flags", "+global_header",
                 "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
        return argv

    def _sink(self, url: str, record: Path | None, mapping: list[str]) -> list[str]:
        """One output's muxer: RTSP to the relay, or a tee of that and the
        mp4 recording."""
        if record is None:
            return mapping + ["-f", "rtsp", "-rtsp_transport", "tcp",
                              "-pkt_size", "1200", url]
        return ["-f", "tee"] + mapping + [
            f"[f=rtsp:rtsp_transport=tcp:pkt_size=1200]{url}|"
            f"[f=mp4:movflags=+frag_keyframe+empty_moov+default_base_moof"
            f":onfail=ignore]{record}"]

    def argv(self) -> list[str]:
        width, height = self.size
        gop = max(1, int(round(self.fps * 2)))
        argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", f"{self.fps:g}", "-i", "pipe:0", "-an"]
        record = self.record_file()
        if self.renditions == ("hi",):
            # Today's command, unchanged: the encoder straight off the input.
            argv += self._codec(gop, self.bitrate)
            argv += self._sink(self.url, record, ["-map", "0:v"] if record else [])
            return argv
        argv += ["-filter_complex", self.filter_graph()]
        for name in self.renditions:
            rendition = RENDITIONS[name]
            rate = rendition.rate(self.fps)
            argv += self._codec(max(1, int(round(rate * 2))),
                                rendition.bits(self.bitrate))
            argv += self._sink(self.rendition_url(name),
                               record if name == "hi" else None,
                               ["-map", f"[{name}]"])
        return argv

    # -- the process --------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _spawn(self) -> bool:
        self.spawns += 1
        try:
            self.proc = self.popen(self.argv(), stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL)
        except OSError as error:
            self.proc = None
            self._failed(f"spawn failed: {error!r}")
            return False
        return True

    def _failed(self, why: str) -> None:
        self.failures += 1
        backoff = self.BACKOFF_S[min(self.failures, len(self.BACKOFF_S)) - 1]
        self._retry_at = self.clock() + backoff
        if self.telemetry is not None:
            self.telemetry.count("overlayPublisherRestarts")
        print(f"overlay publisher: {why}; retry in {backoff:.0f}s", flush=True)
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.proc = None

    def write(self, bgr: np.ndarray) -> bool:
        """One frame to the encoder. False when it was dropped."""
        with self._lock:
            if not self.alive:
                if self.proc is not None:
                    self._failed(f"ffmpeg exited {self.proc.returncode}")
                if self.clock() < self._retry_at or not self._spawn():
                    return False
            try:
                # A flat view, no copy: the canvas is C-contiguous.
                self.proc.stdin.write(
                    memoryview(np.ascontiguousarray(bgr)).cast("B"))
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                self._failed(f"write failed: {error!r}")
                return False
            self.failures = 0
            return True

    def close(self, timeout_s: float = 10.0) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


# -- the renderer --------------------------------------------------------------


class OverlayRenderer(threading.Thread):
    """Frames and detections in from the pipeline's threads, painted frames
    out to the publisher on a fixed tick.

    With `sync` (sync.ViewSync) the view is two cameras': the frames and
    detections offered through `offer_frame`/`offer_pose` are the body
    view's and fill the canvas, and those through `offer_face_frame`/
    `offer_face_pose` are the face view's, drawn as an inset in the
    top-right corner (INSET_*) with its own keypoints. Each tick pairs the
    body frame it draws with the face frame at the same moment on the
    shared clock; a face frame further than the pairing tolerance leaves
    the inset out for that tick rather than show the wrong moment. The
    inset's crop follows the face (face_crop_target) and is mirrored when
    `face_mirror` says the phone's camera faces the user (set_mirror); the
    body picture is never mirrored in this layout.
    """

    def __init__(self, publisher: OverlayPublisher, telemetry,
                 size: tuple[int, int] = (1280, 720), fps: float = DEFAULT_FPS,
                 delay_s: float = 1.0, source_fps: float = 30.0,
                 clock=time.monotonic, capacity: int | None = None,
                 mirror: bool = False, sync=None, face_mirror: bool = False):
        super().__init__(daemon=True, name="overlay")
        self.publisher = publisher
        self.telemetry = telemetry
        self.size = size
        self.fps = fps
        self.delay_s = delay_s
        self.source_fps = source_fps
        self.sync = sync
        self.mirror = mirror and sync is None
        self.clock = clock
        # Enough decoded frames to still hold the one `delay_s` behind the
        # head after a tick's worth of jitter; at 4K each is ~12 MB.
        window = capacity or int((delay_s + 0.5) * source_fps) + 8
        self.frames = FrameStore(window)
        self.track = PoseTrack(frame_s=1.0 / source_fps)
        # The face view's, when there is one: the same window, plus the
        # pairing tolerance either way.
        self.face_frames = FrameStore(
            window + (int(2 * sync.tolerance_s * source_fps) if sync else 0))
        self.face_track = PoseTrack(frame_s=1.0 / source_fps)
        self.face_mirror = bool(face_mirror)
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._context: dict[str, object] = {}
        self._geometry: Geometry | None = None
        self._painter: KeypointPainter | None = None
        self._inset = inset_rect(*size) if sync is not None else None
        self._crop: tuple[float, float, float] | None = None  # cx, cy, h, smoothed
        self._crop_frame: tuple[int, int] | None = None
        self._face_state = "none"
        self._face_age_s: float | None = None
        self._last_index = -1
        self._last_canvas: np.ndarray | None = None
        self._last_state = "none"
        self._stats: dict = {}
        self._stats_at = float("-inf")

    # -- inputs, from other threads ---------------------------------------------

    def offer_frame(self, index: int, at_s: float, yuv: np.ndarray) -> None:
        # Eviction is steady state: the store is a window over the decode,
        # and at 30 fps out of 30 every frame is shown once (at 15 every
        # other one was skipped by design). What would be a fault is the
        # window not holding the frame the view wants, and tick() counts
        # that as overlayTargetMissed; a `duplicate` tick at 30 fps means
        # the decode stalled, not that the cadences differ.
        frame = Frame(index, at_s, yuv, self.clock())
        with self._lock:
            self.frames.offer(frame)

    def offer_pose(self, index: int, at_s: float, keypoints, scores, box,
                   box_score, people: int, unresolved: bool) -> None:
        detection = Detection(
            index=index, at_s=at_s,
            points=to_array(keypoints, 2), scores=to_array(scores, 1),
            box=[float(v) for v in box] if box is not None else None,
            box_score=float(box_score) if box_score is not None else None,
            people=int(people), unresolved=bool(unresolved), wall=self.clock(),
        )
        with self._lock:
            self.track.offer(detection)

    def offer_face_frame(self, index: int, at_s: float, yuv: np.ndarray) -> None:
        """A decoded frame of the face view, on its own media time."""
        frame = Frame(index, at_s, yuv, self.clock())
        with self._lock:
            self.face_frames.offer(frame)

    def offer_face_pose(self, index: int, at_s: float, keypoints, scores, box,
                        box_score, people: int, unresolved: bool) -> None:
        """The face view's full pose result, as offer_pose for the body."""
        detection = Detection(
            index=index, at_s=at_s,
            points=to_array(keypoints, 2), scores=to_array(scores, 1),
            box=[float(v) for v in box] if box is not None else None,
            box_score=float(box_score) if box_score is not None else None,
            people=int(people), unresolved=bool(unresolved), wall=self.clock(),
        )
        with self._lock:
            self.face_track.offer(detection)

    def set_mirror(self, mirror: bool) -> None:
        """Whether the face inset is drawn as a mirror (the phone's camera
        faces the user). Takes effect on the next tick."""
        with self._lock:
            self.face_mirror = bool(mirror)

    def offer_context(self, kind: str, body) -> None:
        """Session context for the HUD: `hud` is the analysis process's list
        of text lines (analysis/protocol.md)."""
        with self._lock:
            self._context[kind] = body

    # -- the tick --------------------------------------------------------------

    def tick(self) -> str:
        """One output-frame decision: idle | duplicate | emitted."""
        with self._lock:
            head = self.frames.head_at_s()
            if head is None:
                return "idle"
            target = head - self.delay_s
            frame = self.frames.nearest(target)
            pose = self.track.at(frame.at_s, frame.index)
            context = dict(self._context)
            pose_rate = self.track.rate_per_s(self.clock())
        if target >= 0 and abs(frame.at_s - target) > 1.0 / self.source_fps:
            # The window did not hold the frame the view wanted: it was
            # evicted (store too small for the delay) or never decoded.
            self._count("overlayTargetMissed")
        if frame.index <= self._last_index:
            # The source stalled or the tick outran the decode: the encoder
            # gets the same picture again so its clock keeps running.
            if self._last_canvas is not None and self.publisher.write(self._last_canvas):
                self._count("overlayDuplicated")
            return "duplicate"
        with self.telemetry.time_stage("overlayRender") if self.telemetry else _Null():
            canvas = self._paint(frame, pose, context, pose_rate)
        with self.telemetry.time_stage("overlayWrite") if self.telemetry else _Null():
            written = self.publisher.write(canvas)
        self._last_index = frame.index
        self._last_canvas = canvas
        self._last_state = pose.state
        self._count("overlayFrames" if written else "overlayWriteDropped")
        if pose.state == "stale":
            self._count("overlayStale")
        if self.telemetry is not None:
            self.telemetry.gauge("overlayLagS", self.clock() - frame.wall)
        return "emitted"

    def _count(self, name: str) -> None:
        if self.telemetry is not None:
            self.telemetry.count(name)

    def _paint(self, frame: Frame, pose: PoseAt, context: dict,
               pose_rate: float) -> np.ndarray:
        width = frame.yuv.shape[1]
        height = frame.yuv.shape[0] * 2 // 3
        if (self._geometry is None or self._geometry.src_w != width
                or self._geometry.src_h != height):
            self._geometry = fit_geometry(width, height, *self.size)
            self._painter = KeypointPainter(self._geometry, mirror=self.mirror)
        canvas = downscale_i420(frame.yuv, self._geometry)
        if self.sync is not None:
            # The inset goes on before the HUD, so the HUD is never under it.
            self._paint_inset(canvas, frame)
        self._painter.paint(canvas, pose, self._hud(frame, pose, context, pose_rate))
        return canvas

    # -- the face inset ---------------------------------------------------------

    def _face_for(self, body_frame: Frame
                  ) -> tuple[Frame | None, PoseAt | None, str, bool]:
        """The face frame at the body frame's moment, its pose, why there
        is none - `waiting` (the clocks are not both placed yet),
        `unpaired` (no face frame within the tolerance) - or the pose's
        state, and whether the inset is a mirror."""
        face_at = self.sync.face_at_s(body_frame.at_s)
        if face_at is None:
            return None, None, "waiting", False
        with self._lock:
            face_frame = self.face_frames.nearest(face_at)
            if face_frame is None or not self.sync.paired(face_at, face_frame.at_s):
                return None, None, "unpaired", False
            face_pose = self.face_track.at(face_frame.at_s, face_frame.index)
            mirror = self.face_mirror
        self._face_age_s = face_frame.at_s - face_at
        return face_frame, face_pose, face_pose.state, mirror

    def _paint_inset(self, canvas: np.ndarray, body_frame: Frame) -> None:
        face_frame, face_pose, state, mirror = self._face_for(body_frame)
        self._face_state = state
        if face_frame is None:
            self._count("overlayFaceWaiting" if state == "waiting" else "overlayFaceUnpaired")
            return
        frame_w = face_frame.yuv.shape[1]
        frame_h = face_frame.yuv.shape[0] * 2 // 3
        target = face_crop_target(face_pose.points, face_pose.scores, frame_w, frame_h)
        if self._crop is None or self._crop_frame != (frame_w, frame_h):
            self._crop = target
            self._crop_frame = (frame_w, frame_h)
        else:
            self._crop = tuple(
                old + (new - old) * INSET_SMOOTH for old, new in zip(self._crop, target))
        box = crop_box(*self._crop, frame_w, frame_h)
        x, y, w, h = self._inset
        fit = fit_geometry(box[2], box[3], w, h)
        inset = downscale_i420(crop_i420(face_frame.yuv, box), fit)
        painter = KeypointPainter(CropGeometry(fit, box[0], box[1]), mirror=mirror)
        painter.paint(inset, face_pose, [])
        if state == "stale":
            self._count("overlayFaceStale")
        # A border, then the inset over the body picture.
        b = INSET_BORDER
        y0, y1 = max(0, y - b), min(canvas.shape[0], y + h + b)
        x0, x1 = max(0, x - b), min(canvas.shape[1], x + w + b)
        canvas[y0:y1, x0:x1] = COLOR_INSET_BORDER
        canvas[y:y + h, x:x + w] = inset

    def _hud(self, frame: Frame, pose: PoseAt, context: dict,
             pose_rate: float) -> list[tuple[str, tuple]]:
        now = self.clock()
        if self.telemetry is not None and now - self._stats_at >= 0.5:
            snap = self.telemetry.snapshot()
            self._stats = {
                "poseP50": snap["stagesMs"].get("pose", {}).get("p50"),
                "dropped": snap["counters"].get("poseDropped", 0),
                "e2eLagS": snap["gauges"].get("e2eLagS"),
            }
            self._stats_at = now
        stats = self._stats
        p50 = stats.get("poseP50")
        lines: list[tuple[str, tuple]] = []
        # `timing` is how the decoder timed the frames (producer.Decoder):
        # the sender's clock, or arrival here.
        timing = context.get("timing")
        lines.append((
            f"t {frame.at_s:7.1f}s   pose {pose_rate:.1f}/s"
            + (f" p50 {p50:.0f}ms" if p50 is not None else "")
            + f" drops {stats.get('dropped', 0)}"
            + f"   view {self.delay_s:.1f}s behind decode, aligned"
            + (f"   clock {timing}" if isinstance(timing, str) and timing else ""),
            COLOR_TEXT))
        if pose.points is not None and pose.scores is not None:
            confident = int((pose.scores >= MIN_KEYPOINT_SCORE).sum())
            detail = (f"Sapiens2 {confident}/{len(pose.scores)} kp >= "
                      f"{MIN_KEYPOINT_SCORE:.2f}")
            if pose.box_score is not None:
                detail += f"   RT-DETRv4 {pose.box_score:.2f}"
            detail += f"   people {pose.people}"
            if pose.unresolved:
                detail += " (identity unresolved)"
            if pose.state == "interpolated":
                detail += "   interpolated"
            elif pose.state == "stale":
                detail += f"   POSE LAGGING +{pose.age_s * 1000:.0f}ms"
            lines.append((detail, COLOR_WARN if pose.state == "stale" else COLOR_TEXT))
        elif pose.state == "missing":
            lines.append((f"NO PERSON  (detector saw {pose.people})", COLOR_BAD))
        else:
            lines.append(("waiting for the first pose", COLOR_WARN))
        if self.sync is not None:
            lines.append(self._face_hud_line())
        # Whatever the analysis process last said about the session
        # (analysis/protocol.md `hud`), drawn verbatim under the pose lines:
        # this renderer knows nothing about what the lines mean.
        for text in context.get("hud") or ():
            if isinstance(text, str) and text:
                lines.append((text, COLOR_TEXT))
        return lines

    def _face_hud_line(self) -> tuple[str, tuple]:
        """The inset's line: what it shows, how the two views stand."""
        state = self._face_state
        skew = self.sync.skew_s()
        stand = (f"skew {skew * 1000:+.0f}ms" if skew is not None else "clocks apart")
        if state == "waiting":
            return f"face: waiting for the phone's camera   {stand}", COLOR_WARN
        if state == "unpaired":
            return f"face: no frame at this moment   {stand}", COLOR_WARN
        if state == "none":
            return f"face: waiting for the first pose   {stand}", COLOR_WARN
        if state == "missing":
            return f"face: NO PERSON   {stand}", COLOR_BAD
        pairing = (f" paired {self._face_age_s * 1000:+.0f}ms"
                   if self._face_age_s is not None else "")
        color = COLOR_WARN if state == "stale" else COLOR_TEXT
        return f"face: {state}{pairing}   {stand}   {self.sync.timing}", color

    # -- the thread -------------------------------------------------------------

    def run(self) -> None:
        period = 1.0 / self.fps
        due = self.clock()
        while not self.stopping.is_set():
            now = self.clock()
            if now < due:
                self.stopping.wait(min(due - now, period))
                continue
            due += period
            if now - due > 4 * period:
                due = now + period  # far behind: resync, do not burst
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - said aloud, survived
                self._count("overlayErrors")
                print(f"overlay tick failed: {error!r}", flush=True)

    def close(self, timeout_s: float = 5.0) -> None:
        self.stopping.set()
        if self.is_alive():
            self.join(timeout=timeout_s)
        self.publisher.close()

    def snapshot(self) -> dict:
        with self._lock:
            stored, detections = len(self.frames), len(self.track)
            face_stored, face_detections = len(self.face_frames), len(self.face_track)
            timing = self._context.get("timing")
            face_mirror = self.face_mirror
        skew = self.sync.skew_s() if self.sync is not None else None
        return {
            "publishUrl": self.publisher.url,
            "encoder": self.publisher.encoder,
            "size": f"{self.size[0]}x{self.size[1]}",
            "fps": self.fps,
            "delayS": self.delay_s,
            "publisherAlive": self.publisher.alive,
            "publisherSpawns": self.publisher.spawns,
            "recordPath": (str(self.publisher.record_path)
                           if self.publisher.record_path else None),
            # Every encoding of the view the relay carries, the phone's to
            # pick from by `path` (relay_proxy.route): `hi` is this canvas.
            "renditions": self.publisher.rendition_table(),
            "framesStored": stored,
            "detections": detections,
            "lastState": self._last_state,
            "lastIndex": self._last_index,
            "timing": timing if isinstance(timing, str) else None,
            # The layout: `single` is one camera filling the view; `inset`
            # is the fixed camera's with the phone's in the corner, where
            # `inset` says (x, y, w, h on the canvas) and `mirror` how.
            "view": {
                "layout": "inset" if self.sync is not None else "single",
                "inset": list(self._inset) if self._inset else None,
                "faceState": self._face_state if self.sync is not None else None,
                "faceFramesStored": face_stored,
                "faceDetections": face_detections,
                "skewMs": round(skew * 1000) if skew is not None else None,
                "mirror": face_mirror,
                "timing": self.sync.timing if self.sync is not None else None,
            },
        }


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def build_renderer(args, telemetry, source_fps: float = 30.0,
                   record_path: str | Path | None = None,
                   probe=probe_nvenc, sync=None) -> OverlayRenderer | None:
    """The renderer for a session's args, or None when no publish URL is
    configured (the default: zero overhead). `sync` (sync.ViewSync) makes
    it the two-camera layout, the face inset mirrored as `args.face_mirror`
    says to begin with."""
    url = getattr(args, "overlay_publish", "") or ""
    if not url:
        return None
    size = parse_size(getattr(args, "overlay_size", "1280x720") or "1280x720")
    fps = float(getattr(args, "overlay_fps", DEFAULT_FPS) or DEFAULT_FPS)
    encoder = getattr(args, "overlay_encoder", "x264") or "x264"
    if encoder == "auto":
        encoder = "nvenc" if probe() else "x264"
        print(f"overlay: encoder auto -> {encoder}", flush=True)
    publisher = OverlayPublisher(
        url, size, fps, encoder=encoder,
        bitrate=getattr(args, "overlay_bitrate", DEFAULT_BITRATE) or DEFAULT_BITRATE,
        record_path=record_path, telemetry=telemetry,
        renditions=parse_renditions(getattr(args, "overlay_renditions", "") or ""))
    return OverlayRenderer(
        publisher, telemetry, size=size, fps=fps,
        delay_s=float(getattr(args, "overlay_delay_s", 1.0) or 1.0),
        source_fps=source_fps,
        mirror=bool(getattr(args, "overlay_mirror", False)),
        sync=sync, face_mirror=bool(getattr(args, "face_mirror", False)))
