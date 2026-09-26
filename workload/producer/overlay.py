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
# order of magnitude (1.3 x the pose interval at 6 fps is 0.22 s; a drop
# or two widens it) without ever bridging a whole lost second.
BRIDGE_S = 1.0
SIZE_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")


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

    x264 ultrafast/zerolatency at 720p15 is well under one core and needs
    nothing from the driver; NVENC is opt-in behind `probe_nvenc`. A
    publisher that dies (relay restart, encoder crash) is respawned with
    backoff on the next write; frames offered while it is down are dropped
    and counted. `close` sends EOF and waits so a recording finalises.
    """

    BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 10.0)

    def __init__(self, url: str, size: tuple[int, int], fps: float,
                 encoder: str = "x264", bitrate: str = "3M",
                 record_path: str | Path | None = None, telemetry=None,
                 popen=subprocess.Popen, clock=time.monotonic):
        self.url = url
        self.size = size
        self.fps = fps
        self.encoder = encoder
        self.bitrate = bitrate
        self.record_path = Path(record_path) if record_path else None
        self.telemetry = telemetry
        self.popen = popen
        self.clock = clock
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

    def argv(self) -> list[str]:
        width, height = self.size
        gop = max(1, int(round(self.fps * 2)))
        argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", f"{self.fps:g}", "-i", "pipe:0", "-an"]
        if self.encoder == "nvenc":
            argv += ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                     "-zerolatency", "1", "-rc", "cbr"]
        else:
            argv += ["-c:v", "libx264", "-preset", "ultrafast",
                     "-tune", "zerolatency", "-x264-params",
                     f"keyint={gop}:min-keyint={gop}:scenecut=0:repeat-headers=1"]
        # global_header is load-bearing for the recording: without extradata
        # the mp4 muxer writes Annex-B samples into an avc1 track and the
        # file is unreadable (seen 2026-09-04). The RTSP side still carries
        # SPS/PPS in-band (repeat-headers) and in the SDP. 1200-byte RTP
        # packets fit the relay's WebRTC MTU without remuxing.
        argv += ["-profile:v", "baseline", "-pix_fmt", "yuv420p",
                 "-g", str(gop), "-bf", "0", "-flags", "+global_header",
                 "-b:v", self.bitrate, "-maxrate", self.bitrate,
                 "-bufsize", self.bitrate]
        record = self.record_file()
        if record is None:
            argv += ["-f", "rtsp", "-rtsp_transport", "tcp",
                     "-pkt_size", "1200", self.url]
        else:
            argv += ["-f", "tee", "-map", "0:v",
                     f"[f=rtsp:rtsp_transport=tcp:pkt_size=1200]{self.url}|"
                     f"[f=mp4:movflags=+frag_keyframe+empty_moov+default_base_moof"
                     f":onfail=ignore]{record}"]
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
    out to the publisher on a fixed tick."""

    def __init__(self, publisher: OverlayPublisher, telemetry,
                 size: tuple[int, int] = (1280, 720), fps: float = 15.0,
                 delay_s: float = 1.0, source_fps: float = 30.0,
                 clock=time.monotonic, capacity: int | None = None,
                 mirror: bool = False):
        super().__init__(daemon=True, name="overlay")
        self.publisher = publisher
        self.telemetry = telemetry
        self.size = size
        self.fps = fps
        self.delay_s = delay_s
        self.source_fps = source_fps
        self.mirror = mirror
        self.clock = clock
        # Enough decoded frames to still hold the one `delay_s` behind the
        # head after a tick's worth of jitter; at 4K each is ~12 MB.
        self.frames = FrameStore(capacity or int((delay_s + 0.5) * source_fps) + 8)
        self.track = PoseTrack(frame_s=1.0 / source_fps)
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._context: dict[str, object] = {}
        self._geometry: Geometry | None = None
        self._painter: KeypointPainter | None = None
        self._last_index = -1
        self._last_canvas: np.ndarray | None = None
        self._last_state = "none"
        self._stats: dict = {}
        self._stats_at = float("-inf")

    # -- inputs, from other threads ---------------------------------------------

    def offer_frame(self, index: int, at_s: float, yuv: np.ndarray) -> None:
        # Eviction is steady state (the store is a window, and at 15 out
        # of 30 fps half the frames are skipped by design); what would be
        # a fault is the window not holding the frame the view wants, and
        # tick() counts that as overlayTargetMissed.
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
        self._painter.paint(canvas, pose, self._hud(frame, pose, context, pose_rate))
        return canvas

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
        lines.append((
            f"t {frame.at_s:7.1f}s   pose {pose_rate:.1f}/s"
            + (f" p50 {p50:.0f}ms" if p50 is not None else "")
            + f" drops {stats.get('dropped', 0)}"
            + f"   view {self.delay_s:.1f}s behind decode, aligned",
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
        # Whatever the analysis process last said about the session
        # (analysis/protocol.md `hud`), drawn verbatim under the pose lines:
        # this renderer knows nothing about what the lines mean.
        for text in context.get("hud") or ():
            if isinstance(text, str) and text:
                lines.append((text, COLOR_TEXT))
        return lines

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
            "framesStored": stored,
            "detections": detections,
            "lastState": self._last_state,
            "lastIndex": self._last_index,
        }


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def build_renderer(args, telemetry, source_fps: float = 30.0,
                   record_path: str | Path | None = None,
                   probe=probe_nvenc) -> OverlayRenderer | None:
    """The renderer for a session's args, or None when no publish URL is
    configured (the default: zero overhead)."""
    url = getattr(args, "overlay_publish", "") or ""
    if not url:
        return None
    size = parse_size(getattr(args, "overlay_size", "1280x720") or "1280x720")
    fps = float(getattr(args, "overlay_fps", 15.0) or 15.0)
    encoder = getattr(args, "overlay_encoder", "x264") or "x264"
    if encoder == "auto":
        encoder = "nvenc" if probe() else "x264"
        print(f"overlay: encoder auto -> {encoder}", flush=True)
    publisher = OverlayPublisher(
        url, size, fps, encoder=encoder,
        bitrate=getattr(args, "overlay_bitrate", "3M") or "3M",
        record_path=record_path, telemetry=telemetry)
    return OverlayRenderer(
        publisher, telemetry, size=size, fps=fps,
        delay_s=float(getattr(args, "overlay_delay_s", 1.0) or 1.0),
        source_fps=source_fps,
        mirror=bool(getattr(args, "overlay_mirror", False)))
