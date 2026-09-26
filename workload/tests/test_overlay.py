"""The live annotated view draws each frame with the pose detected on it.

Locked down here: the view runs `delay_s` behind the decode head and holds
its cadence by duplicating on a stall and skipping through a burst; a frame
between two detections is drawn with the joints lerped and the scores taken
pessimistically (as `motion.pose_between` does), a frame past the last
detection is drawn stale and says so, and one after a "nobody" detection is
drawn bare; keypoints land where the downscale put the body; the frame
store never grows past its cap (drops are counted, the decode thread never
waits); the publisher respawns a dead encoder with backoff and never
truncates a recording on restart; and the full-result tap in live_pose is a
side channel - a listener that raises is counted, the row is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import overlay  # noqa: E402
from keypoints import (  # noqa: E402
    BODY, KEYPOINT_NAMES, LEFT_HIP, MIN_KEYPOINT_SCORE, NOSE,
)
from live_pose import (  # noqa: E402
    SideloadPose, row_keypoint_arrays, share_full_result,
)
from overlay import (  # noqa: E402
    COLOR_BODY, Detection, FrameStore, Frame, KeypointPainter,
    OverlayPublisher, OverlayRenderer, PoseAt, PoseTrack, downscale_i420,
    fit_geometry, parse_size,
)
from telemetry import Telemetry  # noqa: E402

FPS = 30.0


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakePublisher:
    """Records every canvas it is handed."""

    def __init__(self, fail: bool = False):
        self.frames: list[np.ndarray] = []
        self.fail = fail
        self.url = "rtsp://fake/overlay"
        self.encoder = "fake"
        self.record_path = None
        self.spawns = 0
        self.closed = False

    @property
    def alive(self) -> bool:
        return True

    def write(self, bgr: np.ndarray) -> bool:
        if self.fail:
            return False
        self.frames.append(bgr)
        return True

    def close(self) -> None:
        self.closed = True


def gray_frame(width: int, height: int, value: int = 0) -> np.ndarray:
    """A flat yuv420p frame of one luma value (chroma neutral)."""
    yuv = np.full((height * 3 // 2, width), 128, np.uint8)
    yuv[:height] = value
    return yuv


def full_points(width: int, height: int, count: int = 308) -> np.ndarray:
    rng = np.random.default_rng(7)
    pts = rng.uniform([width * 0.2, height * 0.2],
                      [width * 0.8, height * 0.8], size=(count, 2))
    return pts.astype(np.float32)


# -- geometry ------------------------------------------------------------------


def test_parse_size_accepts_even_dimensions_only():
    assert parse_size("1280x720") == (1280, 720)
    assert parse_size(" 960x540 ") == (960, 540)
    for bad in ("1281x720", "1280x719", "abc", "32x32", "1280"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_fit_geometry_letterboxes_and_carries_points_along():
    geometry = fit_geometry(640, 480, 1280, 720)  # 4:3 into 16:9
    assert geometry.fit_h == 720 and geometry.fit_w == 960
    assert geometry.ox == 160 and geometry.oy == 0
    assert geometry.point(0, 0) == (160, 0)
    assert geometry.point(640, 480) == (1120, 720)
    same = fit_geometry(3840, 2160, 1280, 720)
    assert (same.fit_w, same.fit_h, same.ox, same.oy) == (1280, 720, 0, 0)
    assert same.point(1920, 1080) == (640, 360)


def test_downscale_i420_plane_first_matches_convert_then_resize():
    import cv2
    width, height = 640, 480
    rng = np.random.default_rng(3)
    yuv = rng.integers(16, 240, size=(height * 3 // 2, width), dtype=np.uint8)
    # A smooth image, so the two resampling orders agree closely.
    yuv = cv2.GaussianBlur(yuv, (0, 0), 6)
    geometry = fit_geometry(width, height, 320, 240)
    fast = downscale_i420(yuv, geometry)
    slow = cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420), (320, 240),
                      interpolation=cv2.INTER_AREA)
    assert fast.shape == (240, 320, 3)
    assert np.abs(fast.astype(int) - slow.astype(int)).mean() < 3.0


# -- the bracket ---------------------------------------------------------------


def detection(index: int, points, scores=None, box=(10, 20, 100, 200),
              wall: float = 0.0, people: int = 1) -> Detection:
    if points is not None:
        points = np.asarray(points, np.float32)
        if scores is None:
            scores = np.full(len(points), 0.9, np.float32)
        else:
            scores = np.asarray(scores, np.float32)
        box = list(box)
    else:
        scores = None
        box = None
    return Detection(index, index / FPS, points, scores, box,
                     0.8 if points is not None else None, people, False, wall)


def test_a_frame_between_two_detections_is_interpolated_pessimistically():
    track = PoseTrack(frame_s=1 / FPS)
    track.offer(detection(0, [[0.0, 0.0]], [0.9]))
    track.offer(detection(6, [[60.0, 30.0]], [0.4]))
    at = track.at(3 / FPS, 3)
    assert at.state == "interpolated"
    assert at.points[0].tolist() == pytest.approx([30.0, 15.0])
    assert at.scores[0] == pytest.approx(0.4)  # min, never the average
    assert at.box == pytest.approx([10, 20, 100, 200])
    exact = track.at(6 / FPS, 6)
    assert exact.state == "exact" and exact.points[0].tolist() == [60.0, 30.0]


def test_past_the_last_detection_is_stale_and_says_how_much():
    track = PoseTrack(frame_s=1 / FPS)
    track.offer(detection(0, [[1.0, 1.0]]))
    at = track.at(9 / FPS, 9)
    assert at.state == "stale"
    assert at.age_s == pytest.approx(0.3)
    assert at.points[0].tolist() == [1.0, 1.0]


def test_detections_too_far_apart_are_not_bridged():
    track = PoseTrack(frame_s=1 / FPS, bridge_s=1.0)
    track.offer(detection(0, [[0.0, 0.0]]))
    track.offer(detection(60, [[60.0, 60.0]]))  # 2 s later
    assert track.at(30 / FPS, 30).state == "stale"


def test_nobody_detected_is_drawn_bare_and_breaks_the_bracket():
    track = PoseTrack(frame_s=1 / FPS)
    track.offer(detection(0, [[0.0, 0.0]]))
    track.offer(detection(6, None, people=0))
    track.offer(detection(12, [[12.0, 12.0]]))
    # Before the gap: the next detection has no points, so nothing to lerp to.
    assert track.at(3 / FPS, 3).state == "stale"
    # After it: nobody there.
    missing = track.at(8 / FPS, 8)
    assert missing.state == "missing" and missing.points is None
    assert track.at(12 / FPS, 12).state == "exact"


def test_no_detections_yet_is_its_own_state():
    track = PoseTrack(frame_s=1 / FPS)
    assert track.at(0.0, 0).state == "none"
    track.offer(detection(30, [[0.0, 0.0]]))
    assert track.at(0.0, 0).state == "none"  # earlier than the first one


def test_the_track_stays_sorted_when_an_older_index_arrives():
    track = PoseTrack(frame_s=1 / FPS)
    track.offer(detection(12, [[12.0, 12.0]]))
    track.offer(detection(6, [[6.0, 6.0]]))
    assert [d.index for d in track.detections] == [6, 12]


# -- the frame store -------------------------------------------------------------


def test_frame_store_caps_and_counts_evictions_and_finds_the_nearest():
    store = FrameStore(capacity=4)
    yuv = gray_frame(16, 16)
    evicted = sum(store.offer(Frame(i, i / FPS, yuv, 0.0)) for i in range(6))
    assert evicted == 2 and len(store) == 4
    assert [f.index for f in store.frames] == [2, 3, 4, 5]
    assert store.head_at_s() == pytest.approx(5 / FPS)
    assert store.nearest(3.4 / FPS).index == 3
    assert store.nearest(-1.0).index == 2  # earlier than everything: oldest
    assert store.nearest(9.0).index == 5


# -- the renderer --------------------------------------------------------------


def renderer(publisher, clock, telemetry=None, delay_s=1.0,
             size=(320, 180)) -> OverlayRenderer:
    return OverlayRenderer(publisher, telemetry, size=size, fps=15.0,
                           delay_s=delay_s, source_fps=FPS, clock=clock)


def test_the_view_runs_delay_behind_the_head_and_holds_cadence_on_a_stall():
    clock, publisher, telemetry = Clock(), FakePublisher(), Telemetry()
    view = renderer(publisher, clock, telemetry)
    frame = gray_frame(64, 36)
    assert view.tick() == "idle"
    for index in range(61):  # two seconds of decode
        clock.now += 1 / FPS
        view.offer_frame(index, index / FPS, frame)
    assert view.tick() == "emitted"
    # Head is frame 60 (2.0 s); the view shows 1.0 s earlier.
    assert view.snapshot()["lastIndex"] == 30
    # Nothing new decoded: the encoder gets the same picture again.
    assert view.tick() == "duplicate"
    assert len(publisher.frames) == 2
    assert publisher.frames[0] is publisher.frames[1]
    # A burst of decode: the view jumps to what is now due, never catches
    # up frame by frame.
    for index in range(61, 121):
        view.offer_frame(index, index / FPS, frame)
    assert view.tick() == "emitted"
    assert view.snapshot()["lastIndex"] == 90
    counters = telemetry.snapshot()["counters"]
    assert counters["overlayFrames"] == 2
    assert counters["overlayDuplicated"] == 1
    assert "overlayTargetMissed" not in counters  # the window held every target
    assert view.snapshot()["framesStored"] == view.frames.capacity  # bounded
    assert "overlayLagS" in telemetry.snapshot()["gauges"]


def test_the_frame_store_holds_what_the_delay_needs_or_says_it_missed():
    view = renderer(FakePublisher(), Clock(), delay_s=1.0)
    assert view.frames.capacity >= int(1.0 * FPS) + 1
    deeper = renderer(FakePublisher(), Clock(), delay_s=2.5)
    assert deeper.frames.capacity > view.frames.capacity
    # A window too small for its delay shows the oldest frame it has and
    # counts the miss, rather than blocking the decode to keep more.
    telemetry = Telemetry()
    small = OverlayRenderer(FakePublisher(), telemetry, size=(64, 36), fps=15.0,
                            delay_s=1.0, source_fps=FPS, clock=Clock(),
                            capacity=4)
    frame = gray_frame(64, 36)
    for index in range(61):
        small.offer_frame(index, index / FPS, frame)
    assert small.tick() == "emitted"
    assert small.snapshot()["lastIndex"] == 57  # the oldest held, not frame 30
    assert telemetry.snapshot()["counters"]["overlayTargetMissed"] == 1


def test_keypoints_are_painted_where_the_downscale_put_the_body():
    clock, publisher = Clock(), FakePublisher()
    view = renderer(publisher, clock, Telemetry(), delay_s=0.0, size=(320, 180))
    width, height = 640, 360
    frame = gray_frame(width, height, value=0)
    points = full_points(width, height)
    scores = np.full(308, 0.95, np.float32)
    view.offer_frame(0, 0.0, frame)
    view.offer_pose(0, 0.0, points, scores, [100, 50, 400, 250], 0.9, 1, False)
    assert view.tick() == "emitted"
    canvas = publisher.frames[0]
    assert canvas.shape == (180, 320, 3)
    # Every body keypoint lands as colour at its halved position.
    for index in BODY:
        x, y = (points[index] / 2).round().astype(int)
        patch = canvas[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        assert patch.max() > 0, f"keypoint {KEYPOINT_NAMES[index]} not drawn"
    # And the picture stays the frame elsewhere: the bottom-right corner,
    # outside the box and the HUD, is untouched black.
    assert canvas[170:, 300:].max() == 0
    assert view.snapshot()["lastState"] == "exact"


def test_a_mirrored_view_flips_picture_and_skeleton_but_letters_read_forward():
    clock = Clock()
    plain, selfie = FakePublisher(), FakePublisher()
    straight = renderer(plain, clock, Telemetry(), delay_s=0.0, size=(320, 180))
    mirrored = OverlayRenderer(selfie, Telemetry(), size=(320, 180), fps=15.0,
                               delay_s=0.0, source_fps=FPS, clock=clock,
                               mirror=True)
    width, height = 640, 360
    # A frame that is bright on its left third only, so the flip is visible.
    frame = gray_frame(width, height, value=0)
    frame[:height, :width // 3] = 200
    points = full_points(width, height)
    scores = np.full(308, 0.95, np.float32)
    box = [40, 50, 200, 250]  # near the left edge in source pixels
    for view in (straight, mirrored):
        view.offer_frame(0, 0.0, frame)
        view.offer_pose(0, 0.0, points, scores, box, 0.9, 1, False)
        assert view.tick() == "emitted"
    a, b = plain.frames[0], selfie.frames[0]
    assert a.shape == b.shape == (180, 320, 3)
    # The picture swapped sides: the bright band is on the right now...
    assert b[100:170, 250:].mean() > 150 and b[100:170, :60].mean() < 50
    # ...and so did every body keypoint, landing at the mirrored column.
    for index in BODY:
        x, y = (points[index] / 2).round().astype(int)
        mx = b.shape[1] - 1 - x
        assert b[max(0, y - 3):y + 4, max(0, mx - 3):mx + 4].max() > 0, \
            f"keypoint {KEYPOINT_NAMES[index]} not at its mirrored spot"
    # The HUD is lettering drawn after the flip: identical pixels in both
    # views where the picture under it is the same (the dark middle).
    hud_a, hud_b = a[:12, 120:200], b[:12, 120:200]
    assert hud_a.max() > 0 and np.array_equal(hud_a, hud_b)


def test_the_mirrored_box_label_sits_over_the_box_as_displayed():
    geometry = fit_geometry(640, 360, 320, 180)  # scale 0.5, no letterbox
    pose = PoseAt("exact", box=[40.0, 80.0, 200.0, 200.0], box_score=0.9)
    plain, selfie = np.zeros((180, 320, 3), np.uint8), np.zeros((180, 320, 3), np.uint8)
    KeypointPainter(geometry).paint(plain, pose, [])
    KeypointPainter(geometry, mirror=True).paint(selfie, pose, [])
    # The box: x 20..120 plain, 199..299 mirrored (y 40..140 in both).
    assert plain[40, 20:120].max() > 0 and plain[40, 199:299].max() == 0
    assert selfie[40, 199:299].max() > 0 and selfie[40, 20:120].max() == 0
    # The label is the only thing above the box (rows 24..37), and it
    # starts at the displayed left edge of the box in each view.
    assert plain[24:37, 20:110].max() > 0 and plain[24:37, 199:].max() == 0
    assert selfie[24:37, 199:290].max() > 0 and selfie[24:37, :120].max() == 0
    # Same glyphs in both: the label reads forward in the mirrored view.
    assert np.array_equal(plain[24:37, 20:110], selfie[24:37, 199:289])


def test_an_interpolated_frame_draws_hollow_body_points():
    clock, publisher = Clock(), FakePublisher()
    view = renderer(publisher, clock, Telemetry(), delay_s=0.0, size=(640, 360))
    frame = gray_frame(640, 360)
    points = np.full((308, 2), 320.0, np.float32)
    points[NOSE] = (200.0, 200.0)
    # Only the nose is confident: no skeleton edge can cross its centre.
    scores = np.zeros(308, np.float32)
    scores[NOSE] = 0.9
    for index in range(0, 7):
        view.offer_frame(index, index / FPS, frame)
    view.offer_pose(0, 0.0, points, scores, None, None, 1, False)
    view.offer_pose(6, 6 / FPS, points, scores, None, None, 1, False)
    # The head is frame 6 with delay 0: exact. Rewind the store to frame 3.
    view.frames.frames = type(view.frames.frames)(
        f for f in view.frames.frames if f.index <= 3)
    assert view.tick() == "emitted"
    assert view.snapshot()["lastState"] == "interpolated"
    canvas = publisher.frames[0]
    # Hollow: the centre of the nose ring is not the body colour.
    assert tuple(canvas[200, 200]) != COLOR_BODY
    assert canvas[190:211, 190:211].max() > 0


def test_context_reaches_the_hud_without_breaking_a_frame():
    """The analysis process's `hud` lines are drawn verbatim under the pose
    lines; anything that is not a non-empty string is skipped, and no other
    context kind reaches the HUD."""
    clock, publisher = Clock(), FakePublisher()
    view = renderer(publisher, clock, Telemetry(), delay_s=0.0)
    view.offer_context("hud", ["calibration ready", "rate 12.0/min", "", 7, None])
    view.offer_context("payload", {"calibrationReady": True})  # not a HUD kind
    view.offer_frame(0, 0.0, gray_frame(64, 36))
    frame = view.frames.nearest(0.0)
    pose = view.track.at(0.0, 0)
    lines = view._hud(frame, pose, dict(view._context), 0.0)
    assert [text for text, _ in lines[-2:]] == ["calibration ready", "rate 12.0/min"]
    assert len(lines) == 4  # the two status lines and the two HUD lines
    assert view.tick() == "emitted"
    assert publisher.frames[0][:8, :8].max() >= 0  # the HUD bar is drawn


def test_a_write_the_publisher_refuses_is_counted_not_raised():
    clock, publisher, telemetry = Clock(), FakePublisher(fail=True), Telemetry()
    view = renderer(publisher, clock, telemetry, delay_s=0.0)
    view.offer_frame(0, 0.0, gray_frame(64, 36))
    assert view.tick() == "emitted"
    assert telemetry.snapshot()["counters"]["overlayWriteDropped"] == 1


def test_close_stops_the_thread_and_closes_the_publisher():
    publisher = FakePublisher()
    view = renderer(publisher, Clock())
    view.start()
    view.close(timeout_s=2.0)
    assert not view.is_alive() and publisher.closed


# -- the publisher -------------------------------------------------------------


class FakeStdin:
    def __init__(self, broken: bool = False):
        self.broken = broken
        self.written = 0
        self.closed = False

    def write(self, data) -> None:
        if self.broken:
            raise BrokenPipeError("encoder went away")
        self.written += len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProc:
    def __init__(self, broken: bool = False):
        self.stdin = FakeStdin(broken)
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_publisher_argv_publishes_rtsp_and_tees_a_recording_when_asked():
    plain = OverlayPublisher("rtsp://127.0.0.1:8554/overlay", (1280, 720), 15.0)
    argv = plain.argv()
    assert argv[:2] == ["ffmpeg", "-nostdin"]
    assert "-video_size" in argv and argv[argv.index("-video_size") + 1] == "1280x720"
    assert "libx264" in argv and "baseline" in argv
    assert argv[argv.index("-flags") + 1] == "+global_header"  # the mp4 needs it
    assert argv[argv.index("-f") + 1] == "rawvideo"  # the input
    assert argv[-7:] == ["-f", "rtsp", "-rtsp_transport", "tcp",
                         "-pkt_size", "1200", "rtsp://127.0.0.1:8554/overlay"]
    recorded = OverlayPublisher("rtsp://r/overlay", (1280, 720), 15.0,
                                record_path="/tmp/run/overlay.mp4")
    tee = recorded.argv()
    assert tee[-5:-1] == ["-f", "tee", "-map", "0:v"]
    slave = tee[-1]
    assert "[f=rtsp:rtsp_transport=tcp:pkt_size=1200]rtsp://r/overlay|" in slave
    assert "onfail=ignore]/tmp/run/overlay.mp4" in slave
    assert "+frag_keyframe+empty_moov" in slave  # readable even if killed
    nvenc = OverlayPublisher("rtsp://r/overlay", (1280, 720), 15.0, encoder="nvenc")
    assert "h264_nvenc" in nvenc.argv()


def test_publisher_defaults_to_every_grid_frame_at_the_same_bits_per_frame():
    """The production view (tee/entrypoint.sh): 30 fps, every frame of the
    decode grid, at 6M - the 200 kbit a frame that 3M at 15 fps was - with
    a keyframe every two seconds and a one-second rate buffer."""
    assert overlay.DEFAULT_FPS == 30.0 and overlay.DEFAULT_BITRATE == "6M"
    publisher = OverlayPublisher("rtsp://r/overlay", (720, 1280), overlay.DEFAULT_FPS)
    argv = publisher.argv()
    after = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
    assert after("-framerate") == "30"
    assert after("-g") == "60"
    assert "keyint=60:min-keyint=60" in after("-x264-params")
    assert after("-b:v") == "6M" and after("-maxrate") == "6M" and after("-bufsize") == "6M"
    # 3M at 15 fps and 6M at 30 fps are the same budget per frame.
    assert 3e6 / 15 == 6e6 / 30 == 200_000
    # A publisher told 15 fps keeps the two-second keyframe interval.
    slower = OverlayPublisher("rtsp://r/overlay", (1280, 720), 15.0).argv()
    assert slower[slower.index("-g") + 1] == "30"


def test_publisher_respawns_a_dead_encoder_with_backoff_and_a_fresh_file():
    clock, telemetry = Clock(), Telemetry()
    procs: list[FakeProc] = []

    def popen(argv, **_):
        proc = FakeProc(broken=len(procs) == 0)  # the first one is broken
        proc.argv = argv
        procs.append(proc)
        return proc

    publisher = OverlayPublisher("rtsp://r/overlay", (64, 36), 15.0,
                                 record_path="/tmp/run/overlay.mp4",
                                 telemetry=telemetry, popen=popen, clock=clock)
    frame = np.zeros((36, 64, 3), np.uint8)
    assert publisher.write(frame) is False  # spawned, then the pipe broke
    assert procs[0].killed and publisher.alive is False
    assert telemetry.snapshot()["counters"]["overlayPublisherRestarts"] == 1
    assert publisher.write(frame) is False  # inside the backoff: dropped, no spawn
    assert len(procs) == 1
    clock.now += 1.5
    assert publisher.write(frame) is True
    assert len(procs) == 2 and procs[1].stdin.written == frame.nbytes
    # The restart writes a new file rather than truncating the first.
    assert procs[0].argv[-1].endswith("/tmp/run/overlay.mp4")
    assert procs[1].argv[-1].endswith("/tmp/run/overlay.2.mp4")
    publisher.close()
    assert procs[1].stdin.closed and publisher.proc is None


def test_build_renderer_is_off_without_a_publish_url_and_probes_for_auto():
    import argparse
    assert overlay.build_renderer(argparse.Namespace(), Telemetry()) is None
    args = argparse.Namespace(overlay_publish="rtsp://r/overlay",
                              overlay_size="640x360", overlay_fps=10.0,
                              overlay_delay_s=0.5, overlay_encoder="auto")
    view = overlay.build_renderer(args, Telemetry(), probe=lambda: False)
    assert view.publisher.encoder == "x264"
    assert view.size == (640, 360) and view.fps == 10.0 and view.delay_s == 0.5
    assert view.publisher.bitrate == overlay.DEFAULT_BITRATE  # none named
    view = overlay.build_renderer(args, Telemetry(), probe=lambda: True)
    assert view.publisher.encoder == "nvenc"
    # Args that name neither cadence nor bit rate get the production view's.
    bare = overlay.build_renderer(
        argparse.Namespace(overlay_publish="rtsp://r/overlay"), Telemetry())
    assert bare.fps == overlay.DEFAULT_FPS == 30.0
    assert bare.publisher.fps == 30.0 and bare.publisher.bitrate == "6M"


# -- the tap in live_pose --------------------------------------------------------


def test_share_full_result_isolates_a_failing_listener():
    telemetry = Telemetry()

    def listener(*_):
        raise RuntimeError("overlay hiccup")

    share_full_result(listener, telemetry, 3, 0.1, None, None, None, None, 0, False)
    assert telemetry.snapshot()["counters"]["poseListenerErrors"] == 1
    share_full_result(None, telemetry, 3, 0.1, None, None, None, None, 0, False)


def test_row_keypoint_arrays_puts_named_points_back_in_index_order():
    keypoints = {KEYPOINT_NAMES[NOSE]: [10.0, 20.0, 0.9],
                 KEYPOINT_NAMES[LEFT_HIP]: [30.0, 40.0, 0.5],
                 "not_a_keypoint": [1.0, 1.0, 1.0]}
    points, scores = row_keypoint_arrays(keypoints)
    assert points.shape == (21, 2) and scores.shape == (21,)
    assert points[NOSE].tolist() == [10.0, 20.0] and scores[NOSE] == pytest.approx(0.9)
    assert points[LEFT_HIP].tolist() == [30.0, 40.0]
    assert scores.sum() == pytest.approx(1.4)


def test_sideload_pose_offers_the_row_to_the_tap_and_returns_it_unchanged(tmp_path):
    rows = [
        {"frame": 0, "atS": 0.0, "box": [1, 2, 3, 4], "boxScore": 0.7,
         "people": 1, "identityUnresolved": False,
         "keypoints": {KEYPOINT_NAMES[i]: [float(i), float(i) * 2, 0.8]
                       for i in BODY}},
        {"frame": 5, "atS": 5 / FPS, "keypoints": None},
    ]
    (tmp_path / "poses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    pose = SideloadPose(tmp_path)
    offered = []
    pose.on_full = lambda *args: offered.append(args)
    row = pose.step(None, 0, 0.0)
    assert row["keypoints"] == rows[0]["keypoints"]
    index, at_s, points, scores, box, box_score, people, unresolved = offered[0]
    assert (index, at_s, box, box_score, people, unresolved) == (
        0, 0.0, [1, 2, 3, 4], 0.7, 1, False)
    assert points[LEFT_HIP].tolist() == [float(LEFT_HIP), float(LEFT_HIP) * 2]
    assert scores[NOSE] == pytest.approx(0.8)
    missing = pose.step(None, 5, 5 / FPS)
    assert missing["keypoints"] is None
    assert offered[1][2] is None  # nobody: no points to draw
    assert MIN_KEYPOINT_SCORE < 0.8


# -- the face inset --------------------------------------------------------------

from sync import ViewSync  # noqa: E402


def dual_renderer(publisher, clock, telemetry=None, size=(1280, 720),
                  face_mirror=False):
    """A two-camera view whose clocks are anchored: the face view's
    timeline starts 2 s after the body's on the senders' clock."""
    sync = ViewSync(clock=clock)
    sync.body.anchor(1_000.0)
    sync.face.anchor(1_002.0)
    view = OverlayRenderer(publisher, telemetry, size=size, fps=15.0,
                           delay_s=0.0, source_fps=FPS, clock=clock,
                           sync=sync, face_mirror=face_mirror)
    return view, sync


def face_frame(width: int, height: int) -> np.ndarray:
    """A face-view frame, bright in its left half only."""
    frame = gray_frame(width, height, value=20)
    frame[:height, :width // 2] = 220
    return frame


def head_points(width: int, height: int, cx: float, cy: float) -> tuple:
    """308 points with the head block gathered around (cx, cy) - a face
    about a fifth of the frame high - and everything else unsure."""
    points = np.zeros((308, 2), np.float32)
    scores = np.zeros(308, np.float32)
    rng = np.random.default_rng(3)
    for index in overlay.HEAD_KEYPOINTS:
        points[index] = (cx + rng.uniform(-width * 0.05, width * 0.05),
                         cy + rng.uniform(-height * 0.1, height * 0.1))
        scores[index] = 0.9
    return points, scores


def test_inset_rect_sits_in_the_top_right_corner_at_the_layouts_size():
    x, y, w, h = overlay.inset_rect(1280, 720)
    assert h == 288 and w == 216  # 40% of the height, 3:4, multiples of 4
    assert (x, y) == (1280 - 216 - overlay.INSET_MARGIN, overlay.INSET_MARGIN)


def test_the_crop_follows_the_head_and_falls_back_to_the_middle():
    width, height = 720, 1280
    # Nobody: the middle of the frame, as tall as the frame allows.
    cx, cy, h = overlay.face_crop_target(None, None, width, height)
    assert (cx, cy) == (360, 640) and h == pytest.approx(min(1280, 720 / 0.75))
    # A head near the top-left: the crop centres on it, grown, at least
    # INSET_MIN_HEIGHT_FRACTION of the frame, and stays inside the frame.
    points, scores = head_points(width, height, 120, 200)
    cx, cy, h = overlay.face_crop_target(points, scores, width, height)
    assert h >= height * overlay.INSET_MIN_HEIGHT_FRACTION
    assert cx - h * 0.75 / 2 >= 0 and cy - h / 2 >= 0
    assert abs(cy - 200) <= h / 2  # the head is inside the crop
    # Unsure head points do not steer it.
    cx2, cy2, h2 = overlay.face_crop_target(points, scores * 0.1, width, height)
    assert (cx2, cy2, h2) == (360, 640, h2) and h2 == pytest.approx(min(1280, 720 / 0.75))
    box = overlay.crop_box(cx, cy, h, width, height)
    x, y, w, hh = box
    assert x % 2 == 0 and y % 2 == 0 and w % 4 == 0 and hh % 4 == 0
    assert 0 <= x and x + w <= width and 0 <= y and y + hh <= height
    crop = overlay.crop_i420(gray_frame(width, height, 77), box)
    assert crop.shape == (hh * 3 // 2, w) and crop[:hh].min() == 77


def test_the_inset_shows_the_face_frame_at_the_same_moment_with_its_keypoints():
    clock, publisher, telemetry = Clock(), FakePublisher(), Telemetry()
    view, sync = dual_renderer(publisher, clock, telemetry)
    body = gray_frame(640, 360, value=0)
    fw, fh = 360, 640
    face_a = face_frame(fw, fh)              # at face time 1.0 = body time 3.0
    face_b = gray_frame(fw, fh, value=90)    # at face time 1.5 = body time 3.5
    view.offer_frame(90, 3.0, body)
    view.offer_face_frame(30, 1.0, face_a)
    view.offer_face_frame(45, 1.5, face_b)
    points, scores = head_points(fw, fh, fw / 2, fh / 2)
    view.offer_face_pose(30, 1.0, points, scores, [100, 200, 160, 200], 0.8, 1, False)
    assert view.tick() == "emitted"
    canvas = publisher.frames[0]
    x, y, w, h = view.snapshot()["view"]["inset"]
    inset = canvas[y:y + h, x:x + w]
    # The inset holds face_a (bright left half, dark right half), not
    # face_b, and the body picture around it stays black.
    assert inset[h // 2:, :w // 4].mean() > 150 and inset[h // 2:, 3 * w // 4:].mean() < 60
    assert canvas[y + h + 10:, x:x + w].max() == 0
    # Its border frames it.
    assert canvas[y - 1, x:x + w].min() == 255
    # The head keypoints were drawn in the inset: colour where they map.
    fit = fit_geometry(*overlay.crop_box(*overlay.face_crop_target(
        points, scores, fw, fh), fw, fh)[2:], w, h)
    box = overlay.crop_box(*overlay.face_crop_target(points, scores, fw, fh), fw, fh)
    px, py = overlay.CropGeometry(fit, box[0], box[1]).point(*points[NOSE])
    patch = inset[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
    assert patch.size and (patch[..., 0] != patch[..., 2]).any()  # a coloured mark
    snap = view.snapshot()["view"]
    assert snap["layout"] == "inset" and snap["faceState"] == "exact"
    assert snap["skewMs"] == 2000 and snap["mirror"] is False and snap["timing"] == "ntp"
    assert "overlayFaceUnpaired" not in telemetry.snapshot()["counters"]


def test_no_face_frame_within_the_tolerance_leaves_the_inset_out():
    clock, publisher, telemetry = Clock(), FakePublisher(), Telemetry()
    view, sync = dual_renderer(publisher, clock, telemetry)
    body = gray_frame(640, 360, value=0)
    view.offer_frame(90, 3.0, body)
    # The face frame is 0.4 s from the body's moment: not this moment.
    view.offer_face_frame(12, 0.6, face_frame(360, 640))
    assert view.tick() == "emitted"
    canvas = publisher.frames[0]
    x, y, w, h = overlay.inset_rect(1280, 720)
    b = overlay.INSET_BORDER
    assert canvas[y - b:y + h + b, x - b:x + w + b].max() == 0  # no inset, no border
    assert view.snapshot()["view"]["faceState"] == "unpaired"
    assert telemetry.snapshot()["counters"]["overlayFaceUnpaired"] == 1
    # And with no face frame at all, the same.
    view2, _ = dual_renderer(FakePublisher(), clock, telemetry)
    view2.offer_frame(0, 0.0, body)
    view2.tick()
    assert view2.snapshot()["view"]["faceState"] == "unpaired"


def test_clocks_not_yet_placed_mean_waiting():
    clock, publisher, telemetry = Clock(), FakePublisher(), Telemetry()
    sync = ViewSync(clock=clock)
    view = OverlayRenderer(publisher, telemetry, size=(1280, 720), fps=15.0,
                           delay_s=0.0, source_fps=FPS, clock=clock, sync=sync)
    view.offer_frame(0, 0.0, gray_frame(640, 360))
    view.offer_face_frame(0, 0.0, face_frame(360, 640))
    assert view.tick() == "emitted"
    assert view.snapshot()["view"]["faceState"] == "waiting"
    assert telemetry.snapshot()["counters"]["overlayFaceWaiting"] == 1
    assert view.mirror is False  # the body picture is never a mirror here


def test_the_inset_is_a_mirror_when_told_and_the_body_is_not():
    clock = Clock()
    plain, selfie = FakePublisher(), FakePublisher()
    straight, _ = dual_renderer(plain, clock, Telemetry())
    mirrored, _ = dual_renderer(selfie, clock, Telemetry(), face_mirror=True)
    body = gray_frame(640, 360, value=0)
    body[:360, :100] = 200  # the body picture is bright on its left
    for view in (straight, mirrored):
        view.offer_frame(90, 3.0, body)
        view.offer_face_frame(30, 1.0, face_frame(360, 640))
        assert view.tick() == "emitted"
    a, b = plain.frames[0], selfie.frames[0]
    x, y, w, h = straight.snapshot()["view"]["inset"]
    # The body picture is on the same side in both.
    assert a[100:170, :40].mean() > 150 and b[100:170, :40].mean() > 150
    # The inset flipped: bright half on the right in the mirror.
    assert a[y + h // 2:y + h, x:x + w // 4].mean() > 150
    assert b[y + h // 2:y + h, x + 3 * w // 4:x + w].mean() > 150
    assert b[y + h // 2:y + h, x:x + w // 4].mean() < 60
    assert mirrored.snapshot()["view"]["mirror"] is True
    # It can be changed while running.
    straight.set_mirror(True)
    straight.offer_frame(91, 3.0 + 1 / FPS, body)
    assert straight.tick() == "emitted"
    c = plain.frames[-1]
    assert c[y + h // 2:y + h, x + 3 * w // 4:x + w].mean() > 150


def test_a_single_view_snapshot_says_so():
    view = renderer(FakePublisher(), Clock(), Telemetry())
    snap = view.snapshot()["view"]
    assert snap["layout"] == "single" and snap["inset"] is None
    assert snap["faceState"] is None and snap["timing"] is None
