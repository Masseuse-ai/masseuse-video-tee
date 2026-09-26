"""`GpuPose.step` around a stub tracker, and the boot-time parity check.

The models never run here. What is pinned is the shape of the step: one
frame handed to the tracker, the detector cached across the stride, the
keypoints arriving on the host as one array that the row builder and the
overlay tap read as plain numpy, the stage names the readings report - and
the parity check's verdicts over stub graphs.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpu_graph
import live_pose
import pose_rows
from keypoints import BODY, KEYPOINT_NAMES
from telemetry import Telemetry

WIDTH, HEIGHT = 640, 360
K = 308


class StubTracker:
    """The tracker's surface as `GpuPose.step` uses it, over numpy."""

    def __init__(self):
        self.previous = None
        self.scenery = []
        self.frames: list = []
        self.detects = 0
        self.candidate_calls = 0
        self.posed: list = []

    def frame_tensor(self, rgb, device):
        self.frames.append((rgb.shape, device))
        return ("frame", len(self.frames))

    def detection_candidates(self, frame):
        assert frame == ("frame", len(self.frames))
        self.candidate_calls += 1
        return [([10.0, 20.0, 110.0, 220.0], 0.9), ([300.0, 20.0, 330.0, 60.0], 0.4)]

    def detect(self, frame):
        assert frame == ("frame", len(self.frames))
        self.detects += 1
        return [10.0, 20.0, 100.0, 200.0], 0.9, 2, True

    def pose_keypoints(self, frame, box):
        assert frame == ("frame", len(self.frames))
        self.posed.append(box)
        packed = np.zeros((K, 3), np.float32)
        packed[:, 0] = np.arange(K)
        packed[:, 1] = 2 * np.arange(K)
        packed[:, 2] = 0.8
        return packed


def _pose(monkeypatch, tracker, stride: str = "3"):
    monkeypatch.setenv("POSE_DETECT_STRIDE", stride)
    pose = live_pose.GpuPose(device="cuda", telemetry=Telemetry())
    pose.tracker = tracker
    pose._scenery_done = True
    return pose


def test_step_uploads_once_and_hands_numpy_to_the_row_and_the_tap(monkeypatch):
    tracker = StubTracker()
    pose = _pose(monkeypatch, tracker)
    offered = []
    pose.on_full = lambda *args: offered.append(args)
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    row = pose.step(rgb, 7, 1.5)

    assert tracker.frames == [((HEIGHT, WIDTH, 3), "cuda")]
    assert tracker.posed == [[10.0, 20.0, 100.0, 200.0]]
    assert row["frame"] == 7 and row["box"] == [10.0, 20.0, 100.0, 200.0]
    assert row["people"] == 2 and row["identityUnresolved"] is True
    for index in BODY:
        assert row["keypoints"][KEYPOINT_NAMES[index]] == [
            float(index), float(2 * index), 0.8]
    (index, at_s, points, scores, box, box_score, people, unresolved), = offered
    assert (index, at_s, box, box_score, people, unresolved) == (
        7, 1.5, [10.0, 20.0, 100.0, 200.0], 0.9, 2, True)
    assert isinstance(points, np.ndarray) and points.shape == (K, 2)
    assert isinstance(scores, np.ndarray) and scores.shape == (K,)
    assert points[5].tolist() == [5.0, 10.0] and scores[5] == pytest.approx(0.8)

    stages = pose.telemetry.snapshot()["stagesMs"]
    assert set(stages) == {"frameUpload", "detect", "poseInfer"}
    assert "pil" not in stages


def test_the_detector_runs_once_per_stride_and_only_real_detects_are_observed(monkeypatch):
    tracker = StubTracker()
    pose = _pose(monkeypatch, tracker, stride="3")
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for index in range(6):
        pose.step(rgb, index, index / 6)
    assert tracker.detects == 2
    assert len(tracker.frames) == 6 and len(tracker.posed) == 6
    stages = pose.telemetry.snapshot()["stagesMs"]
    assert stages["detect"]["n"] == 2
    assert stages["poseInfer"]["n"] == 6 and stages["frameUpload"]["n"] == 6


def test_scenery_learning_takes_every_candidate_from_the_device_frame(monkeypatch):
    tracker = StubTracker()
    pose = _pose(monkeypatch, tracker)
    pose._scenery_done = False
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    pose.step(rgb, 0, 0.0)
    assert tracker.candidate_calls == 1
    assert pose._warmup[0][0] == [[10.0, 20.0, 110.0, 220.0], [300.0, 20.0, 330.0, 60.0]]
    assert pose._warmup[0][1].shape == (HEIGHT // live_pose.SCENERY_SCALE,
                                        WIDTH // live_pose.SCENERY_SCALE)


def test_nobody_in_frame_is_a_missing_row_and_an_empty_tap(monkeypatch):
    tracker = StubTracker()
    tracker.detect = lambda frame: (None, 0.0, 0, False)
    pose = _pose(monkeypatch, tracker)
    offered = []
    pose.on_full = lambda *args: offered.append(args)
    row = pose.step(np.zeros((HEIGHT, WIDTH, 3), np.uint8), 3, 0.5)
    assert row == pose_rows.missing_row(3, 0.5)
    assert offered[0][2] is None and tracker.posed == []


# -- two views on one model ------------------------------------------------------

class AnchoringTracker(StubTracker):
    """A tracker whose detect keeps an identity anchor, as the real one
    does: what the second view must not disturb."""

    def detect(self, frame):
        box, score, people, unresolved = super().detect(frame)
        self.previous = [self.detects * 1.0] + box[1:]
        return self.previous, score, people, unresolved


def test_a_second_view_keeps_its_own_anchor_scenery_cache_and_tap(monkeypatch):
    tracker = AnchoringTracker()
    tracker.scenery = [[1.0, 2.0, 3.0, 4.0]]  # the body view's, learnt
    pose = _pose(monkeypatch, tracker, stride="3")
    body_taps, face_taps = [], []
    pose.on_full = lambda *args: body_taps.append(args[0])
    pose.view("face").on_full = lambda *args: face_taps.append(args[0])
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    # The body view's step: its anchor lands on the tracker.
    pose.step(rgb, 0, 0.0)
    body_anchor = tracker.previous
    assert body_anchor == [1.0, 20.0, 100.0, 200.0]

    # The face view has not learnt its scenery and has no anchor: its
    # first step learns from the frame (candidates asked) and detects,
    # and afterwards the tracker holds the body's anchor and scenery again.
    face_row = pose.step(rgb, 0, 0.0, view="face")
    assert face_row["frame"] == 0 and face_row["keypoints"]
    assert tracker.candidate_calls == 1
    assert tracker.previous == body_anchor
    assert tracker.scenery == [[1.0, 2.0, 3.0, 4.0]]
    face = pose.view("face")
    assert face.previous == [2.0, 20.0, 100.0, 200.0]
    assert face.scenery == [] and not face.scenery_done
    assert len(face.warmup) == 1

    # Each view's detect cache is its own: the body's second and third
    # steps reuse its box (no detect), the face's too.
    pose.step(rgb, 5, 5 / 30)
    pose.step(rgb, 10, 10 / 30)
    pose.step(rgb, 10, 10 / 30, view="face")
    pose.step(rgb, 20, 20 / 30, view="face")
    assert tracker.detects == 2
    pose.step(rgb, 15, 15 / 30)
    assert tracker.detects == 3 and tracker.previous == [3.0, 20.0, 100.0, 200.0]
    pose.step(rgb, 30, 30 / 30, view="face")
    assert tracker.detects == 4
    assert pose.view("face").previous == [4.0, 20.0, 100.0, 200.0]
    assert tracker.previous == [3.0, 20.0, 100.0, 200.0]

    # Each tap saw its own view's frames only.
    assert body_taps == [0, 5, 10, 15]
    assert face_taps == [0, 10, 20, 30]

    # A session reset clears every view's state but keeps the taps.
    pose.reset_session_state()
    assert tracker.previous is None and tracker.scenery == []
    assert pose.view("face").previous is None and pose.view("face").warmup == []
    assert pose.view("face").cached_detection is None
    assert pose.on_full is not None and pose.view("face").on_full is not None
    assert pose._scenery_done is False


def test_the_body_views_state_keeps_its_old_names(monkeypatch):
    pose = _pose(monkeypatch, StubTracker())
    assert pose._scenery_done is True and pose.views["body"].scenery_done is True
    pose._cached_detection = ("box", 0.5, 1, False)
    assert pose.views["body"].cached_detection == ("box", 0.5, 1, False)
    pose._detect_countdown = 2
    assert pose.views["body"].detect_countdown == 2
    assert pose._warmup is pose.views["body"].warmup


def test_sideload_replays_the_body_only(tmp_path):
    import json

    rows = [{"frame": 0, "atS": 0.0, "keypoints": {"nose": [1.0, 2.0, 0.9]}},
            {"frame": 5, "atS": 5 / 30, "keypoints": {"nose": [3.0, 4.0, 0.9]}}]
    (tmp_path / "poses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pose = live_pose.SideloadPose(tmp_path)
    face_taps = []
    pose.view("face").on_full = lambda *args: face_taps.append(args)
    assert pose.step(None, 5, 5 / 30)["keypoints"] == {"nose": [3.0, 4.0, 0.9]}
    face_row = pose.step(None, 5, 5 / 30, view="face")
    assert face_row == {"frame": 5, "atS": round(5 / 30, 4), "keypoints": None}
    assert len(face_taps) == 1 and face_taps[0][2] is None


def test_sideload_serves_a_nine_fps_session_from_a_six_fps_capture(tmp_path):
    """A capture posed at 6 fps (rows every fifth frame) replayed by a
    session picking 9 fps slots (0, 3, 7, 10, ...): a slot within half the
    capture's interval of a row gets that row; one exactly between two rows
    is not guessed at."""
    import json

    rows = [{"frame": f, "atS": f / 30, "keypoints": {"nose": [float(f), 0.0, 0.9]}}
            for f in range(0, 30, 5)]
    (tmp_path / "poses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pose = live_pose.SideloadPose(tmp_path)
    assert pose.stride == 5 and pose.span == 30
    served = {slot: pose.step(None, slot, slot / 30)["keypoints"]
              for slot in (0, 3, 7, 10, 13, 17, 20, 23, 27)}
    # 3 -> 5 (two away), 7 -> 5, 13 -> 15, 17 -> 15, 23 -> 25, 27 -> 25;
    # 0, 10, 20 are rows of their own.
    assert {s: (kp or {}).get("nose", [None])[0] for s, kp in served.items()} == {
        0: 0.0, 3: 5.0, 7: 5.0, 10: 10.0, 13: 15.0, 17: 15.0, 20: 20.0,
        23: 25.0, 27: 25.0}
    # A 9 fps capture replayed at 9 fps: every slot is its own row and the
    # interval is the pattern's shortest gap.
    nine = [{"frame": f, "atS": f / 30, "keypoints": {"nose": [float(f), 0.0, 0.9]}}
            for f in (0, 3, 7, 10, 13, 17, 20, 23, 27)]
    (tmp_path / "poses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nine))
    pose = live_pose.SideloadPose(tmp_path)
    assert pose.stride == 3 and pose.span == 30
    # ... and the replay wraps at the span, so the second second's slots
    # get the first's rows.
    assert all(pose.step(None, f, f / 30)["keypoints"]["nose"][0] == float(f % 30)
               for f in (0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37))
    # A slot two away from the nearest row (frame 5, between 3 and 7) has
    # no row within half the interval: replayed as no pose.
    assert pose.step(None, 5, 5 / 30)["keypoints"] is None


# -- the parity check ------------------------------------------------------------


class _Graph:
    """A stand-in for a captured graph: replays a fixed function, optionally
    disturbed, over static inputs it does not need."""

    graphed = True

    def __init__(self, fn, disturb=None):
        self.fn = fn
        self.disturb = disturb
        self.static_inputs = ()
        self.replays = 0

    def replay(self, *inputs):
        self.replays += 1
        out = self.fn(*inputs)
        return out if self.disturb is None else self.disturb(out)


class StubDetector:
    def __init__(self, torch):
        self.torch = torch
        self.calls = 0
        self.graph = None

    def pixels(self, frame):
        return frame.float().mean(dim=0, keepdim=True)

    def forward(self, pixels):
        self.calls += 1
        torch = self.torch
        rows = torch.linspace(0.0, 0.9, 300)
        return torch.stack([rows, rows * 0.5, rows + 0.1, rows * 0.5 + 0.1,
                            (1 - rows) * 0.9, torch.zeros(300)], dim=-1)


class ParityTracker:
    def __init__(self, torch):
        self.torch = torch
        self.detector_backend = StubDetector(torch)
        self.pose_graph = None
        self.crops: list = []
        self.forwards = 0

    def pose_crop(self, frame, box):
        self.crops.append(box)
        return frame.float()[:, :8, :8]

    def pose_forward(self, crop, box):
        self.forwards += 1
        torch = self.torch
        base = torch.arange(K, dtype=torch.float32)
        return torch.stack([base + box[0], base * 2 + box[1],
                            torch.full((K,), 0.7)], dim=-1)


def test_parity_keeps_graphs_that_agree_and_reports_them(capsys):
    torch = pytest.importorskip("torch")
    tracker = ParityTracker(torch)
    tracker.detector_backend.graph = _Graph(tracker.detector_backend.forward)
    tracker.pose_graph = _Graph(tracker.pose_forward)
    pose = live_pose.GpuPose(device="cpu", telemetry=Telemetry())
    pose.tracker = tracker

    report = pose.check_parity()

    assert report == {"detectParityPx": 0.0, "detectParityScore": 0.0,
                      "poseParityPx": 0.0, "poseParityScore": 0.0}
    assert tracker.detector_backend.graph.graphed and tracker.pose_graph.graphed
    gauges = pose.telemetry.snapshot()["gauges"]
    assert gauges["detectGraph"] == 1.0 and gauges["poseGraph"] == 1.0
    assert gauges["poseParityPx"] == 0.0 and gauges["detectParityPx"] == 0.0
    assert "graphParityFailed" not in pose.telemetry.snapshot()["counters"]
    out = capsys.readouterr().out
    assert "detectGraph=on poseGraph=on" in out and "poseParityPx=0.0000" in out
    # The frame has structure and the box frames its block.
    frame = live_pose.GpuPose.parity_frame("cpu")
    assert frame.dtype == torch.uint8 and tuple(frame.shape) == (3, 360, 640)
    assert int(frame[:, 180, 320].min()) == 200 and int(frame[0, 0, 0]) == 0
    assert tracker.crops == [[640 / 3, 90.0, 640 / 3, 180.0]]


def test_a_graph_that_disagrees_is_demoted_to_eager(capsys):
    torch = pytest.importorskip("torch")
    tracker = ParityTracker(torch)
    # The detector's rows come back permuted: still the same set, so it passes.
    tracker.detector_backend.graph = _Graph(
        tracker.detector_backend.forward, disturb=lambda out: out.flip(0))
    # The pose graph is off by two pixels on one keypoint: a fault.
    def nudge(out):
        out = out.clone()
        out[3, 0] += 2.0
        return out
    tracker.pose_graph = _Graph(tracker.pose_forward, disturb=nudge)
    pose = live_pose.GpuPose(device="cpu", telemetry=Telemetry())
    pose.tracker = tracker

    report = pose.check_parity()

    assert report["detectParityPx"] == pytest.approx(0.0, abs=1e-3)
    assert report["poseParityPx"] == pytest.approx(2.0)
    assert tracker.detector_backend.graph.graphed
    assert isinstance(tracker.pose_graph, gpu_graph.Eager)
    snapshot = pose.telemetry.snapshot()
    assert snapshot["counters"]["graphParityFailed"] == 1
    assert snapshot["gauges"]["poseGraph"] == 0.0
    assert snapshot["gauges"]["detectGraph"] == 1.0
    assert "pose: graph disagrees with eager" in capsys.readouterr().out
    # The demoted twin is the eager forward itself.
    before = tracker.forwards
    tracker.pose_graph.replay(torch.zeros(3, 8, 8), torch.zeros(4))
    assert tracker.forwards == before + 1


def test_a_score_disagreement_demotes_the_detector():
    torch = pytest.importorskip("torch")
    tracker = ParityTracker(torch)

    def dampen(out):
        out = out.clone()
        out[:, 4] *= 0.9
        return out
    tracker.detector_backend.graph = _Graph(
        tracker.detector_backend.forward, disturb=dampen)
    tracker.pose_graph = _Graph(tracker.pose_forward)
    pose = live_pose.GpuPose(device="cpu", telemetry=Telemetry())
    pose.tracker = tracker
    report = pose.check_parity()
    assert report["detectParityScore"] == pytest.approx(0.09, abs=1e-3)
    assert isinstance(tracker.detector_backend.graph, gpu_graph.Eager)
    assert tracker.pose_graph.graphed


def test_eager_graphs_are_warmed_not_compared():
    torch = pytest.importorskip("torch")
    tracker = ParityTracker(torch)
    tracker.detector_backend.graph = gpu_graph.Eager(tracker.detector_backend.forward)
    tracker.pose_graph = gpu_graph.Eager(tracker.pose_forward)
    pose = live_pose.GpuPose(device="cpu", telemetry=Telemetry())
    pose.tracker = tracker
    assert pose.check_parity() == {}
    assert tracker.detector_backend.calls == 1 and tracker.forwards == 1
    gauges = pose.telemetry.snapshot()["gauges"]
    assert gauges["detectGraph"] == 0.0 and gauges["poseGraph"] == 0.0


# -- the batch bench knob ---------------------------------------------------------


def _bench_pose(monkeypatch, run):
    import pose_bench

    monkeypatch.setattr(pose_bench, "run", run)
    monkeypatch.setattr(live_pose.GpuPose, "parity_frame",
                        staticmethod(lambda device, size=(360, 640): ("frame", device)))
    pose = live_pose.GpuPose(device="cpu", telemetry=Telemetry())
    pose.tracker = StubTracker()
    return pose


def test_the_bench_runs_only_when_the_knob_names_batches(monkeypatch):
    calls = []

    def run(tracker, frame, batches, **kwargs):
        calls.append((tracker, frame, batches, kwargs["telemetry"]))
        return {"before": None, "after": None, "batches": []}

    pose = _bench_pose(monkeypatch, run)
    assert pose.bench_batches(None) is None
    assert pose.bench_batches("") is None
    assert pose.bench_batches("0") is None
    assert calls == []
    assert pose.bench_batches("1,2,4,8") == {"before": None, "after": None, "batches": []}
    assert calls == [(pose.tracker, ("frame", "cpu"), [1, 2, 4, 8], pose.telemetry)]


def test_a_bench_that_fails_is_printed_and_counted_not_fatal(monkeypatch, capsys):
    def run(tracker, frame, batches, **kwargs):
        raise RuntimeError("CUDA out of memory")

    pose = _bench_pose(monkeypatch, run)
    assert pose.bench_batches("8") is None
    assert "poseBench: failed: RuntimeError('CUDA out of memory')" in capsys.readouterr().out
    assert pose.telemetry.snapshot()["counters"]["poseBenchFailed"] == 1


# -- the load bench knob ----------------------------------------------------------


def _load_pose(monkeypatch, run):
    import pose_load

    monkeypatch.setattr(pose_load, "run", run)
    pose = live_pose.GpuPose(device="cuda", telemetry=Telemetry())
    pose.tracker = StubTracker()
    return pose


def test_the_load_bench_runs_only_when_the_knob_names_a_load(monkeypatch):
    import pose_load

    calls = []

    def run(step, spec, **kwargs):
        calls.append((step, spec, kwargs))
        return {"steps": 0}

    pose = _load_pose(monkeypatch, run)
    assert pose.bench_load(None) is None
    assert pose.bench_load("") is None
    assert pose.bench_load("0") is None
    assert calls == []
    assert pose.bench_load("9,9,120") == {"steps": 0}
    (step, spec, kwargs), = calls
    assert spec == pose_load.LoadSpec(9.0, 9.0, 120.0)
    assert callable(step)
    # The session's lock, so the bench's steps take turns as views do, and
    # the telemetry the gauges land on.
    assert kwargs["lock"] is pose._lock and kwargs["telemetry"] is pose.telemetry


def test_the_load_step_is_the_sessions_device_work(monkeypatch):
    """Upload as the decoder's frame is uploaded, detect on every
    POSE_DETECT_STRIDE-th step of a view, pose a fixed box on every step;
    and what the bench's detections left on the tracker is cleared."""
    monkeypatch.setenv("POSE_DETECT_STRIDE", "3")
    tracker = StubTracker()
    pose = live_pose.GpuPose(device="cuda", telemetry=Telemetry())
    pose.tracker = tracker
    step = pose.load_stepper(size=(720, 1280))
    for k in range(6):
        step("body", k)
    assert tracker.frames == [((720, 1280, 3), "cuda")] * 6
    assert tracker.detects == 2  # k = 0 and 3
    assert len(tracker.posed) == 6 and all(box == tracker.posed[0] for box in tracker.posed)
    x, y, w, h = tracker.posed[0]
    assert 0 <= x and x + w <= 1280 and 0 <= y and y + h <= 720 and w > 0 and h > 0
    frame = live_pose.GpuPose.load_frame((720, 1280))
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8 and frame.flags.c_contiguous
    assert frame.std() > 0

    # After a run - here one that returns at once - the tracker's identity
    # anchor and scenery are the fresh slot's.
    import pose_load
    monkeypatch.setattr(pose_load, "run", lambda step, spec, **kwargs: {"steps": 1})
    tracker.previous = "anchor"
    tracker.scenery = ["box"]
    assert pose.bench_load("9,9,5") == {"steps": 1}
    assert tracker.previous is None and tracker.scenery == []


def test_a_load_bench_that_fails_is_printed_and_counted_not_fatal(monkeypatch, capsys):
    def run(step, spec, **kwargs):
        raise RuntimeError("CUDA error: device-side assert")

    pose = _load_pose(monkeypatch, run)
    pose.tracker.previous = "anchor"
    assert pose.bench_load("9,9,120") is None
    assert ("poseLoad: failed: RuntimeError('CUDA error: device-side assert')"
            in capsys.readouterr().out)
    assert pose.telemetry.snapshot()["counters"]["poseLoadFailed"] == 1
    assert pose.tracker.previous is None
