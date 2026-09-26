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
