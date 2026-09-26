"""The batch bench's contract over stubs: no GPU, no torch.

What is pinned is the shape of a run - the production graph timed before
and after, one row per batch with its capture, replay, per-crop, parity,
memory and queued-behind figures, the `poseBench` lines and gauges - and
its manners: a capture that fails is a `graph=off` row rather than a failed
boot, a slot that disagrees with the batch of one shows in the parity, the
graphs it captured are gone when it returns and the production graph is the
object it was.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import gpu_graph
import pose_bench
from telemetry import Telemetry

K = 12
FRAME = np.zeros((3, 360, 640), np.uint8)


class _Buffer:
    """A static device tensor as replay uses it: `copy_` and a value."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def copy_(self, other):
        self.value = np.array(_value(other))


def _value(x):
    return x.value if isinstance(x, _Buffer) else np.asarray(x)


class _Graph:
    """A production graph: replays the function it was captured over."""

    graphed = True

    def __init__(self, fn):
        self.fn = fn
        self.static_inputs = ()
        self.replays = 0

    def replay(self, *inputs):
        self.replays += 1
        return self.fn(*inputs)


class StubDetector:
    def __init__(self):
        self.graph = _Graph(self.forward)

    def pixels(self, frame):
        return frame.mean()

    def forward(self, pixels):
        return np.zeros((300, 6), np.float32)


class StubTracker:
    """The tracker's surface as the bench uses it, over numpy."""

    def __init__(self, disturb=None):
        self.device = "cuda"
        self.detector_backend = StubDetector()
        self.pose_graph = _Graph(self.pose_forward)
        self.disturb = disturb
        self.last_batch = 0
        self.batch_forwards = 0
        self.static_batches: list[int] = []

    def pose_static_inputs(self, batch):
        self.static_batches.append(batch)
        return (_Buffer(np.zeros((batch, 3, 8, 8), np.float32)),
                _Buffer(np.tile([0.0, 0.0, 8.0, 8.0], (batch, 1))))

    def pose_inputs(self, frame, boxes):
        boxes = np.asarray(boxes, np.float32)
        return np.zeros((len(boxes), 3, 8, 8), np.float32), boxes

    def pose_forward(self, crop, box):
        self.last_batch = 1
        box = _value(box)
        base = np.arange(K, dtype=np.float32)
        return np.stack([base + box[0], base * 2 + box[1],
                         np.full((K,), 0.7, np.float32)], axis=-1)

    def pose_forward_batch(self, crops, boxes):
        crops, boxes = _value(crops), _value(boxes)
        self.last_batch = crops.shape[0]
        self.batch_forwards += 1
        out = np.stack([self.pose_forward(crops[i:i + 1], boxes[i])
                        for i in range(crops.shape[0])])
        self.last_batch = crops.shape[0]
        return out if self.disturb is None else self.disturb(out)


class FakeTimer:
    """Replay time linear in the batch the tracker last ran, memory that
    grows with each captured graph, releases counted."""

    def __init__(self, tracker, state):
        self.tracker = tracker
        self.state = state
        self.releases = 0

    def time(self, fn):
        fn()
        wall = 10.0 * self.tracker.last_batch + 2.0
        return wall, wall - 1.0

    def memory_mib(self):
        return self.state["memory"]

    def release(self):
        self.releases += 1


def _captured_class(state):
    """A stand-in for `Graphed` that recomputes on replay, as the real one
    does over its static buffers, and takes memory when captured."""
    refs: list = []

    class _Captured:
        graphed = True

        def __init__(self, fn, static_inputs):
            self.fn = fn
            self.static_inputs = tuple(static_inputs)
            self.static_outputs = fn(*self.static_inputs)
            state["memory"] += 64.0
            refs.append(weakref.ref(self))

        def replay(self, *inputs):
            for static, value in zip(self.static_inputs, inputs, strict=True):
                static.copy_(value)
            return self.fn(*self.static_inputs)

    return _Captured, refs


class _Refuses:
    graphed = True

    def __init__(self, fn, static_inputs):
        raise RuntimeError("operation not permitted when stream is capturing")


def _run(monkeypatch, tracker, batches, graphed_class=None, **kwargs):
    monkeypatch.delenv("POSE_CUDA_GRAPHS", raising=False)
    state = {"memory": 100.0}
    if graphed_class is None:
        graphed_class, _ = _captured_class(state)
    timer = FakeTimer(tracker, state)
    telemetry = Telemetry()
    report = pose_bench.run(
        tracker, FRAME, batches, repeats=10, warm=2, telemetry=telemetry,
        timer=timer, graphed_class=graphed_class, **kwargs)
    return report, timer, telemetry


def test_the_knob_names_the_batches():
    assert pose_bench.parse_batches("1,2,4,8") == [1, 2, 4, 8]
    assert pose_bench.parse_batches("8, 4; x, 8, -1, 0") == [8, 4]
    assert pose_bench.parse_batches("1") == [1]
    assert pose_bench.parse_batches(None) == []
    assert pose_bench.parse_batches("") == []
    assert pose_bench.parse_batches("0") == []


def test_slot_boxes_are_distinct_crops_inside_the_frame():
    boxes = pose_bench.slot_boxes(8, 640.0, 360.0)
    assert len(boxes) == 8 and len({tuple(box) for box in boxes}) == 8
    for x, y, w, h in boxes:
        assert (w, h) == (640 / 3, 180.0)
        assert 0 <= x <= 640 - w and 0 <= y <= 360 - h


def test_a_run_reports_the_production_graph_around_each_batch(monkeypatch, capsys):
    tracker = StubTracker()
    report, timer, telemetry = _run(monkeypatch, tracker, [1, 2, 4])

    before, after = report["before"], report["after"]
    assert before["batch"] == 1 and before["replayMs"] == 12.0
    assert before["gpuMs"] == 11.0 and before["replayP95Ms"] == 12.0
    assert after["replayMs"] == 12.0 and after["driftPx"] == 0.0
    rows = report["batches"]
    assert [row["batch"] for row in rows] == [1, 2, 4]
    assert tracker.static_batches == [1, 2, 4]
    for row in rows:
        batch = row["batch"]
        assert row["graph"] == "on"
        assert row["replayMs"] == 10.0 * batch + 2.0
        assert row["gpuMs"] == row["replayMs"] - 1.0
        assert row["perCropMs"] == pytest.approx(row["replayMs"] / batch)
        assert row["cropsPerS"] == pytest.approx(1000.0 * batch / row["replayMs"])
        assert row["parityPx"] == 0.0 and row["parityScore"] == 0.0
        assert row["memMiB"] == 64.0  # one graph's worth, measured while it lived
        assert row["captureMs"] >= 0.0
        # Queued behind the batch: the batch's time plus the queued call's.
        assert row["detectBehindMs"] == 10.0 * batch + 2.0
        assert row["pose1BehindMs"] == 12.0
    assert timer.releases == 3

    out = capsys.readouterr().out.splitlines()
    bench_lines = [line for line in out if line.startswith("poseBench ")]
    assert bench_lines[0].startswith("poseBench before batch=1 replayMs=12.0/12.0 gpuMs=11.0")
    assert bench_lines[-1].startswith("poseBench after batch=1 replayMs=12.0/12.0 gpuMs=11.0 driftPx=0.0000")
    assert bench_lines[2].startswith(
        "poseBench batch=2 graph=on captureMs=")
    assert (" replayMs=22.0/22.0 gpuMs=21.0 perCropMs=11.0 cropsPerS=90.9"
            " parityPx=0.0000 parityScore=0.0000 memMiB=64.0"
            " detectBehindMs=22.0 pose1BehindMs=12.0") in bench_lines[2]
    assert "poseBench2: graph captured in" in "\n".join(out)
    gauges = telemetry.snapshot()["gauges"]
    assert gauges["poseBench2ReplayMs"] == 22.0
    assert gauges["poseBench4PerCropMs"] == 10.5
    assert "poseBench1ReplayMs" in gauges and "poseBench8ReplayMs" not in gauges


def test_a_capture_that_fails_is_a_graph_off_row_not_a_failed_boot(monkeypatch, capsys):
    tracker = StubTracker()
    report, _, telemetry = _run(monkeypatch, tracker, [8], graphed_class=_Refuses)

    (row,) = report["batches"]
    assert row["graph"] == "off" and row["batch"] == 8
    assert row["replayMs"] == 82.0 and row["parityPx"] == 0.0
    assert row["memMiB"] == 0.0
    assert telemetry.snapshot()["counters"]["graphCaptureFailed"] == 1
    assert tracker.batch_forwards > 0  # the eager twin ran the batch
    out = capsys.readouterr().out
    assert "poseBench8: graph capture failed" in out
    assert "poseBench batch=8 graph=off " in out
    assert "poseBench after batch=1" in out


def test_a_slot_that_disagrees_with_the_batch_of_one_shows_in_parity(monkeypatch):
    def nudge(out):
        out = out.copy()
        if out.shape[0] > 1:
            out[1, 3, 0] += 2.0
            out[1, 5, 2] -= 0.05
        return out

    tracker = StubTracker(disturb=nudge)
    report, _, _ = _run(monkeypatch, tracker, [1, 2])

    one, two = report["batches"]
    assert one["parityPx"] == 0.0 and one["parityScore"] == 0.0
    assert two["parityPx"] == pytest.approx(2.0)
    assert two["parityScore"] == pytest.approx(0.05)
    assert report["after"]["driftPx"] == 0.0


def test_the_graphs_are_gone_and_the_production_graph_is_the_object_it_was(monkeypatch):
    monkeypatch.delenv("POSE_CUDA_GRAPHS", raising=False)
    tracker = StubTracker()
    production = tracker.pose_graph
    state = {"memory": 0.0}
    graphed_class, refs = _captured_class(state)
    timer = FakeTimer(tracker, state)

    report = pose_bench.run(tracker, FRAME, [2, 4, 8], repeats=4, warm=1,
                            timer=timer, graphed_class=graphed_class)

    gc.collect()
    assert len(refs) == 3 and all(ref() is None for ref in refs)
    assert tracker.pose_graph is production
    assert isinstance(tracker.detector_backend.graph, _Graph)
    assert timer.releases == 3
    assert report["after"]["driftPx"] == 0.0
    # The production graph replayed: before and after, each slot's parity
    # partner, and the queued-behind measurement.
    assert production.replays > 0


def test_no_batches_is_no_work(monkeypatch, capsys):
    tracker = StubTracker()
    report, timer, _ = _run(monkeypatch, tracker, [])
    assert report == {"before": None, "after": None, "batches": []}
    assert tracker.pose_graph.replays == 0 and timer.releases == 0
    assert capsys.readouterr().out == ""


def test_the_cpu_timer_is_the_default_off_cuda():
    assert isinstance(pose_bench.make_timer("cpu"), pose_bench.WallTimer)
    wall, device = pose_bench.WallTimer().time(lambda: None)
    assert wall == device and wall >= 0.0
    assert gpu_graph.enabled("cpu") is False
