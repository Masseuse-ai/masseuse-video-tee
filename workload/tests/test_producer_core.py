"""The producer core: the online row assembler and the pose worker.

The assembler turns 6 fps pose rows into one row per 30 fps frame, decided
as late as it must be (snap, bridge, or poseless) and never later; the
worker keeps the pose off the decode thread, drops when its queue is full,
and hands every slot to the assembler - and every real pose row to
`on_pose` - in slot order whatever order inference finishes in.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "producer"))
sys.path.insert(0, str(WORKLOAD / "pixel"))

from motion import POSE_SNAP_S, RowAssembler  # noqa: E402
from producer import MOUNT_ROOT, PoseWorker, Session, mounted_path  # noqa: E402
from telemetry import Telemetry  # noqa: E402


def pose_rows(count: int, pose_fps: float = 6.0) -> list[dict]:
    """`count` posed rows at the pose cadence, the hips drifting a pixel a
    row so an interpolated row is told apart from a snapped one."""
    stride = int(round(30.0 / pose_fps))
    return [
        {"frame": i * stride, "atS": round(i / pose_fps, 4),
         "box": [0.0, 0.0, 8.0, 8.0], "boxScore": 0.9, "people": 1,
         "keypoints": {"left_hip": [1.0 + i, 2.0, 0.9],
                       "right_hip": [3.0 + i, 2.0, 0.9]}}
        for i in range(count)
    ]


def test_the_assembler_waits_for_the_far_endpoint_of_a_bridge():
    """A dropped pose row advances the clock; the frames between the last
    tracked row and the drop must not be decided until the row after the
    drop arrives (bridged) or the bridge horizon passes (poseless)."""
    track = pose_rows(12)
    drop = 6
    track[drop]["keypoints"] = None
    assembler = RowAssembler(30.0)
    for row in track[:drop + 1]:
        assembler.push_pose(row)
    early = assembler.flush()
    last_tracked = float(track[drop - 1]["atS"])
    assert early, "frames up to the last tracked row are final"
    assert all(row["atS"] <= last_tracked + POSE_SNAP_S for row in early), (
        "frames past the last tracked row were decided before their bridge's "
        "far endpoint could arrive")
    assembler.push_pose(track[drop + 1])
    late = assembler.flush()
    between = [row for row in late
               if last_tracked < row["atS"] < float(track[drop + 1]["atS"])]
    assert between and all(row["keypoints"] is not None for row in between)


def test_the_assembler_never_deadlocks_on_a_lost_tracker():
    """Keypoint-less pose rows advance the clock; the affected frames emit
    with no pose instead of waiting for one."""
    assembler = RowAssembler(30.0)
    for index in range(6):
        assembler.push_pose({"frame": index, "atS": round(index / 6.0, 4),
                             "keypoints": None})
    rows = assembler.flush()
    assert len(rows) >= 24
    assert all(row["keypoints"] is None for row in rows)


def test_a_full_pose_queue_drops_and_advances_the_clock_in_slot_order():
    """Drops are counted at once but reach the assembler behind the slots
    queued ahead of them, so the clock never runs past a pose still being
    inferred."""
    class LatePose:
        def step(self, rgb, index, at_s):
            return {"frame": index, "atS": round(at_s, 4),
                    "box": [0.0, 0.0, 8.0, 8.0], "boxScore": 0.9,
                    "people": 1,
                    "keypoints": {"left_hip": [1.0, 2.0, 0.9],
                                  "right_hip": [3.0, 2.0, 0.9]}}

    telemetry = Telemetry()
    assembler = RowAssembler(30.0)
    recorded: list[dict] = []
    worker = PoseWorker(LatePose(), assembler, threading.Lock(), telemetry,
                        on_row=recorded.append)
    frame = np.zeros((8, 8, 3), np.uint8)
    for index in range(0, 40, 5):
        worker.submit(frame, index, index / 30.0)
    dropped = telemetry.snapshot()["counters"].get("poseDropped", 0)
    assert dropped >= 6, "queue depth is 2; the rest must drop, counted"
    assert assembler.clock_s is None, (
        "the dropped rows wait behind the two slots still in the queue")
    assert recorded == []

    worker.start()
    import time
    deadline = time.monotonic() + 2.0
    while len(recorded) < 8 and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stopping.set()
    worker.join(timeout=1.0)

    assert [row["frame"] for row in recorded] == list(range(0, 40, 5)), (
        "every slot is a captured row, in slot order")
    assert assembler.clock_s == pytest.approx(35 / 30.0, abs=1e-4)
    assert len(assembler.flush()) > 0
    assert sum(1 for row in recorded if row.get("dropped")) == dropped
    assert all(row["keypoints"] is None for row in recorded
               if row.get("dropped"))
    assert all(row["wallS"] > 0 for row in recorded)


def test_a_drop_behind_a_slow_pose_does_not_move_the_clock_first():
    """The order the assembler sees: a slow slot 0, a drop at slot 5 (known
    first), a posed slot 10 - delivered 0, 5, 10, with the drop held until
    the slow inference reports."""
    import time

    class SlowFirst:
        def step(self, rgb, index, at_s):
            if index == 0:
                time.sleep(0.3)
            return {"frame": index, "atS": round(at_s, 4),
                    "box": [0.0, 0.0, 8.0, 8.0], "boxScore": 0.9,
                    "people": 1,
                    "keypoints": {"left_hip": [1.0, 2.0, 0.9],
                                  "right_hip": [3.0, 2.0, 0.9]}}

    telemetry = Telemetry()
    assembler = RowAssembler(30.0)
    seen: list[float] = []
    original = assembler.push_pose

    def spy(row):
        seen.append(row["atS"])
        original(row)

    assembler.push_pose = spy
    worker = PoseWorker(SlowFirst(), assembler, threading.Lock(), telemetry,
                        queue_depth=1)
    worker.start()
    frame = np.zeros((8, 8, 3), np.uint8)
    worker.submit(frame, 0, 0.0)
    time.sleep(0.05)  # slot 0 is now inside the slow step
    worker.submit(frame, 5, 5 / 30.0)  # fills the depth-1 queue
    worker.submit(frame, 10, 10 / 30.0)  # refused: the drop
    time.sleep(0.05)
    assert seen == [], "the drop must not reach the assembler ahead of slot 0"
    deadline = time.monotonic() + 2.0
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stopping.set()
    worker.join(timeout=1.0)
    assert seen == [0.0, round(5 / 30.0, 4), round(10 / 30.0, 4)]
    assert telemetry.snapshot()["counters"]["poseDropped"] == 1


def test_the_worker_records_posed_and_errored_rows_with_the_wall_clock():
    class FlakyPose:
        def step(self, rgb, index, at_s):
            if index == 5:
                raise RuntimeError("cuda hiccup")
            return {"frame": index, "atS": round(at_s, 4),
                    "box": [0.0, 0.0, 8.0, 8.0], "boxScore": 0.9,
                    "people": 1,
                    "keypoints": {"left_hip": [1.0, 2.0, 0.9],
                                  "right_hip": [3.0, 2.0, 0.9]}}

    telemetry = Telemetry()
    assembler = RowAssembler(30.0)
    recorded: list[dict] = []
    worker = PoseWorker(FlakyPose(), assembler, threading.Lock(), telemetry,
                        on_row=recorded.append)
    worker.start()
    frame = np.zeros((8, 8, 3), np.uint8)
    import time
    for index in (0, 5, 10):
        worker.submit(frame, index, index / 30.0)
        time.sleep(0.05)
    deadline = time.monotonic() + 2.0
    while len(recorded) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stopping.set()
    worker.join(timeout=1.0)

    assert [row["frame"] for row in recorded] == [0, 5, 10]
    assert recorded[0]["keypoints"]["left_hip"] == [1.0, 2.0, 0.9]
    assert "dropped" not in recorded[0] and "error" not in recorded[0]
    assert recorded[1]["keypoints"] is None
    assert "cuda hiccup" in recorded[1]["error"]
    assert all(row["wallS"] > 1.6e9 for row in recorded)
    assert telemetry.snapshot()["counters"]["poseErrors"] == 1


def test_the_worker_hands_every_real_pose_row_to_on_pose_with_the_frame_size():
    """`on_pose(row, flags, frame_size)` sees the worker's own rows - posed,
    dropped or errored, never the assembler's interpolated ones - in slot
    order, with the size of the frames it was fed."""
    class OnePose:
        def step(self, rgb, index, at_s):
            if index == 5:
                raise RuntimeError("cuda hiccup")
            keypoints = None if index == 10 else {
                "left_hip": [40.0, 30.0, 0.9], "right_hip": [40.0, 50.0, 0.9],
            }
            return {"frame": index, "atS": round(at_s, 4), "keypoints": keypoints}

    telemetry = Telemetry()
    seen: list[tuple[dict, dict, tuple[int, int] | None]] = []
    recorded: list[dict] = []
    worker = PoseWorker(OnePose(), RowAssembler(30.0), threading.Lock(), telemetry,
                        on_row=recorded.append,
                        on_pose=lambda row, flags, size: seen.append((row, flags, size)))
    worker.start()
    frame = np.zeros((90, 160, 3), np.uint8)
    import time
    for index in (0, 5, 10):
        worker.submit(frame, index, index / 30.0)
        time.sleep(0.05)
    deadline = time.monotonic() + 2.0
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stopping.set()
    worker.join(timeout=1.0)

    assert [row["frame"] for row, _, _ in seen] == [0, 5, 10]
    assert all(size == (160, 90) for _, _, size in seen)
    assert seen[0][1] == {} and seen[0][0]["keypoints"]["left_hip"] == [40.0, 30.0, 0.9]
    assert "cuda hiccup" in seen[1][1]["error"] and seen[1][0]["keypoints"] is None
    # The last posed row had no body: handed over as it is, keypoint-less.
    assert seen[2][1] == {} and seen[2][0]["keypoints"] is None
    # The capture saw the same three rows, wall clock attached.
    assert [row["frame"] for row in recorded] == [0, 5, 10]


def test_named_runs_get_their_own_capture_dir():
    from producer import capture_dir

    assert capture_dir("/tmp/capture", "") == Path("/tmp/capture")
    assert capture_dir("/tmp/capture", "session-file-1") == Path(
        "/tmp/capture/session-file-1")
    for bad in ("../escape", "a/b", "-dash-first", "x" * 65, "sp ace"):
        with pytest.raises(ValueError):
            capture_dir("/tmp/capture", bad)


def test_a_broken_row_sink_is_counted_never_raised():
    import time

    class Pose:
        def step(self, rgb, index, at_s):
            return {"frame": index, "atS": round(at_s, 4), "keypoints": None}

    telemetry = Telemetry()
    assembler = RowAssembler(30.0)

    def sink(_row):
        raise OSError("disk full")

    worker = PoseWorker(Pose(), assembler, threading.Lock(), telemetry,
                        on_row=sink)
    frame = np.zeros((8, 8, 3), np.uint8)
    for index in range(0, 40, 5):
        worker.submit(frame, index, index / 30.0)
    worker.start()
    deadline = time.monotonic() + 2.0
    while (telemetry.snapshot()["counters"].get("poseRowSinkErrors", 0) < 8
           and time.monotonic() < deadline):
        time.sleep(0.02)
    worker.stopping.set()
    worker.join(timeout=1.0)
    counters = telemetry.snapshot()["counters"]
    assert counters["poseDropped"] >= 6
    assert counters["poseRowSinkErrors"] == 8, (
        "every row - posed or dropped - met the broken sink, none raised")
    assert assembler.clock_s is not None


def test_telemetry_snapshot_carries_stages_and_boot_phases():
    telemetry = Telemetry()
    with telemetry.time_stage("pose"):
        pass
    telemetry.count("framesIn", 3)
    telemetry.gauge("realtimeFactor", 1.25)
    import time
    telemetry.boot_phase("weights", time.monotonic() - 0.05)
    snap = telemetry.snapshot()
    assert snap["stagesMs"]["pose"]["n"] == 1
    assert snap["counters"]["framesIn"] == 3
    assert snap["gauges"]["realtimeFactor"] == 1.25
    assert snap["bootMs"]["weights"] >= 45.0
    assert "pose=" in telemetry.log_line()


def test_classify_requests_reach_the_audio_stage_or_are_refused():
    """The analysis may ask for a span of audio to be measured; the request
    goes to the audio thread and nowhere else, and a session without an
    audio stage answers with an error rather than silence."""
    class Sink:
        def __init__(self):
            self.items = []

        def send(self, message):
            self.items.append(message)

        request = send

    session = Session.__new__(Session)
    session.analysis = Sink()
    session.audio = None
    request = {"kind": "classify", "id": 3, "fromS": 1.0, "toS": 1.5}
    session._on_analysis(request)
    assert session.analysis.items == [
        {"kind": "segment", "id": 3, "error": "no-audio"}]
    session.audio = Sink()
    session._on_analysis(request)
    assert session.audio.items == [request]
    assert len(session.analysis.items) == 1


def test_a_request_may_only_name_files_under_the_mount():
    """/produce takes a `track` path from the query; it must not reach
    anything but the bucket mount, however it is spelled."""
    root = MOUNT_ROOT
    assert mounted_path(f"{root}/runs/ref-x/poses.jsonl") == (
        f"{root}/runs/ref-x/poses.jsonl")
    assert mounted_path(f"{root}/runs/a/../b") == f"{root}/runs/b"
    assert mounted_path(f"{root}/../etc/passwd") is None
    assert mounted_path(root) is None
    assert mounted_path(f"{root}extra/x") is None
    assert mounted_path("/etc/passwd") is None
    assert mounted_path("relative/poses.jsonl") is None
