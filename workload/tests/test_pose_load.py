"""The load bench's contract over stubs: no GPU, no torch.

What is pinned: the knob's grammar (body, face and seconds, with the face
following the body and the run's length defaulting); the schedule (each
view's picker slots on the grid, in time order, the body first at a
shared slot); a worker that drops at a full queue and counts it, steps
under the shared lock and survives a failing step; the report's shape,
its `poseLoad` line and gauges; and that a run leaves nothing behind.
"""

from __future__ import annotations

import threading
import time

import pytest

import pose_load
from pose_load import LoadSpec, Worker
from telemetry import Telemetry


# -- the knob ------------------------------------------------------------------


def test_the_spec_grammar():
    assert pose_load.parse_spec(None) is None
    assert pose_load.parse_spec("") is None
    assert pose_load.parse_spec("0") is None
    assert pose_load.parse_spec("nine") is None
    assert pose_load.parse_spec("9,9,120,4") is None
    assert pose_load.parse_spec("-9") is None
    assert pose_load.parse_spec("9,-1") is None
    assert pose_load.parse_spec("9,9,120") == LoadSpec(9.0, 9.0, 120.0)
    # The face follows the body; the run is DEFAULT_SECONDS long.
    assert pose_load.parse_spec("9") == LoadSpec(9.0, 9.0, pose_load.DEFAULT_SECONDS)
    assert pose_load.parse_spec("9,6") == LoadSpec(9.0, 6.0, pose_load.DEFAULT_SECONDS)
    # A face at 0 is the body alone; ";" is a separator too.
    assert pose_load.parse_spec("9;0;30").views() == [("body", 9.0)]
    # The run's length is clamped, cadences to the grid.
    assert pose_load.parse_spec("9,9,0.1").seconds == pose_load.MIN_SECONDS
    assert pose_load.parse_spec("9,9,99999").seconds == pose_load.MAX_SECONDS
    assert pose_load.parse_spec("60,60,5") == LoadSpec(30.0, 30.0, 5.0)
    assert LoadSpec(9.0, 9.0, 1.0).label() == "body@9,face@9"
    assert LoadSpec(7.5, 0.0, 1.0).label() == "body@7.5"


def test_the_queue_depth_is_the_producers():
    assert pose_load.queue_depth({}) == pose_load.DEFAULT_QUEUE_DEPTH
    assert pose_load.queue_depth({"POSE_QUEUE_DEPTH": "4"}) == 4
    assert pose_load.queue_depth({"POSE_QUEUE_DEPTH": "0"}) == pose_load.DEFAULT_QUEUE_DEPTH
    assert pose_load.queue_depth({"POSE_QUEUE_DEPTH": "many"}) == pose_load.DEFAULT_QUEUE_DEPTH


# -- the schedule --------------------------------------------------------------


def test_the_schedule_is_each_views_picker_slots_in_time_order():
    plan = pose_load.schedule(LoadSpec(9.0, 9.0, 1.0))
    body = [(due, index) for due, view, index in plan if view == "body"]
    face = [(due, index) for due, view, index in plan if view == "face"]
    slots = [0, 3, 7, 10, 13, 17, 20, 23, 27]
    assert [index for _, index in body] == slots
    assert [index for _, index in face] == slots
    assert [due for due, _ in body] == pytest.approx([s / 30 for s in slots])
    # Time order, the body first where the two share a slot.
    assert [due for due, _, _ in plan] == sorted(due for due, _, _ in plan)
    assert [view for _, view, _ in plan[:2]] == ["body", "face"]
    # One view, another cadence: the stride it always was.
    alone = pose_load.schedule(LoadSpec(6.0, 0.0, 2.0))
    assert [index for _, view, index in alone] == list(range(0, 60, 5))
    assert all(view == "body" for _, view, _ in alone)


# -- the worker ----------------------------------------------------------------


def test_a_full_queue_drops_the_submission_and_counts_it():
    worker = Worker("body", lambda view, k: None, threading.Lock(), depth=2)
    # Not started: nothing drains, so the third submission finds it full.
    assert worker.submit(0) and worker.submit(3)
    assert worker.submit(7) is False
    stats = worker.snapshot()
    assert (stats.submitted, stats.drops, stats.steps) == (3, 1, 0)
    stepped = []
    worker.step = lambda view, k: stepped.append((view, k))
    worker.start()
    assert worker.finish(timeout_s=5.0)
    stats = worker.snapshot()
    assert stepped == [("body", 0), ("body", 1)]
    assert (stats.submitted, stats.drops, stats.steps, stats.errors) == (3, 1, 2, 0)
    assert len(stats.step_ms) == len(stats.queue_wait_ms) == len(stats.lock_wait_ms) == 2


def test_a_failing_step_is_counted_and_the_worker_goes_on(capsys):
    def step(view, k):
        if k == 0:
            raise RuntimeError("CUDA error")

    worker = Worker("face", step, threading.Lock(), depth=4)
    worker.submit(0)
    worker.submit(3)
    worker.start()
    assert worker.finish(timeout_s=5.0)
    stats = worker.snapshot()
    assert (stats.steps, stats.errors, stats.drops) == (2, 1, 0)
    assert "poseLoad: face step 0 failed: RuntimeError('CUDA error')" in capsys.readouterr().out


def test_steps_run_under_the_shared_lock():
    lock = threading.Lock()
    held = []

    def step(view, k):
        held.append(lock.locked())

    worker = Worker("body", step, lock, depth=4)
    worker.submit(0)
    worker.start()
    assert worker.finish(timeout_s=5.0)
    assert held == [True]


# -- the run -------------------------------------------------------------------


IDLE = "1755 MHz, 70.12 W, 38, 0 %, 0x0000000000000000"
LOADED = "1980 MHz, 652.12 W, 61, 100 %, 0x0000000000000000"
CAPPED = "1740 MHz, 700.05 W, 66, 100 %, 0x0000000000000004"


def test_the_gpu_line_is_parsed_into_numbers():
    sample = pose_load.parse_gpu_status(CAPPED)
    assert sample == {"clockMhz": 1740.0, "powerW": 700.05, "tempC": 66.0,
                      "utilPct": 100.0, "throttle": 4,
                      "text": "[1740 MHz/700.05 W/66/100 %/0x0000000000000004]"}
    assert pose_load.parse_gpu_status("N/A, N/A, 61, 100 %, 0x0")["clockMhz"] is None
    assert pose_load.parse_gpu_status("garbage") is None
    # Under load: the lowest clock, the highest power and temperature, the
    # mean utilization, every throttle reason seen.
    summary = pose_load._gpu_summary([pose_load.parse_gpu_status(LOADED),
                                      pose_load.parse_gpu_status(CAPPED)])
    assert summary == {"samples": 2, "clockMinMhz": 1740.0, "powerMaxW": 700.05,
                       "tempMaxC": 66.0, "utilMeanPct": 100.0, "throttle": 4}
    assert pose_load._gpu_summary([]) == {
        "samples": 0, "clockMinMhz": None, "powerMaxW": None, "tempMaxC": None,
        "utilMeanPct": None, "throttle": None}


def test_a_run_submits_the_schedule_steps_it_all_and_reports(capsys):
    telemetry = Telemetry()
    stepped = []
    guard = threading.Lock()

    def step(view, k):
        with guard:
            stepped.append((view, k))

    # Before, the samples under load, after.
    probes = iter([pose_load.parse_gpu_status(IDLE),
                   pose_load.parse_gpu_status(LOADED), pose_load.parse_gpu_status(CAPPED),
                   pose_load.parse_gpu_status(IDLE)])
    report = pose_load.run(step, LoadSpec(9.0, 9.0, 1.0), depth=4,
                           telemetry=telemetry, gpu_probe=lambda: next(probes),
                           gpu_fractions=(0.3, 0.6))
    assert report["steps"] == 18 and report["errors"] == 0 and report["drained"]
    assert report["views"]["body"]["submitted"] == 9 and report["views"]["face"]["submitted"] == 9
    assert report["views"]["body"]["drops"] == report["views"]["face"]["drops"] == 0
    # Each view's k counts its own steps, in order.
    assert sorted(k for view, k in stepped if view == "body") == list(range(9))
    assert sorted(k for view, k in stepped if view == "face") == list(range(9))
    # The run is paced by the wall clock: about a second.
    assert 0.85 <= report["seconds"] <= 3.0
    assert report["stepsPerS"] == pytest.approx(18 / report["seconds"])
    assert report["gpuBefore"]["clockMhz"] == 1755.0 and report["gpuAfter"]["powerW"] == 70.12
    assert report["gpuUnderLoad"] == {
        "samples": 2, "clockMinMhz": 1740.0, "powerMaxW": 700.05, "tempMaxC": 66.0,
        "utilMeanPct": 100.0, "throttle": 4}
    line = capsys.readouterr().out
    assert line.startswith("poseLoad views=body@9,face@9 seconds=")
    assert " depth=4 steps=18 drops=0/0 stepMs=" in line
    assert "busy=" in line
    assert "gpuUnderLoad=clockMin=1740MHz/powerMax=700W/tempMax=66C/utilMean=100%/throttle=0x4" in line
    assert ("gpu=[1755 MHz/70.12 W/38/0 %/0x0000000000000000] -> "
            "[1755 MHz/70.12 W/38/0 %/0x0000000000000000]") in line
    gauges = telemetry.snapshot()["gauges"]
    assert {"poseLoadBodyDrops", "poseLoadFaceDrops", "poseLoadStepP50Ms",
            "poseLoadStepP95Ms", "poseLoadQueueWaitP95Ms", "poseLoadBusy",
            "poseLoadStepsPerS", "poseLoadSeconds", "poseLoadErrors"} <= set(gauges)
    assert gauges["poseLoadBodyDrops"] == 0 and gauges["poseLoadFaceDrops"] == 0
    assert gauges["poseLoadErrors"] == 0
    # The GPU under load, as gauges: what a production-posture slot shows.
    assert gauges["poseLoadGpuClockMinMhz"] == 1740.0
    assert gauges["poseLoadGpuPowerMaxW"] == 700.05
    assert gauges["poseLoadGpuTempMaxC"] == 66.0
    assert gauges["poseLoadGpuUtilMeanPct"] == 100.0
    assert gauges["poseLoadGpuThrottle"] == 4.0
    # Nothing survives: the workers and the sampler are gone.
    assert not any(t.name.startswith("poseLoad-") for t in threading.enumerate())


def test_busy_is_the_step_time_over_the_run(capsys):
    def step(view, k):
        time.sleep(0.01)

    report = pose_load.run(step, LoadSpec(9.0, 0.0, 1.0), depth=4,
                           telemetry=Telemetry(), gpu_probe=lambda: None)
    assert report["steps"] == 9 and report["views"]["body"]["drops"] == 0
    # Nine steps of ~10 ms over ~1 s.
    assert 0.05 <= report["busy"] <= 0.3
    assert report["stepP50Ms"] >= 9.0
    assert "face" not in report["views"] and report["label"] == "body@9"
    # No nvidia-smi: no GPU figures, said so, and no GPU gauges.
    assert report["gpuUnderLoad"]["samples"] == 0
    line = capsys.readouterr().out
    assert "gpuUnderLoad=" not in line and "gpu=[gpu unavailable] -> [gpu unavailable]" in line


def test_a_step_slower_than_the_cadence_drops_and_says_so(capsys):
    """A step that outlasts the interval fills a depth-1 queue: submissions
    drop, are counted per view, and the run still completes and drains."""
    def step(view, k):
        time.sleep(0.08)

    report = pose_load.run(step, LoadSpec(9.0, 9.0, 1.0), depth=1,
                           gpu_probe=lambda: None)
    drops = report["views"]["body"]["drops"] + report["views"]["face"]["drops"]
    assert drops > 0
    assert report["steps"] + drops == 18
    assert report["drained"]
    assert " drops=" in capsys.readouterr().out
