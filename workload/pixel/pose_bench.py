"""A boot-time measurement of the pose graph over a batch of crops.

The slot serves one session, and its pose graph is captured over one crop
(`pose_track.Tracker.capture`). Serving several sessions from one GPU would
mean one forward over their crops together, and whether that pays depends
on how the replay time grows with the batch: a 1B-parameter model over a
single 1024x768 crop may leave an H100 with room to spare, so eight crops
could cost far less than eight replays - or the forward could already be
saturating it. This module measures that on the slot itself, at boot, so
the answer comes from the enclave's own GPU, driver and kernels rather than
from a workstation.

It runs only when `POSE_GRAPH_BENCH` names the batch sizes (a debug-slot
knob: the launch policy admits it, `live_pose` reads it after the parity
check, and nothing in production sets it). For each batch it captures a
graph over `Tracker.pose_forward_batch`, times its replay (wall time to
completion and device time between CUDA events), checks every slot of the
batch against the production batch-of-one graph, notes the memory the
graph reserved and how long a detector replay and a batch-of-one replay
take when queued behind a batch replay on the same stream, then frees the
graph. The production graph is timed before and after so the bench is
seen to leave it as it was. Everything is printed as `poseBench` lines and
the replay figures reported as gauges.
"""

from __future__ import annotations

import time

import numpy as np

DEFAULT_REPEATS = 50
DEFAULT_WARM = 5


def parse_batches(spec: str | None) -> list[int]:
    """`POSE_GRAPH_BENCH`'s value as the batch sizes to measure, in the
    order given, each once: "1,2,4,8". Unset, empty or "0" means no bench.
    Tokens that are not positive integers are dropped rather than fatal: a
    mistyped debug knob must not stop a boot."""
    batches: list[int] = []
    for token in (spec or "").replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit() and int(token) > 0 and int(token) not in batches:
            batches.append(int(token))
    return batches


class WallTimer:
    """Wall-clock timing where there is no device to synchronise: the CPU
    path and the tests. Device time is reported as the wall time."""

    def time(self, fn) -> tuple[float, float]:
        started = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - started) * 1000
        return elapsed, elapsed

    def memory_mib(self) -> float:
        return 0.0

    def release(self) -> None:
        pass


class CudaTimer:
    """Wall time to completion (what a caller waits for) and the device
    time between two CUDA events (what the GPU was busy for), the device
    synchronised before and after each call so nothing else is in flight."""

    def __init__(self):
        import torch

        self.torch = torch

    def time(self, fn) -> tuple[float, float]:
        torch = self.torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return (time.perf_counter() - began) * 1000, start.elapsed_time(end)

    def memory_mib(self) -> float:
        # Reserved, not allocated: a graph's private pool stays reserved
        # for its lifetime whether or not a tensor is live in it.
        return self.torch.cuda.memory_reserved() / 2 ** 20

    def release(self) -> None:
        self.torch.cuda.empty_cache()


def make_timer(device: str):
    return CudaTimer() if str(device).startswith("cuda") else WallTimer()


def slot_boxes(count: int, width: float, height: float) -> list[list[float]]:
    """`count` person boxes (COCO xywh) scattered over a frame of the given
    size, each a third of the width and half the height like the parity
    check's box, so every slot of a batch frames a different crop."""
    box_width, box_height = width / 3, height / 2
    return [
        [round((i * 37 % 60) / 100 * (width - box_width), 2),
         round((i * 53 % 50) / 100 * (height - box_height), 2),
         box_width, box_height]
        for i in range(count)
    ]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _timed(fn, repeats: int, warm: int, timer) -> dict[str, float]:
    """`fn` run `warm` times untimed, then `repeats` times timed: the wall
    p50 and p95 and the device p50, in milliseconds."""
    for _ in range(warm):
        fn()
    samples = [timer.time(fn) for _ in range(max(1, repeats))]
    walls = [wall for wall, _ in samples]
    devices = [device for _, device in samples]
    return {"replayMs": _percentile(walls, 0.5),
            "replayP95Ms": _percentile(walls, 0.95),
            "gpuMs": _percentile(devices, 0.5)}


def _numpy(tensor) -> np.ndarray:
    """A host copy of a `(K, 3)` or `(B, K, 3)` result, whichever library
    produced it; a copy, since a graph's output buffer is rewritten by the
    next replay."""
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy().copy()
    return np.array(tensor, dtype=np.float32)


def _parity(batched: np.ndarray, single: np.ndarray) -> tuple[float, float]:
    """Largest position (image pixels) and score difference between one
    slot of a batch and the batch-of-one result for the same crop."""
    return (float(np.abs(batched[:, :2] - single[:, :2]).max()),
            float(np.abs(batched[:, 2] - single[:, 2]).max()))


def _format(row: dict) -> str:
    """One `poseBench` line: `poseBench batch=8 graph=on captureMs=...`, or
    `poseBench before batch=1 ...` for the production graph's rows."""
    parts = []
    for name, value in row.items():
        if name == "phase":
            parts.append(str(value))
        elif name == "replayMs":
            parts.append(f"replayMs={value:.1f}/{row['replayP95Ms']:.1f}")
        elif name == "replayP95Ms":
            continue
        elif name in ("parityPx", "parityScore", "driftPx"):
            parts.append(f"{name}={value:.4f}")
        elif isinstance(value, float):
            parts.append(f"{name}={value:.1f}")
        else:
            parts.append(f"{name}={value}")
    return "poseBench " + " ".join(parts)


def _bench_batch(tracker, batch: int, crops, boxes, pixels, repeats: int,
                 warm: int, timer, telemetry, graphed_class) -> dict:
    """One batch size: capture, time, compare, queue behind, free."""
    import gpu_graph

    detector = tracker.detector_backend
    memory_before = timer.memory_mib()
    static = tracker.pose_static_inputs(batch)
    started = time.perf_counter()
    graph = gpu_graph.build(
        tracker.pose_forward_batch, static, name=f"poseBench{batch}",
        device=tracker.device, telemetry=telemetry,
        graphed_class=graphed_class)
    capture_ms = (time.perf_counter() - started) * 1000
    crops_b, boxes_b = crops[:batch], boxes[:batch]

    def replay():
        return graph.replay(crops_b, boxes_b)

    timing = _timed(replay, repeats, warm, timer)
    row = {"batch": batch, "graph": "on" if graph.graphed else "off",
           "captureMs": capture_ms, **timing,
           "perCropMs": timing["replayMs"] / batch,
           "cropsPerS": 1000.0 * batch / max(timing["replayMs"], 1e-6)}
    # Every slot of the batch against the production graph over the same
    # crop: the batch must be the same computation, not just a faster one.
    out = _numpy(replay())
    parity_px = parity_score = 0.0
    for i in range(batch):
        single = _numpy(tracker.pose_graph.replay(crops_b[i:i + 1], boxes_b[i]))
        px, score = _parity(out[i], single)
        parity_px, parity_score = max(parity_px, px), max(parity_score, score)
    row["parityPx"], row["parityScore"] = parity_px, parity_score
    # What the graph costs to keep, and what a co-tenant would wait: a
    # detector replay and a batch-of-one replay queued right behind a
    # batch replay on the same stream, launch to completion.
    row["memMiB"] = timer.memory_mib() - memory_before
    behind = max(3, repeats // 5)

    def detect_behind():
        replay()
        detector.graph.replay(pixels)

    def pose_behind():
        replay()
        tracker.pose_graph.replay(crops_b[:1], boxes_b[0])

    row["detectBehindMs"] = _percentile(
        [timer.time(detect_behind)[0] for _ in range(behind)], 0.5)
    row["pose1BehindMs"] = _percentile(
        [timer.time(pose_behind)[0] for _ in range(behind)], 0.5)
    return row


def run(tracker, frame, batches: list[int], *, repeats: int = DEFAULT_REPEATS,
        warm: int = DEFAULT_WARM, telemetry=None, timer=None,
        graphed_class=None) -> dict:
    """The bench over `batches` on a booted, captured `tracker`, with
    `frame` (a CHW uint8 frame on the device, the parity frame) supplying
    the crops. Returns the before and after rows and one row per batch;
    prints each as a `poseBench` line and reports the replay figures as
    gauges. The graphs it captures are released before it returns."""
    import gpu_graph

    if graphed_class is None:
        graphed_class = gpu_graph.Graphed
    if timer is None:
        timer = make_timer(tracker.device)
    batches = [batch for batch in batches if batch > 0]
    report: dict = {"before": None, "after": None, "batches": []}
    if not batches:
        return report
    height, width = (int(size) for size in frame.shape[-2:])
    crops, boxes = tracker.pose_inputs(
        frame, slot_boxes(max(batches), float(width), float(height)))
    pixels = tracker.detector_backend.pixels(frame)

    def production():
        return tracker.pose_graph.replay(crops[:1], boxes[0])

    before = {"phase": "before", "batch": 1,
              **_timed(production, repeats, warm, timer)}
    reference = _numpy(production())
    print(_format(before), flush=True)
    report["before"] = before
    for batch in batches:
        row = _bench_batch(tracker, batch, crops, boxes, pixels, repeats,
                           warm, timer, telemetry, graphed_class)
        timer.release()
        print(_format(row), flush=True)
        if telemetry is not None:
            telemetry.gauge(f"poseBench{batch}ReplayMs", round(row["replayMs"], 2))
            telemetry.gauge(f"poseBench{batch}PerCropMs", round(row["perCropMs"], 2))
        report["batches"].append(row)
    after = {"phase": "after", "batch": 1,
             **_timed(production, repeats, warm, timer)}
    # The production graph's result on the same crop, before and after: the
    # bench's graphs had their own buffers and must have left it alone.
    after["driftPx"] = _parity(_numpy(production()), reference)[0]
    print(_format(after), flush=True)
    report["after"] = after
    return report
