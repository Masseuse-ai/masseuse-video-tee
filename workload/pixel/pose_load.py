"""A boot-time measurement of the slot under a session's pose load.

`pose_bench` times one graph replay: it says what a pose costs, not
whether the slot keeps up with two views submitting frames at their
cadences. This module drives the booted models the way a session does -
each view's frames arriving on a wall-clock 30 fps grid, its pose frames
picked by the producer's own `cadence.CadencePicker` (so the 3-4-3
arrivals of 9 fps are real), each view with its own bounded queue and
worker thread as `producer.PoseWorker` gives it, one frame at a time
through the models under the pose's lock - for long enough to see the
steady state: submissions dropped at a full queue, step and queue-wait
times, the fraction of the run the models were busy, and the GPU's
clocks, power and throttle reasons before and after.

It runs only when `POSE_LOAD_BENCH` names the load: a debug-slot knob the
launch policy admits, read by `live_pose.GpuPose.boot` after the batch
bench; nothing in production sets it. The value is
`bodyFps[,faceFps[,seconds]]` - `9,9,120` is both views at 9 fps for two
minutes, `9,0,60` the body alone; the face follows the body's cadence when
unnamed, as it does in a session, and the run is DEFAULT_SECONDS long
when unnamed. The step is the session's device work (`GpuPose.load_step`):
the frame's upload, the detector on every POSE_DETECT_STRIDE-th step, the
pose crop and graph replay, the copy back - over a synthetic frame and a
fixed box, since the frame shows nobody to detect. Both views' grids start
together, so their submissions coincide every few slots and one waits on
the other's step: the worst alignment two cameras can have, which a gate
should see. What it does not measure is the CPU side of a session (the
decode, colour conversion, descriptors and the annotated view), which has
cores of its own.

Everything is printed as one `poseLoad` line and reported as gauges, so a
production-posture slot, whose stdout goes nowhere, shows the result on
/statz:

    poseLoad views=body@9,face@9 seconds=120.0 depth=4 steps=2160
        drops=0/0 stepMs=47.1/49.8 lockWaitMs=0.1/46.9
        queueWaitMs=0.3/47.4 busy=0.85 stepsPerS=18.0
        gpu=[1980 MHz/652 W/61 C/100 %/0x0] -> [...]

`stepMs` is the time under the lock (upload to copy back), `lockWaitMs`
the wait for the other view's step, `queueWaitMs` the time from submission
to the start of the step, each p50/p95; `drops` is body/face; `busy` is
the summed step time over the run's length. A slot holds the load when
drops stay at zero, `busy` leaves a margin under 1.0 and the p95 step is
under the two views' shared interval.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field

from cadence import CadencePicker

GRID_FPS = 30.0
DEFAULT_SECONDS = 60.0
MIN_SECONDS = 1.0
MAX_SECONDS = 600.0
# producer.POSE_QUEUE_DEPTH: what a PoseWorker's queue holds when the
# environment names no depth (the enclave's image sets POSE_QUEUE_DEPTH=4).
DEFAULT_QUEUE_DEPTH = 2
BODY, FACE = "body", "face"
# How long after the last submission the workers are given to finish what
# is queued before the run is measured: the queue's depth in steps, and
# then some.
DRAIN_TIMEOUT_S = 10.0
GPU_QUERY = ("clocks.sm,power.draw,temperature.gpu,utilization.gpu,"
             "clocks_throttle_reasons.active")


@dataclass(frozen=True)
class LoadSpec:
    """What `POSE_LOAD_BENCH` asked for."""

    body_fps: float
    face_fps: float
    seconds: float

    def views(self) -> list[tuple[str, float]]:
        """The views and their cadences, the body first; a face at 0 fps
        is no face view."""
        views = [(BODY, self.body_fps)]
        if self.face_fps > 0:
            views.append((FACE, self.face_fps))
        return views

    def label(self) -> str:
        return ",".join(f"{name}@{fps:g}" for name, fps in self.views())


def parse_spec(spec: str | None) -> LoadSpec | None:
    """`POSE_LOAD_BENCH`'s value as a LoadSpec, or None for no bench: unset,
    empty, "0", or anything that is not `bodyFps[,faceFps[,seconds]]` with
    a positive body cadence. A mistyped debug knob must not stop a boot,
    so a bad value is no bench rather than an error."""
    tokens = [t.strip() for t in (spec or "").replace(";", ",").split(",")]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 3:
        return None
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        return None
    body = values[0]
    if not body > 0 or body != body:  # zero, negative or NaN
        return None
    face = values[1] if len(values) > 1 else body
    if face < 0 or face != face:
        return None
    seconds = values[2] if len(values) > 2 else DEFAULT_SECONDS
    if seconds != seconds:
        return None
    seconds = min(MAX_SECONDS, max(MIN_SECONDS, seconds))
    return LoadSpec(body_fps=min(body, GRID_FPS), face_fps=min(face, GRID_FPS),
                    seconds=seconds)


def queue_depth(environ=os.environ) -> int:
    """A PoseWorker's queue depth as the producer would read it."""
    try:
        depth = int(environ.get("POSE_QUEUE_DEPTH", "0") or "0")
    except ValueError:
        depth = 0
    return depth if depth > 0 else DEFAULT_QUEUE_DEPTH


def schedule(spec: LoadSpec, grid_fps: float = GRID_FPS) -> list[tuple[float, str, int]]:
    """Every pose submission of the run in time order: `(due_s, view,
    slot)`, the slots each view's CadencePicker picks from a grid of
    `seconds` at `grid_fps`. Views that share a slot are due together,
    the body first."""
    out: list[tuple[float, str, int]] = []
    slots = int(spec.seconds * grid_fps)
    for order, (view, fps) in enumerate(spec.views()):
        picker = CadencePicker(grid_fps, fps)
        out.extend((index / grid_fps, view, index, order)
                   for index in range(slots) if picker.take(index))
    out.sort(key=lambda item: (item[0], item[3]))
    return [(due, view, index) for due, view, index, _ in out]


def gpu_status() -> str:
    """The GPU's SM clock, power, temperature, utilization and active
    throttle reasons in one bracketed string, or why not."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "[nvidia-smi unavailable]"
    if out.returncode != 0:
        return "[nvidia-smi failed]"
    fields = [f.strip() for f in out.stdout.strip().split(",")]
    return "[" + "/".join(fields) + "]"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class ViewStats:
    """What one view's worker saw."""

    submitted: int = 0
    steps: int = 0
    drops: int = 0
    errors: int = 0
    step_ms: list[float] = field(default_factory=list)
    lock_wait_ms: list[float] = field(default_factory=list)
    queue_wait_ms: list[float] = field(default_factory=list)


class Worker(threading.Thread):
    """One view's pose worker as the session runs it: a bounded FIFO the
    submitter never blocks on (a full queue drops the submission, counted),
    drained one frame at a time through `step(view, k)` under the shared
    lock. `k` counts this view's steps, so a stepper can run the detector
    on every Nth as the session does."""

    def __init__(self, view: str, step, lock: threading.Lock, depth: int,
                 clock=time.monotonic):
        super().__init__(daemon=True, name=f"poseLoad-{view}")
        self.view = view
        self.step = step
        self.lock = lock
        self.clock = clock
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stats = ViewStats()
        self.stopping = threading.Event()
        self._stats_lock = threading.Lock()

    def submit(self, index: int) -> bool:
        """A pose frame from the decode: queued, or dropped when the queue
        is full. Returns whether it was queued."""
        with self._stats_lock:
            self.stats.submitted += 1
        try:
            self.queue.put_nowait((index, self.clock()))
        except queue.Full:
            with self._stats_lock:
                self.stats.drops += 1
            return False
        return True

    def run(self) -> None:
        k = 0
        while True:
            try:
                index, submitted = self.queue.get(timeout=0.05)
            except queue.Empty:
                if self.stopping.is_set():
                    return
                continue
            dequeued = self.clock()
            with self.lock:
                began = self.clock()
                try:
                    self.step(self.view, k)
                    failed = False
                except Exception as error:  # noqa: BLE001 - counted, the run goes on
                    failed = True
                    print(f"poseLoad: {self.view} step {k} failed: {error!r}",
                          flush=True)
                ended = self.clock()
            k += 1
            with self._stats_lock:
                stats = self.stats
                stats.steps += 1
                if failed:
                    stats.errors += 1
                stats.queue_wait_ms.append((began - submitted) * 1000.0)
                stats.lock_wait_ms.append((began - dequeued) * 1000.0)
                stats.step_ms.append((ended - began) * 1000.0)

    def finish(self, timeout_s: float = DRAIN_TIMEOUT_S) -> bool:
        """Let the queue drain, then stop. Returns whether it drained in
        time."""
        self.stopping.set()
        self.join(timeout=timeout_s)
        return not self.is_alive()

    def snapshot(self) -> ViewStats:
        with self._stats_lock:
            stats = self.stats
            return ViewStats(stats.submitted, stats.steps, stats.drops,
                             stats.errors, list(stats.step_ms),
                             list(stats.lock_wait_ms), list(stats.queue_wait_ms))


def _format(report: dict) -> str:
    """The one `poseLoad` line."""
    views = report["views"]
    drops = "/".join(str(views[name]["drops"]) for name in report["order"])
    parts = [
        f"views={report['label']}",
        f"seconds={report['seconds']:.1f}",
        f"depth={report['depth']}",
        f"steps={report['steps']}",
        f"drops={drops}",
        f"stepMs={report['stepP50Ms']:.1f}/{report['stepP95Ms']:.1f}",
        f"lockWaitMs={report['lockWaitP50Ms']:.1f}/{report['lockWaitP95Ms']:.1f}",
        f"queueWaitMs={report['queueWaitP50Ms']:.1f}/{report['queueWaitP95Ms']:.1f}",
        f"busy={report['busy']:.2f}",
        f"stepsPerS={report['stepsPerS']:.1f}",
    ]
    if report["errors"]:
        parts.append(f"errors={report['errors']}")
    if not report["drained"]:
        parts.append("drained=no")
    parts.append(f"gpu={report['gpuBefore']} -> {report['gpuAfter']}")
    return "poseLoad " + " ".join(parts)


def run(step, spec: LoadSpec, *, lock: threading.Lock | None = None,
        depth: int | None = None, telemetry=None, clock=time.monotonic,
        sleep=time.sleep, gpu_probe=gpu_status,
        grid_fps: float = GRID_FPS) -> dict:
    """The load bench: `spec`'s submissions, on the wall clock, through
    one Worker per view calling `step(view, k)` under `lock`. Returns the
    report, printed as a `poseLoad` line and reported as gauges. Nothing
    of it survives: the workers are joined before it returns."""
    lock = lock if lock is not None else threading.Lock()
    depth = depth if depth is not None else queue_depth()
    before = gpu_probe()
    workers = {view: Worker(view, step, lock, depth, clock=clock)
               for view, _ in spec.views()}
    for worker in workers.values():
        worker.start()
    plan = schedule(spec, grid_fps)
    start = clock()
    for due, view, index in plan:
        target = start + due
        now = clock()
        if target > now:
            sleep(target - now)
        workers[view].submit(index)
    drained = all(worker.finish() for worker in workers.values())
    elapsed = max(clock() - start, 1e-6)
    after = gpu_probe()

    order = [view for view, _ in spec.views()]
    stats = {view: workers[view].snapshot() for view in order}
    step_ms = [ms for s in stats.values() for ms in s.step_ms]
    lock_ms = [ms for s in stats.values() for ms in s.lock_wait_ms]
    wait_ms = [ms for s in stats.values() for ms in s.queue_wait_ms]
    steps = sum(s.steps for s in stats.values())
    report = {
        "label": spec.label(),
        "order": order,
        "seconds": elapsed,
        "depth": depth,
        "steps": steps,
        "errors": sum(s.errors for s in stats.values()),
        "drained": drained,
        "views": {
            view: {"fps": fps, "submitted": s.submitted, "steps": s.steps,
                   "drops": s.drops, "errors": s.errors,
                   "stepP95Ms": _percentile(s.step_ms, 0.95),
                   "queueWaitP95Ms": _percentile(s.queue_wait_ms, 0.95)}
            for (view, fps), s in zip(spec.views(), stats.values())
        },
        "stepP50Ms": _percentile(step_ms, 0.5),
        "stepP95Ms": _percentile(step_ms, 0.95),
        "lockWaitP50Ms": _percentile(lock_ms, 0.5),
        "lockWaitP95Ms": _percentile(lock_ms, 0.95),
        "queueWaitP50Ms": _percentile(wait_ms, 0.5),
        "queueWaitP95Ms": _percentile(wait_ms, 0.95),
        "busy": sum(step_ms) / 1000.0 / elapsed,
        "stepsPerS": steps / elapsed,
        "gpuBefore": before,
        "gpuAfter": after,
    }
    print(_format(report), flush=True)
    if telemetry is not None:
        gauges = {
            "poseLoadBodyDrops": report["views"][BODY]["drops"],
            "poseLoadFaceDrops": report["views"].get(FACE, {}).get("drops", 0),
            "poseLoadStepP50Ms": round(report["stepP50Ms"], 2),
            "poseLoadStepP95Ms": round(report["stepP95Ms"], 2),
            "poseLoadQueueWaitP95Ms": round(report["queueWaitP95Ms"], 2),
            "poseLoadBusy": round(report["busy"], 4),
            "poseLoadStepsPerS": round(report["stepsPerS"], 2),
            "poseLoadSeconds": round(elapsed, 1),
            "poseLoadErrors": report["errors"],
        }
        for name, value in gauges.items():
            telemetry.gauge(name, float(value))
    return report
