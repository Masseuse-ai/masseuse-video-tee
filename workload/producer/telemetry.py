"""Producer telemetry: what the pipeline is doing, said with numbers.

Stage timers keep a bounded window of durations and report p50/p95;
counters and gauges are plain numbers; boot phases record what cold start
spent where, since scale-to-zero makes the boot a per-session cost. One
snapshot dictionary serves `/statz`, the periodic log line, and the SSE
stream, so nothing can disagree about what happened.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque


class Telemetry:
    def __init__(self, window: int = 600):
        self._lock = threading.Lock()
        self._stages: dict[str, deque[float]] = {}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._boot: dict[str, float] = {}
        self._window = window
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self._listeners: list = []

    def boot_phase(self, name: str, since_monotonic: float) -> None:
        self.boot_seconds(name, time.monotonic() - since_monotonic)

    def boot_seconds(self, name: str, seconds: float) -> None:
        """A boot phase whose duration was measured elsewhere - a nested
        load that already finished by the time its owner could report it."""
        with self._lock:
            self._boot[name] = round(seconds * 1000.0, 1)

    def time_stage(self, name: str):
        telemetry = self

        class _Timer:
            def __enter__(self):
                self.t0 = time.monotonic()
                return self

            def __exit__(self, *exc):
                telemetry.observe(name, time.monotonic() - self.t0)

        return _Timer()

    def observe(self, stage: str, seconds: float) -> None:
        with self._lock:
            bucket = self._stages.setdefault(
                stage, deque(maxlen=self._window))
            bucket.append(seconds * 1000.0)

    def count(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    @staticmethod
    def _percentile(values: deque[float], q: float) -> float:
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return round(ordered[index], 2)

    def snapshot(self) -> dict:
        with self._lock:
            stages = {
                name: {
                    "p50": self._percentile(values, 0.50),
                    "p95": self._percentile(values, 0.95),
                    "n": len(values),
                }
                for name, values in self._stages.items() if values
            }
            return {
                "atWall": round(time.time(), 3),
                "uptimeS": round(time.monotonic() - self.started_mono, 1),
                "bootMs": dict(self._boot),
                "stagesMs": stages,
                "counters": dict(self._counters),
                "gauges": {k: round(v, 4) for k, v in self._gauges.items()},
            }

    def log_line(self) -> str:
        snap = self.snapshot()
        stages = " ".join(
            f"{name}={value['p50']:.0f}/{value['p95']:.0f}ms"
            for name, value in sorted(snap["stagesMs"].items())
        )
        counters = " ".join(f"{k}={v}"
                            for k, v in sorted(snap["counters"].items()))
        gauges = " ".join(f"{k}={v}"
                          for k, v in sorted(snap["gauges"].items()))
        return f"telemetry {stages} {counters} {gauges}".strip()

    # --- SSE -----------------------------------------------------------
    def subscribe(self):
        queue: deque = deque(maxlen=256)
        event = threading.Event()
        with self._lock:
            self._listeners.append((queue, event))
        return queue, event

    def unsubscribe(self, subscription) -> None:
        with self._lock:
            if subscription in self._listeners:
                self._listeners.remove(subscription)

    def emit(self, kind: str, payload: dict) -> None:
        message = json.dumps({"kind": kind, **payload})
        with self._lock:
            listeners = list(self._listeners)
        for queue, event in listeners:
            queue.append(message)
            event.set()
