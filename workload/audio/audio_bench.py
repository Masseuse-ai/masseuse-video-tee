"""A boot-time measurement of the audio stage beside the pose load bench.

`pixel/pose_load.py` drives the booted pose the way a session does; this
module drives the audio stage the way a session does, at the same time and
for the same seconds, so that a debug slot shows whether the pose cadence
and the stage's half-second hop both hold with the second tagger running,
and what the tagger costs.

It runs only when `AUDIO_LOAD_BENCH` names it, and only beside a
`POSE_LOAD_BENCH` run (`live_pose.GpuPose.bench_load` starts it before the
pose load and finishes it after): a debug-slot knob the launch policy
admits, never set in production. The value is `mode[,threads]`:

    off    the classifier alone (ced.py), the stage as v0.12 ran it
    cpu    the classifier and the second tagger (beats.py) on the CPU,
           with `threads` torch threads (beats.DEFAULT_THREADS when unnamed)
    cuda   the classifier and the second tagger as CUDA graphs

The tagger reads the windows a session's does (audio_stage.TAGGER_WINDOWS_S),
one engine each.

The stream is synthetic and paced in real time: seeded noise, tones and
silence in 0.5 s chunks, the hop the stage takes from ffmpeg. It is no
one's audio, and nothing is kept: the stage's messages are counted, timed
and dropped. The report is one `audioLoad` line and `audioLoad*` gauges:

    hopMs p50/p95      the stage's whole hop (classifier, pitch, contour)
    misses             hops whose work took longer than the hop itself
    lagS p95           how far the stage's audio clock fell behind the
                       stream's (the decode a session has is not here)
    tagger windows / dropped, ms p50/p95, lagS p50/p95 (every window's
                       message, of every length), graphed
    cpuPerS            the process's CPU seconds per wall second over the
                       run, the pose bench's host work included (compare a
                       run with the tagger to an `off` run)

What it does not measure is the rest of a session's CPU side (the video
decode, the descriptors, the annotated view, the analysis process): a real
session is the test of that.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from audio_stage import HOP_S, SAMPLE_RATE, TAGGER_WINDOW_S, TAGGER_WINDOWS_S, AudioStage

MODES = ("off", "cpu", "cuda")


@dataclass(frozen=True)
class AudioLoadSpec:
    mode: str
    threads: int | None = None


def parse_spec(spec: str | None) -> AudioLoadSpec | None:
    """`AUDIO_LOAD_BENCH`'s value, or None for no bench: unset, empty, or
    not `mode[,threads]` with a known mode. A mistyped debug knob is no
    bench rather than a failed boot."""
    tokens = [t.strip().lower() for t in (spec or "").replace(";", ",").split(",")]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 2 or tokens[0] not in MODES:
        return None
    threads = None
    if len(tokens) == 2:
        try:
            threads = int(tokens[1])
        except ValueError:
            return None
        if threads < 1:
            return None
    return AudioLoadSpec(mode=tokens[0], threads=threads)


class SyntheticSource:
    """`seconds` of seeded noise, tones and silence in hop-sized float32
    chunks, each yielded at its real-time due moment."""

    def __init__(self, seconds: float, hop_s: float = HOP_S, seed: int = 7,
                 clock=time.monotonic, sleep=time.sleep):
        self.hop_samples = int(round(hop_s * SAMPLE_RATE))
        self.hops = max(1, int(seconds / hop_s))
        self.hop_s = hop_s
        self.rng = np.random.default_rng(seed)
        self.clock = clock
        self.sleep = sleep
        self.closed = False
        self.started: float | None = None

    def chunk(self, index: int) -> np.ndarray:
        t = (index * self.hop_samples + np.arange(self.hop_samples)) / SAMPLE_RATE
        phase = index % 12
        if phase < 2:
            return np.zeros(self.hop_samples, dtype=np.float32)
        noise = 0.05 * self.rng.standard_normal(self.hop_samples)
        tone = 0.2 * np.sin(2 * np.pi * (220.0 + 40.0 * phase) * t)
        return (noise + tone * (phase % 3 != 0)).astype(np.float32)

    def chunks(self):
        self.started = self.clock()
        for index in range(self.hops):
            if self.closed:
                return
            due = self.started + (index + 1) * self.hop_s
            wait = due - self.clock()
            if wait > 0:
                self.sleep(wait)
            yield self.chunk(index)

    def stream_s(self) -> float | None:
        return None if self.started is None else self.clock() - self.started

    def stop(self) -> None:
        self.closed = True


class _Recorder:
    """The stage's telemetry interface, kept to the bench: stage durations,
    counters and gauges, so a bench does not mix into the slot's own."""

    def __init__(self):
        self._lock = threading.Lock()
        self.stages: dict[str, list[float]] = {}
        self.counters: dict[str, int] = {}
        self.gauges: dict[str, float] = {}

    def time_stage(self, name: str):
        recorder = self

        class _Timer:
            def __enter__(self):
                self.t0 = time.monotonic()
                return self

            def __exit__(self, *exc):
                with recorder._lock:
                    recorder.stages.setdefault(name, []).append(
                        (time.monotonic() - self.t0) * 1000.0)

        return _Timer()

    def count(self, name: str, by: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + by

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value


def _percentile(values, q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class AudioBench:
    """A running bench: `finish()` stops it and returns the report."""

    def __init__(self, spec: AudioLoadSpec, seconds: float, telemetry=None,
                 classifier=None, tagger=None, source: SyntheticSource | None = None,
                 taggers=None):
        self.spec = spec
        self.telemetry = telemetry
        self.recorder = _Recorder()
        self.source = source if source is not None else SyntheticSource(seconds)
        self.audio_lag: list[float] = []
        self.tagger_lag: list[float] = []
        self.tagger_ms: list[float] = []
        self.messages = {"audio": 0, "beats": 0}
        self._lock = threading.Lock()
        if classifier is None:
            from ced import CedEngine  # noqa: PLC0415 - the library is in the image

            classifier = CedEngine()
            classifier.classify(np.zeros(SAMPLE_RATE, dtype=np.float32))
        if tagger is not None:
            taggers = [(TAGGER_WINDOW_S, tagger)]
        elif taggers is None and spec.mode in ("cpu", "cuda"):
            from beats import DEFAULT_THREADS, BeatsEngine  # noqa: PLC0415

            taggers = [(window_s, BeatsEngine(window_s=window_s, device=spec.mode,
                                              threads=spec.threads or DEFAULT_THREADS,
                                              telemetry=telemetry))
                       for window_s in TAGGER_WINDOWS_S]
        self.taggers = list(taggers or [])
        self.tagger = self.taggers[0][1] if self.taggers else None
        self.stage = AudioStage(self.source, classifier, self._emit, self.recorder,
                                stream_clock=self.source.stream_s, taggers=self.taggers or None)
        self._cpu0 = sum(os.times()[:2])
        self._wall0 = time.monotonic()
        self.stage.start()

    def _emit(self, message: dict) -> None:
        stream_s = self.source.stream_s()
        with self._lock:
            kind = message.get("kind")
            if kind in self.messages:
                self.messages[kind] += 1
            if kind == "audio" and stream_s is not None:
                self.audio_lag.append(stream_s - float(message["atS"]))
            elif kind == "beats":
                self.tagger_ms.append(float(message["computeMs"]))
                if message.get("lagS") is not None:
                    self.tagger_lag.append(float(message["lagS"]))

    def finish(self, timeout_s: float = 30.0) -> dict:
        self.stage.join(timeout=timeout_s)
        if self.stage.is_alive():
            self.stage.stop()
            self.stage.join(timeout=5.0)
        wall = max(time.monotonic() - self._wall0, 1e-6)
        cpu = sum(os.times()[:2]) - self._cpu0
        hops = self.recorder.stages.get("audio", [])
        hop_ms = self.source.hop_s * 1000.0
        report = {
            "mode": self.spec.mode,
            "threads": getattr(self.tagger, "threads", None),
            "graphed": bool(getattr(self.tagger, "graphed", False)),
            "seconds": round(wall, 1),
            "hops": len(hops),
            "hopP50Ms": _percentile(hops, 0.5),
            "hopP95Ms": _percentile(hops, 0.95),
            "misses": sum(1 for ms in hops if ms > hop_ms),
            "lagP95S": _percentile(self.audio_lag, 0.95),
            "errors": self.recorder.counters.get("audioErrors", 0),
            "taggerWindowsS": [window_s for window_s, _ in self.taggers],
            "taggerWindows": self.messages["beats"],
            "taggerDropped": self.recorder.counters.get("beatsDropped", 0),
            "taggerErrors": self.recorder.counters.get("beatsErrors", 0),
            "taggerP50Ms": _percentile(self.tagger_ms, 0.5),
            "taggerP95Ms": _percentile(self.tagger_ms, 0.95),
            "taggerLagP50S": _percentile(self.tagger_lag, 0.5),
            "taggerLagP95S": _percentile(self.tagger_lag, 0.95),
            "cpuPerS": cpu / wall,
        }
        print(_format(report), flush=True)
        if self.telemetry is not None:
            gauges = {
                "audioLoadHops": report["hops"],
                "audioLoadHopP50Ms": report["hopP50Ms"],
                "audioLoadHopP95Ms": report["hopP95Ms"],
                "audioLoadMisses": report["misses"],
                "audioLoadLagP95S": report["lagP95S"],
                "audioLoadErrors": report["errors"],
                "audioLoadTaggerWindows": report["taggerWindows"],
                "audioLoadTaggerDropped": report["taggerDropped"],
                "audioLoadTaggerErrors": report["taggerErrors"],
                "audioLoadTaggerP50Ms": report["taggerP50Ms"],
                "audioLoadTaggerP95Ms": report["taggerP95Ms"],
                "audioLoadTaggerLagP50S": report["taggerLagP50S"],
                "audioLoadTaggerLagP95S": report["taggerLagP95S"],
                "audioLoadTaggerGraphed": 1.0 if report["graphed"] else 0.0,
                "audioLoadCpuPerS": round(report["cpuPerS"], 3),
                "audioLoadMode": float(MODES.index(report["mode"])),
            }
            for name, value in gauges.items():
                if value is not None:
                    self.telemetry.gauge(name, float(value))
        return report


def _ms(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def _format(report: dict) -> str:
    parts = [f"mode={report['mode']}", f"seconds={report['seconds']}",
             f"hops={report['hops']}",
             f"hopMs={_ms(report['hopP50Ms'])}/{_ms(report['hopP95Ms'])}",
             f"misses={report['misses']}", f"lagP95S={_ms(report['lagP95S'])}",
             f"cpuPerS={report['cpuPerS']:.2f}"]
    if report["mode"] != "off":
        windows = "/".join(f"{window_s:g}" for window_s in report["taggerWindowsS"])
        parts += [f"windowsS={windows}",
                  f"tagger={report['taggerWindows']}/{report['taggerDropped']}",
                  f"taggerMs={_ms(report['taggerP50Ms'])}/{_ms(report['taggerP95Ms'])}",
                  f"taggerLagS={_ms(report['taggerLagP50S'])}/{_ms(report['taggerLagP95S'])}",
                  f"threads={report['threads']}", f"graphed={'yes' if report['graphed'] else 'no'}"]
    if report["errors"] or report["taggerErrors"]:
        parts.append(f"errors={report['errors']}/{report['taggerErrors']}")
    return "audioLoad " + " ".join(parts)


def start(spec: str | None, seconds: float, telemetry=None) -> AudioBench | None:
    """The bench `spec` names, running, or None. A bench that cannot start
    (no library, no checkpoint) is printed and counted, never a lost boot."""
    plan = parse_spec(spec)
    if plan is None:
        return None
    try:
        return AudioBench(plan, seconds, telemetry=telemetry)
    except Exception as error:  # noqa: BLE001 - a bench, not the service
        print(f"audioLoad: failed to start: {error!r}", flush=True)
        if telemetry is not None:
            telemetry.count("audioLoadFailed")
        return None
