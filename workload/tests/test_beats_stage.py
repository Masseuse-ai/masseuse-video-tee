"""The second tagger on the audio stage, without torch or the checkpoint.

A stand-in tagger records the windows it is given and returns fixed-size
vectors, so what is checked is the stage's side of the contract
(analysis/protocol.md, `beats`): one window per hop of audio, each the
trailing window ending exactly on a multiple of the hop and zero-padded on
the left at the start; one message per window of numbers only, of a
bounded size; a slow tagger never holding up the hop's `audio` message; a
failing tagger counted and survived; and the boot-time bench around it.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

import audio_bench
from audio_stage import (SAMPLE_RATE, TAGGER_EMBEDDING_DECIMALS,
                         TAGGER_SCORE_DIGITS, AudioStage)
from ced import TARGET_LABELS
from telemetry import Telemetry

CLASSES = 527
EMBEDDING = 768


class Classifier:
    """CED's stand-in: a fixed score per target label."""

    version = "scripted"

    def classify(self, wav):
        return {label: 0.25 for label in TARGET_LABELS}


class Tagger:
    """BEATs' stand-in: the scores and the embedding are a function of the
    window, so the test can tell which window each message came from."""

    def __init__(self, delay_s: float = 0.0, fail_at: set[int] | None = None):
        self.windows: list[np.ndarray] = []
        self.delay_s = delay_s
        self.fail_at = fail_at or set()
        self.lock = threading.Lock()

    def tag(self, window):
        with self.lock:
            index = len(self.windows)
            self.windows.append(np.array(window))
        if self.delay_s:
            time.sleep(self.delay_s)
        if index in self.fail_at:
            raise RuntimeError("stand-in failure")
        level = float(np.mean(np.abs(window)))
        scores = np.full(CLASSES, 1.0 / 3.0, dtype=np.float32)
        scores[0] = level
        scores[2] = 1.23456e-5
        embedding = np.linspace(-1.23456, 1.23456, EMBEDDING, dtype=np.float32)
        return scores, embedding


class ListSource:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def chunks(self):
        yield from self._chunks

    def stop(self):
        pass


def ramp(seconds):
    """Each sample its own index, scaled: a window's first sample says
    exactly where in the stream it starts."""
    n = int(SAMPLE_RATE * seconds)
    return (np.arange(n, dtype=np.float32) + 1.0) / (SAMPLE_RATE * 100.0)


def chunks_of(pcm, hop_s):
    size = int(round(hop_s * SAMPLE_RATE))
    return [pcm[i:i + size] for i in range(0, len(pcm) - size + 1, size)]


def stage_with(tagger, *, synchronous=True, hop_s=1.0, window_s=2.0, stream_clock=lambda: None):
    sent: list[dict] = []
    telemetry = Telemetry()
    stage = AudioStage(ListSource([]), Classifier(), sent.append, telemetry,
                       stream_clock=stream_clock, tagger=tagger, tagger_hop_s=hop_s,
                       tagger_window_s=window_s, tagger_synchronous=synchronous)
    return stage, sent, telemetry


def test_one_window_per_hop_ending_on_the_hop_zero_padded_at_the_start():
    tagger = Tagger()
    stage, sent, _ = stage_with(tagger, stream_clock=lambda: 9.5)
    pcm = ramp(5.0)
    for chunk in chunks_of(pcm, 0.5):
        stage.feed(chunk)
    beats = [m for m in sent if m["kind"] == "beats"]
    assert [m["atS"] for m in beats] == [1.0, 2.0, 3.0, 4.0, 5.0]
    window = 2 * SAMPLE_RATE
    assert all(w.shape == (window,) for w in tagger.windows)
    # The first window: one second of zeros, then the stream's first second.
    first = tagger.windows[0]
    assert np.all(first[:SAMPLE_RATE] == 0.0)
    np.testing.assert_array_equal(first[SAMPLE_RATE:], pcm[:SAMPLE_RATE])
    # Every later window is exactly the two seconds ending on its atS.
    for message, got in zip(beats[1:], tagger.windows[1:]):
        end = int(message["atS"] * SAMPLE_RATE)
        np.testing.assert_array_equal(got, pcm[end - window:end])
    # Each beats message follows the audio message of the hop it ends in.
    kinds = [m["kind"] for m in sent]
    assert kinds[:4] == ["audio", "audio", "beats", "audio"]


def test_windows_land_on_the_hop_whatever_the_chunk_size():
    tagger = Tagger()
    stage, sent, _ = stage_with(tagger)
    pcm = ramp(4.2)
    for chunk in chunks_of(pcm, 0.3):
        stage.feed(chunk)
    beats = [m for m in sent if m["kind"] == "beats"]
    assert [m["atS"] for m in beats] == [1.0, 2.0, 3.0, 4.0]
    for message, got in zip(beats[1:], tagger.windows[1:]):
        end = int(message["atS"] * SAMPLE_RATE)
        np.testing.assert_array_equal(got, pcm[end - 2 * SAMPLE_RATE:end])


def test_the_message_is_numbers_only_and_of_bounded_size():
    stage, sent, telemetry = stage_with(Tagger(), stream_clock=lambda: 7.25)
    for chunk in chunks_of(ramp(3.0), 0.5):
        stage.feed(chunk)
    beats = [m for m in sent if m["kind"] == "beats"]
    message = beats[-1]
    assert set(message) == {"kind", "atS", "streamS", "lagS", "hopS", "windowS",
                            "computeMs", "scores", "embedding"}
    assert message["streamS"] == 7.25 and message["lagS"] == round(7.25 - message["atS"], 3)
    assert message["hopS"] == 1.0 and message["windowS"] == 2.0
    assert len(message["scores"]) == CLASSES and len(message["embedding"]) == EMBEDDING
    assert all(isinstance(v, float) for v in message["scores"] + message["embedding"])
    assert TAGGER_SCORE_DIGITS == 3
    assert message["scores"][1] == 0.333
    # Significant digits, not decimals: a score far under a thousandth keeps
    # its size relative to the others.
    assert message["scores"][2] == 1.23e-5
    assert message["embedding"][0] == round(-1.23456, TAGGER_EMBEDDING_DECIMALS)
    # One row a second in the record: under 12 KB of JSON.
    assert len(json.dumps(message, separators=(",", ":"))) < 12_000
    counters = telemetry.snapshot()["counters"]
    assert counters["beatsWindows"] == 3 and "beatsDropped" not in counters


def test_a_slow_tagger_never_holds_up_the_hop():
    tagger = Tagger(delay_s=0.3)
    stage, sent, telemetry = stage_with(tagger, synchronous=False)
    stage.tagger.start()
    started = time.monotonic()
    for chunk in chunks_of(ramp(6.0), 0.5):
        stage.feed(chunk)
    fed_in = time.monotonic() - started
    audio = [m for m in sent if m["kind"] == "audio"]
    assert len(audio) == 12, "every hop's audio message went out at once"
    assert fed_in < 0.3, f"feeding twelve hops took {fed_in:.2f}s: the stage waited on the tagger"
    stage.tagger.stop()
    assert not stage.tagger.is_alive()
    counters = telemetry.snapshot()["counters"]
    assert counters.get("beatsDropped", 0) >= 1, "windows offered while it was busy replaced the waiting one"
    assert stage.tagger.windows + stage.tagger.dropped <= 6


def test_a_failing_tagger_is_counted_and_the_stage_goes_on():
    stage, sent, telemetry = stage_with(Tagger(fail_at={1}))
    for chunk in chunks_of(ramp(4.0), 0.5):
        stage.feed(chunk)
    beats = [m for m in sent if m["kind"] == "beats"]
    assert [m["atS"] for m in beats] == [1.0, 3.0, 4.0]
    assert len([m for m in sent if m["kind"] == "audio"]) == 8
    assert telemetry.snapshot()["counters"]["beatsErrors"] == 1


def test_without_a_tagger_nothing_changes():
    sent: list[dict] = []
    stage = AudioStage(ListSource([]), Classifier(), sent.append, Telemetry())
    for chunk in chunks_of(ramp(3.0), 0.5):
        stage.feed(chunk)
    assert {m["kind"] for m in sent} == {"audio"}
    assert stage.tagger is None


def test_run_stops_the_tagger_with_the_session():
    tagger = Tagger()
    sent: list[dict] = []
    stage = AudioStage(ListSource(chunks_of(ramp(3.0), 0.5)), Classifier(), sent.append,
                       Telemetry(), tagger=tagger)
    stage.start()
    stage.join(timeout=5.0)
    assert not stage.is_alive()
    assert not stage.tagger.is_alive(), "the tagger thread ends with the stage"
    assert len(stage.history) == 0


def test_bench_spec():
    assert audio_bench.parse_spec(None) is None
    assert audio_bench.parse_spec("") is None
    assert audio_bench.parse_spec("gpu") is None
    assert audio_bench.parse_spec("cpu,zero") is None
    assert audio_bench.parse_spec("cpu,0") is None
    assert audio_bench.parse_spec("off") == audio_bench.AudioLoadSpec("off", None)
    assert audio_bench.parse_spec(" CPU , 6 ") == audio_bench.AudioLoadSpec("cpu", 6)
    assert audio_bench.parse_spec("cuda") == audio_bench.AudioLoadSpec("cuda", None)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_bench_runs_the_stage_on_a_paced_synthetic_stream():
    clock = FakeClock()
    source = audio_bench.SyntheticSource(4.0, clock=clock, sleep=clock.sleep)
    telemetry = Telemetry()
    bench = audio_bench.AudioBench(audio_bench.AudioLoadSpec("cpu"), 4.0, telemetry=telemetry,
                                   classifier=Classifier(), tagger=Tagger(), source=source)
    report = bench.finish()
    assert report["hops"] == 8 and report["misses"] == 0
    # The worker tags as it can; a window still waiting when the stream
    # ends goes with the stage, so tagged plus replaced is at most offered.
    assert report["taggerWindows"] >= 1
    assert report["taggerWindows"] + report["taggerDropped"] <= 4
    assert report["errors"] == 0 and report["taggerErrors"] == 0
    gauges = telemetry.snapshot()["gauges"]
    assert gauges["audioLoadHops"] == 8 and "audioLoadCpuPerS" in gauges
    # The synthetic stream is silence, then noise with and without tones.
    assert np.all(source.chunk(0) == 0.0) and np.std(source.chunk(5)) > 0.01
