"""The second tagger on the audio stage, without torch or the checkpoint.

A stand-in tagger records the windows it is given and returns fixed-size
vectors, so what is checked is the stage's side of the contract
(analysis/protocol.md, `beats`): one window per hop of audio and window
length, each the trailing window ending exactly on a multiple of the hop
and zero-padded on the left at the start; one message per window of
numbers only, of a bounded size; a session's two windows tagged in order,
the first exactly as when it was the only one; a slow tagger never holding
up the hop's `audio` message; a failing tagger counted and survived; the
boot-time bench around it; and the producer's side: both windows by
default, one engine each, named in `hello`.
"""

from __future__ import annotations

import argparse
import collections
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import audio_bench
import beats
import producer
from audio_stage import (SAMPLE_RATE, TAGGER_EMBEDDING_DECIMALS,
                         TAGGER_SCORE_DIGITS, TAGGER_WINDOWS_S, AudioStage)
from ced import TARGET_LABELS
from telemetry import Telemetry
from test_session_views import BODY_URL, FakeLink, StubPose

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


# -- a session's windows: two lengths, one engine each -------------------------

def two_window_stage(*, synchronous=True, stream_clock=lambda: 9.5, **options):
    two, three = Tagger(**options.pop("two", {})), Tagger(**options.pop("three", {}))
    sent: list[dict] = []
    telemetry = Telemetry()
    stage = AudioStage(ListSource([]), Classifier(), sent.append, telemetry,
                       stream_clock=stream_clock, taggers=[(2.0, two), (3.0, three)],
                       tagger_synchronous=synchronous, **options)
    return stage, sent, telemetry, two, three


def test_each_hop_tags_both_windows_the_first_as_when_it_was_alone():
    """At every whole second a 2 s and a 3 s window, each its own engine
    and its own message, the 2 s one first; the 2 s messages are those of
    a stage that reads the 2 s window alone, field for field."""
    stage, sent, telemetry, two, three = two_window_stage()
    alone_tagger = Tagger()
    alone, alone_sent, _ = stage_with(alone_tagger, stream_clock=lambda: 9.5)
    pcm = ramp(5.0)
    for chunk in chunks_of(pcm, 0.5):
        stage.feed(chunk)
        alone.feed(chunk)
    beats_sent = [m for m in sent if m["kind"] == "beats"]
    assert [(m["atS"], m["windowS"]) for m in beats_sent] == [
        (at_s, window_s) for at_s in (1.0, 2.0, 3.0, 4.0, 5.0) for window_s in (2.0, 3.0)]
    assert [m["kind"] for m in sent][:5] == ["audio", "audio", "beats", "beats", "audio"]

    def untimed(message):
        return {key: value for key, value in message.items() if key != "computeMs"}

    assert [untimed(m) for m in beats_sent if m["windowS"] == 2.0] == [
        untimed(m) for m in alone_sent if m["kind"] == "beats"]
    assert [list(m) for m in beats_sent if m["windowS"] == 2.0] == [
        list(m) for m in alone_sent if m["kind"] == "beats"]
    for got, want in zip(two.windows, alone_tagger.windows, strict=True):
        np.testing.assert_array_equal(got, want)
    # The 3 s windows: two seconds of zeros, then one; then exactly the
    # three seconds ending on their atS.
    window = 3 * SAMPLE_RATE
    assert len(three.windows) == 5 and all(w.shape == (window,) for w in three.windows)
    assert np.all(three.windows[0][:2 * SAMPLE_RATE] == 0.0)
    np.testing.assert_array_equal(three.windows[0][2 * SAMPLE_RATE:], pcm[:SAMPLE_RATE])
    assert np.all(three.windows[1][:SAMPLE_RATE] == 0.0)
    np.testing.assert_array_equal(three.windows[1][SAMPLE_RATE:], pcm[:2 * SAMPLE_RATE])
    for message, got in zip([m for m in beats_sent if m["windowS"] == 3.0][2:], three.windows[2:]):
        end = int(message["atS"] * SAMPLE_RATE)
        np.testing.assert_array_equal(got, pcm[end - window:end])
    # The first window keeps its telemetry's names; the second carries its length.
    snapshot = telemetry.snapshot()
    assert snapshot["counters"]["beatsWindows"] == 10
    assert {"beats", "beatsW3"} <= set(snapshot["stagesMs"])
    assert {"beatsMs", "beatsLagS", "beatsW3Ms", "beatsW3LagS"} <= set(snapshot["gauges"])
    assert stage.tagger.windows_s == (2.0, 3.0) == TAGGER_WINDOWS_S


def test_the_history_holds_the_longest_window():
    stage, _, _, _, three = two_window_stage(history_s=1.0)
    assert stage.history_samples == 3 * SAMPLE_RATE
    pcm = ramp(6.0)
    for chunk in chunks_of(pcm, 0.5):
        stage.feed(chunk)
    np.testing.assert_array_equal(three.windows[-1], pcm[-3 * SAMPLE_RATE:])


def test_a_slow_tagger_drops_whole_hops():
    """A hop replaced while it waited loses both its windows, counted as
    two; a hop the tagger took is tagged whole, even when the session
    stops under it."""
    stage, sent, telemetry, _, _ = two_window_stage(
        synchronous=False, two={"delay_s": 0.15}, three={"delay_s": 0.15})
    stage.tagger.start()
    for chunk in chunks_of(ramp(6.0), 0.5):
        stage.feed(chunk)
    assert len([m for m in sent if m["kind"] == "audio"]) == 12
    stage.tagger.stop()
    assert not stage.tagger.is_alive()
    per_hop = collections.Counter(m["atS"] for m in sent if m["kind"] == "beats")
    assert per_hop and set(per_hop.values()) == {2}
    dropped = telemetry.snapshot()["counters"].get("beatsDropped", 0)
    assert dropped >= 2 and dropped % 2 == 0 and stage.tagger.dropped == dropped
    assert stage.tagger.windows + stage.tagger.dropped <= 12


def test_a_failing_window_is_counted_and_the_other_goes_on():
    stage, sent, telemetry, _, _ = two_window_stage(three={"fail_at": {1}})
    for chunk in chunks_of(ramp(4.0), 0.5):
        stage.feed(chunk)
    assert [(m["atS"], m["windowS"]) for m in sent if m["kind"] == "beats"] == [
        (1.0, 2.0), (1.0, 3.0), (2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (4.0, 2.0), (4.0, 3.0)]
    assert telemetry.snapshot()["counters"]["beatsErrors"] == 1


def test_one_form_of_the_taggers_at_a_time():
    with pytest.raises(ValueError):
        AudioStage(ListSource([]), Classifier(), lambda m: None, Telemetry(),
                   tagger=Tagger(), taggers=[(3.0, Tagger())])


def test_bench_tags_a_sessions_windows():
    clock = FakeClock()
    source = audio_bench.SyntheticSource(4.0, clock=clock, sleep=clock.sleep)
    bench = audio_bench.AudioBench(audio_bench.AudioLoadSpec("cpu"), 4.0, telemetry=Telemetry(),
                                   classifier=Classifier(), source=source,
                                   taggers=[(2.0, Tagger()), (3.0, Tagger())])
    report = bench.finish()
    assert report["taggerWindowsS"] == [2.0, 3.0]
    assert report["taggerWindows"] % 2 == 0 and report["taggerDropped"] % 2 == 0
    assert report["taggerWindows"] + report["taggerDropped"] <= 8
    assert "windowsS=2/3" in audio_bench._format(report)


# -- the producer's side: the windows from its arguments, and `hello` ---------

class FakeEngine:
    """beats.BeatsEngine as the producer builds and calls it."""

    made: list["FakeEngine"] = []
    fail_window_s: float | None = None
    version = "unilm-beats:stand-in.pt"
    class_ids = tuple(f"/m/{index:05d}" for index in range(CLASSES))
    embedding_size = EMBEDDING
    graphed = False

    def __init__(self, model_path=None, *, window_s=2.0, device="cpu", threads=None,
                 telemetry=None):
        if window_s == FakeEngine.fail_window_s:
            raise RuntimeError("stand-in load failure")
        self.window_s = window_s
        self.window_samples = int(round(window_s * SAMPLE_RATE))
        self.device = device
        self.threads = threads
        FakeEngine.made.append(self)

    def tag(self, window):
        assert window.shape == (self.window_samples,)
        return (np.full(CLASSES, 0.5, dtype=np.float32),
                np.full(EMBEDDING, self.window_s, dtype=np.float32))


@pytest.fixture
def stand_in_engines(monkeypatch):
    """The producer's engines unloaded, and built from the stand-in."""
    FakeEngine.made = []
    FakeEngine.fail_window_s = None
    monkeypatch.setattr(beats, "BeatsEngine", FakeEngine)
    monkeypatch.setitem(producer._BEATS, "engines", None)
    monkeypatch.setitem(producer._BEATS, "failed", None)
    return FakeEngine


def audio_session(monkeypatch, tmp_path, *argv):
    """A Session built from the producer's own parser with --audio and
    the given arguments; the pose, the classifier and the link stand-ins."""
    links: list[FakeLink] = []

    def open_link(path, on_message, on_error, **fields):
        links.append(FakeLink(fields))
        return links[-1]

    monkeypatch.setattr(producer, "acquire_gpu_pose", lambda args, telemetry: StubPose())
    monkeypatch.setattr(producer, "acquire_audio_classifier", lambda telemetry: Classifier())
    monkeypatch.setattr(producer, "open_link", open_link)
    args = producer.build_parser().parse_args(
        ["--stream", BODY_URL, "--analysis-socket", "", "--sink-dir", str(tmp_path),
         "--audio", *argv])
    telemetry = Telemetry()
    return producer.Session(args, telemetry), links[0], telemetry


def test_a_session_tags_both_windows_by_default_and_hello_names_them(
        monkeypatch, tmp_path, stand_in_engines):
    session, link, _ = audio_session(monkeypatch, tmp_path, "--beats")
    assert [(e.window_samples, e.device, e.threads) for e in FakeEngine.made] == [
        (32_000, "cpu", 4), (48_000, "cpu", 4)]
    fields = link.fields
    assert fields["beats"] is True and fields["beatsModel"] == FakeEngine.version
    assert fields["beatsClassIds"] == list(FakeEngine.class_ids)
    # beatsWindowS stays the first window, as when it was the only one.
    assert fields["beatsWindowS"] == 2.0 and fields["beatsWindowsS"] == [2.0, 3.0]
    assert fields["beatsHopS"] == 1.0 and fields["beatsEmbeddingSize"] == EMBEDDING
    tagger = session.audio.tagger
    assert tagger.windows_s == (2.0, 3.0)
    assert [engine for _, engine in tagger.taggers] == FakeEngine.made
    # The session's own path to the analysis: each hop's audio message,
    # then its windows' messages, the 2 s first.
    tagger.synchronous = True
    for chunk in chunks_of(ramp(3.0), 0.5):
        session.audio.feed(chunk)
    beats_sent = [m for m in link.sent if m["kind"] == "beats"]
    assert [(m["atS"], m["windowS"]) for m in beats_sent] == [
        (1.0, 2.0), (1.0, 3.0), (2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)]
    assert {m["embedding"][0] for m in beats_sent if m["windowS"] == 3.0} == {3.0}
    assert [m["atS"] for m in link.sent if m["kind"] == "audio" and "spectral" in m] == [
        1.0, 2.0, 3.0]
    # The engines are the process's: a second session loads nothing.
    audio_session(monkeypatch, tmp_path, "--beats")
    assert len(FakeEngine.made) == 2


def test_one_window_when_asked_and_none_without_the_tagger(monkeypatch, tmp_path, stand_in_engines):
    _, link, _ = audio_session(monkeypatch, tmp_path, "--beats", "--beats-window-s", "2")
    assert link.fields["beatsWindowS"] == 2.0 and link.fields["beatsWindowsS"] == [2.0]
    session, link, _ = audio_session(monkeypatch, tmp_path)
    assert link.fields["beats"] is False and session.audio.tagger is None
    assert link.fields["beatsWindowS"] is None and link.fields["beatsWindowsS"] is None


def test_a_window_that_fails_to_load_leaves_the_tagger_out(monkeypatch, tmp_path, stand_in_engines):
    FakeEngine.fail_window_s = 3.0
    session, link, telemetry = audio_session(monkeypatch, tmp_path, "--beats")
    assert telemetry.snapshot()["counters"]["beatsLoadFailed"] == 1
    assert link.fields["beats"] is False and link.fields["beatsWindowsS"] is None
    assert session.audio is not None and session.audio.tagger is None


def test_the_windows_argument_and_the_enclaves_command_line():
    parser = producer.build_parser()
    base = ["--analysis-socket", ""]
    assert parser.parse_args(base).beats_windows_s == TAGGER_WINDOWS_S == (2.0, 3.0)
    assert parser.parse_args([*base, "--beats-windows-s", "3, 2,3"]).beats_windows_s == (3.0, 2.0)
    assert parser.parse_args([*base, "--beats-window-s", "2"]).beats_windows_s == (2.0,)
    for bad in ("0", "-1", "x", "", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, "--beats-windows-s", bad])
    with pytest.raises(argparse.ArgumentTypeError):
        producer.parse_beats_windows(",")
    entrypoint = (Path(__file__).resolve().parents[1] / "tee" / "entrypoint.sh").read_text()
    assert "    --audio \\\n    --beats \\\n    --beats-windows-s 2,3 \\\n" in entrypoint
