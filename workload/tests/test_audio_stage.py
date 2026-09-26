"""The audio stage without ffmpeg or a model: what leaves it, and when.

A scripted classifier stands in for CED and a list of chunks for the
source. What is checked is the shape of every outgoing message against
analysis/protocol.md, that samples are never among them, that the frame
contour is sent exactly once, and how `classify` requests are answered
against the rolling history.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from audio_features import loudness_stats, pitch_stats, spectral_stats
from audio_stage import (HISTORY_S, SAMPLE_RATE, AudioSource, AudioStage,
                         WINDOW_S)
from ced import TARGET_LABELS
from pitch import estimate_pitch, pitch_frames
from telemetry import Telemetry


class ScriptedClassifier:
    """Scores are a function of the loudest sample: loud windows read as a
    moan, quiet ones as breathing, so the test can tell which span was
    classified from the answer."""

    version = "scripted"

    def __init__(self):
        self.calls: list[int] = []

    def classify(self, wav):
        self.calls.append(len(wav))
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        scores = {label: 0.0 for label in TARGET_LABELS}
        if peak > 0.2:
            scores["Wail, moan"] = 0.9
        else:
            scores["Breathing"] = 0.4
        return scores


class ListSource:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.stopped = False

    def chunks(self):
        yield from self._chunks

    def stop(self):
        self.stopped = True


def tone(seconds, hz=180.0, amplitude=0.3):
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def hops(*pieces, hop_s=0.5):
    pcm = np.concatenate(pieces)
    size = int(hop_s * SAMPLE_RATE)
    return [pcm[i:i + size] for i in range(0, len(pcm) - size + 1, size)]


def run_stage(chunks, stream_clock=lambda: None, **options):
    """Feed every chunk the way `run` would, without the end-of-session
    teardown, so the history can be inspected afterwards."""
    telemetry = Telemetry()
    classifier = ScriptedClassifier()
    sent: list[dict] = []
    stage = AudioStage(ListSource([]), classifier, sent.append, telemetry,
                       stream_clock=stream_clock, **options)
    for chunk in chunks:
        stage.feed(chunk)
    return stage, sent, classifier, telemetry


def test_every_hop_emits_one_audio_message_of_numbers_only():
    stage, sent, classifier, telemetry = run_stage(
        hops(silence(1.0), tone(1.0), silence(1.0)), stream_clock=lambda: 12.25)
    audio = [m for m in sent if m["kind"] == "audio"]
    assert len(audio) == 6
    assert [m["atS"] for m in audio] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # The window grows to WINDOW_S and stays there.
    assert [m["windowS"] for m in audio] == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]
    first = audio[0]
    assert set(first) == {"kind", "atS", "streamS", "hopS", "windowS",
                          "scores", "pitch", "frames"}
    assert set(first["scores"]) == set(TARGET_LABELS)
    assert set(first["pitch"]) == {"pitchHz", "pitchConfidence",
                                   "voicedFramePct", "loudnessDbfs"}
    assert first["streamS"] == 12.25
    # Serialisable, and nothing sample-sized in it: a hop of audio is 8000
    # numbers, a hop of contour is ~31 frames of three.
    text = json.dumps(sent)
    assert len(text) < 60_000
    for message in audio:
        assert len(message["frames"]) <= 40
        for frame in message["frames"]:
            assert len(frame) == 3
            assert frame[2] is None or 70 <= frame[2] <= 700
    # The tone hops read as a moan, the silent ones as breathing.
    assert audio[2]["scores"]["Wail, moan"] == 0.9
    assert audio[0]["scores"]["Breathing"] == 0.4
    # Pitch on the tone window.
    assert abs(audio[3]["pitch"]["pitchHz"] - 180) < 3
    assert audio[0]["pitch"]["pitchHz"] is None
    assert telemetry.snapshot()["counters"]["audioHops"] == 6
    assert stage.hops == 6


def test_frame_contour_is_sent_once_and_in_order():
    _, sent, _, _ = run_stage(hops(silence(0.5), tone(1.0), silence(0.5)))
    times = [frame[0] for m in sent if m["kind"] == "audio" for frame in m["frames"]]
    assert times == sorted(times)
    assert len(times) == len(set(times))
    # 16 ms hop over 2 s of audio, minus the first frame's half-width.
    assert 110 <= len(times) <= 125
    # Levels and pitch follow the signal: silent frames at the floor and
    # unvoiced, tone frames loud and voiced.
    frames = [frame for m in sent if m["kind"] == "audio" for frame in m["frames"]]
    silent = [f for f in frames if f[0] < 0.4]
    voiced = [f for f in frames if 0.7 < f[0] < 1.3]
    assert all(f[1] <= -80 and f[2] is None for f in silent)
    assert all(f[1] > -20 and f[2] is not None for f in voiced)


def test_classify_request_measures_the_named_span_from_history():
    telemetry = Telemetry()
    classifier = ScriptedClassifier()
    sent: list[dict] = []
    chunks = hops(silence(1.0), tone(0.6), silence(1.4))
    stage = AudioStage(ListSource([]), classifier, sent.append, telemetry)
    # A request queued before the first hop is served right after it, on
    # what has been decoded by then: nothing of [1.0, 1.6] yet.
    stage.request({"kind": "classify", "id": 1, "fromS": 1.0, "toS": 1.6})
    for chunk in chunks:
        stage.feed(chunk)
    segments = [m for m in sent if m["kind"] == "segment"]
    assert segments == [{"kind": "segment", "id": 1, "error": "empty"}]

    answer = stage.segment({"id": 2, "fromS": 1.0, "toS": 1.6})
    assert "error" not in answer
    assert set(answer) == {"kind", "id", "fromS", "toS", "ced", "pitch",
                           "loudness", "spectral"}
    assert answer["ced"]["topLabel"] == "Wail, moan"
    assert answer["ced"]["topScore"] == 0.9
    assert set(answer["ced"]["scores"]) == set(TARGET_LABELS)
    # The 0.6 s sound was centred in a 1 s context for the classifier.
    assert classifier.calls[-1] == SAMPLE_RATE
    assert abs(answer["pitch"]["medianHz"] - 180) < 3
    assert answer["pitch"]["pitchReliable"] is True
    assert answer["loudness"]["peakDbfs"] > -20
    assert answer["spectral"]["centroidHz"] is not None
    assert telemetry.snapshot()["counters"]["audioSegments"] == 1


def test_classify_refusals():
    stage, _, _, telemetry = run_stage(hops(silence(2.0)), history_s=1.0)
    # Older than the history, malformed, too long, in the future.
    assert stage.segment({"id": 1, "fromS": 0.2, "toS": 0.4})["error"] == "expired"
    assert stage.segment({"id": 2, "fromS": 1.5, "toS": 1.5})["error"] == "span"
    assert stage.segment({"id": 3, "fromS": "x", "toS": 1.5})["error"] == "span"
    assert stage.segment({"id": 4, "fromS": 1.0, "toS": 12.0})["error"] == "span"
    assert stage.segment({"id": 5, "fromS": 2.5, "toS": 3.0})["error"] == "empty"
    ok = stage.segment({"id": 6, "fromS": 1.2, "toS": 1.9})
    assert "error" not in ok and ok["fromS"] == 1.2 and ok["toS"] == 1.9
    assert telemetry.snapshot()["counters"]["audioSegmentErrors"] == 5


def test_run_drives_the_source_then_answers_closed_and_drops_history():
    telemetry = Telemetry()
    sent: list[dict] = []
    source = ListSource(hops(silence(1.0)))
    stage = AudioStage(source, ScriptedClassifier(), sent.append, telemetry)
    stage.run()
    assert [m["kind"] for m in sent] == ["audio", "audio"]
    assert stage.history.size == 0
    # A request that arrives after the end is told so.
    stage.request({"kind": "classify", "id": 7, "fromS": 0.0, "toS": 0.5})
    stage._serve(drain=True)
    assert sent[-1] == {"kind": "segment", "id": 7, "error": "closed"}
    stage.stop()
    assert source.stopped and stage.stopping.is_set()


def test_history_is_bounded():
    stage, _, _, _ = run_stage(hops(silence(HISTORY_S + 3.0)))
    assert stage.history.size == int(HISTORY_S * SAMPLE_RATE)
    assert stage.end_s == pytest.approx(HISTORY_S + 3.0)
    assert stage.window_samples == int(WINDOW_S * SAMPLE_RATE)


def test_source_argv_reads_only_the_audio_track():
    source = AudioSource("rtsp://127.0.0.1:8554/cam", Telemetry())
    argv = source._argv()
    assert argv[:1] == ["ffmpeg"]
    assert "-vn" in argv and "-map" in argv and argv[argv.index("-map") + 1] == "0:a:0"
    assert argv[argv.index("-ar") + 1] == "16000"
    assert argv[argv.index("-ac") + 1] == "1"
    assert argv[argv.index("-f") + 1] == "s16le"
    assert "-rtsp_transport" in argv and "-re" not in argv
    file_source = AudioSource("/tmp/clip.mp4", Telemetry(), realtime=True)
    assert "-re" in file_source._argv() and "-rtsp_transport" not in file_source._argv()


# -- the measurements themselves ---------------------------------------------

def test_estimate_pitch_tracks_a_voiced_tone_and_reports_none_for_silence():
    result = estimate_pitch(tone(1.0, hz=180.0, amplitude=0.2), SAMPLE_RATE)
    assert abs(result["pitchHz"] - 180) < 2
    assert result["pitchConfidence"] > 0.8
    assert result["voicedFramePct"] > 90
    quiet = estimate_pitch(silence(1.0))
    assert quiet["pitchHz"] is None
    assert quiet["pitchConfidence"] == 0
    assert quiet["loudnessDbfs"] == -90


def test_pitch_frames_emit_every_position_and_mark_unvoiced():
    wav = np.concatenate([tone(0.4, hz=210.0, amplitude=0.2), silence(0.5)])
    frames = pitch_frames(wav)
    assert len(frames) == (len(wav) - 1024) // 256 + 1
    assert [f.time_s for f in frames] == sorted(f.time_s for f in frames)
    voiced = [f for f in frames if f.pitch_hz is not None]
    assert voiced and all(abs(f.pitch_hz - 210) < 3 for f in voiced)
    assert all(f.time_s < 0.45 for f in voiced)
    assert all(f.pitch_hz is None for f in frames if f.time_s > 0.5)
    assert all(math.isfinite(f.dbfs) for f in frames)


def test_segment_features_on_a_breath_and_a_tone():
    breath = (np.random.default_rng(1).standard_normal(SAMPLE_RATE // 2)
              .astype(np.float32) * 0.02)
    unvoiced = pitch_stats(breath)
    assert unvoiced["medianHz"] is None and unvoiced["pitchReliable"] is False
    voiced = pitch_stats(tone(0.5, hz=150.0))
    assert voiced["pitchReliable"] and abs(voiced["medianHz"] - 150) < 3
    assert voiced["slopeHzPerS"] is not None
    louder = loudness_stats(tone(0.5, amplitude=0.5))
    quieter = loudness_stats(tone(0.5, amplitude=0.05))
    assert louder["peakDbfs"] > quieter["peakDbfs"] + 15
    noise = spectral_stats(breath)
    pure = spectral_stats(tone(0.5, hz=150.0))
    assert noise["centroidHz"] > pure["centroidHz"]
    assert spectral_stats(np.zeros(100, dtype=np.float32)) == {
        "centroidHz": None, "rolloff85Hz": None}
