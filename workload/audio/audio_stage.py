"""The audio stage: the only code in the enclave that reads audio samples.

The stream the producer decodes for its frames may carry an audio track
(the phone's microphone published next to its camera, or a home camera's
own microphone). This module pulls that track through ffmpeg as 16 kHz mono
PCM, keeps the most recent `HISTORY_S` seconds of it in memory, and turns
it into numbers for the analysis process (analysis/protocol.md):

- every `HOP_S`, an `audio` message: the classifier's scores for the
  non-speech vocalization labels (`ced.TARGET_LABELS`) over the trailing
  `WINDOW_S` window, a pitch and level summary of the same window, and the
  per-frame level and pitch contour of the new samples;
- on request (`classify`, from the analysis), a `segment` message: the
  same measurements over one short span the analysis names, taken from the
  history, so that a sound it proposed from the frame contour can be typed
  and measured as a whole.

Samples never leave this process: the history is discarded as it ages out
and when the session ends, and nothing is recorded. What crosses the socket
is listed exhaustively in the protocol document.

`AudioSource` is the ffmpeg side; `AudioStage` is the thread that consumes
any iterable of float32 chunks, which is how the tests drive it without
ffmpeg or a model.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from typing import Callable, Iterable

import numpy as np

from audio_features import (CED_MIN_CONTEXT_S, loudness_stats, pitch_stats,
                            spectral_stats)
from ced import TARGET_LABELS
from pitch import estimate_pitch, pitch_frames

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
# One message per hop; the classifier looks back over the window.
HOP_S = 0.5
WINDOW_S = 2.0
# How much audio a `classify` request can still reach. 20 s of float32 mono
# is 1.3 MB.
HISTORY_S = 20.0
# The frame contour is computed with this much left context so that hop
# boundaries do not blind the 64 ms frame window (pitch.py), and only frames
# newer than the last one sent are sent.
FRAME_CONTEXT_S = 0.128
FRAME_MIN_DBFS = -60.0
# A `classify` span longer than this is refused: a sound is short, and the
# analysis splits longer runs itself.
MAX_SEGMENT_S = 10.0
# Requests waiting for the stage thread; beyond this the newest are refused
# rather than queued behind a stall.
REQUEST_QUEUE_DEPTH = 64
# A stream without an audio track (a phone with its microphone kept out, a
# camera without one) is retried at this cadence for as long as the session
# runs: the track could arrive with a re-publish.
NO_TRACK_RETRY_S = 5.0


class AudioSource:
    """ffmpeg's 16 kHz mono s16le output off the same stream the frames come
    from, in hop-sized chunks, with reconnect. `-vn` drops the video: the
    decode for frames is the producer's own ffmpeg, this one reads only the
    audio track."""

    def __init__(self, url: str, telemetry, hop_s: float = HOP_S,
                 realtime: bool = False,
                 stopping: threading.Event | None = None,
                 sleep=time.sleep):
        self.url = url
        self.telemetry = telemetry
        self.hop_samples = max(1, int(round(hop_s * SAMPLE_RATE)))
        self.network = url.startswith(("rtsp://", "rtsps://"))
        # -re paces a file like the microphone it stands in for; network
        # streams pace themselves.
        self.realtime = realtime and not self.network
        self.stopping = stopping
        self.sleep = sleep
        self.closed = False
        self.proc: subprocess.Popen | None = None

    def _argv(self) -> list[str]:
        pacing = ["-re"] if self.realtime else []
        transport = ["-rtsp_transport", "tcp"] if self.network else []
        # -loglevel info so that ffmpeg describes the input's streams once
        # per connection (codec, rate, channels: what the track is as it
        # arrives); -nostats keeps the progress line out. _relay_stderr
        # passes on the stream lines and the warnings, nothing else.
        return ["ffmpeg", "-nostdin", "-hide_banner", "-nostats",
                "-loglevel", "info",
                *pacing, *transport, "-i", self.url,
                "-vn", "-sn", "-dn", "-map", "0:a:0",
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1"]

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(self._argv(), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        threading.Thread(target=self._relay_stderr, args=(self.proc.stderr,),
                         daemon=True, name="audio-ffmpeg-log").start()

    @staticmethod
    def _relay_stderr(pipe) -> None:
        """ffmpeg's stderr into the producer's log, prefixed: the input's
        stream lines (the audio track as ffmpeg sees it) and anything a
        component reports (`[rtsp @ ...]`, `[opus @ ...]`: losses, decode
        errors), capped so a failing connection cannot flood the log."""
        kept = 0
        try:
            for raw in pipe:
                line = raw.decode("utf-8", "replace").rstrip()
                stripped = line.strip()
                if not (stripped.startswith("Stream #0:")
                        or stripped.startswith("Input #0")
                        or stripped.startswith("[")):
                    continue
                if stripped.startswith("Stream #0:") and (
                        "->" in stripped or "pcm_s16le" in stripped):
                    continue  # the output side: ours already
                kept += 1
                if kept <= 40:
                    print(f"audio ffmpeg: {stripped}", flush=True)
                elif kept == 41:
                    print("audio ffmpeg: (further lines dropped)", flush=True)
        except Exception:  # noqa: BLE001 - a log relay never takes the stage down
            pass
        finally:
            try:
                pipe.close()
            except Exception:  # noqa: BLE001
                pass

    def chunks(self) -> Iterable[np.ndarray]:
        """Yields float32 arrays of `hop_samples` samples in [-1, 1]."""
        hop_bytes = self.hop_samples * BYTES_PER_SAMPLE
        delivered = 0
        while True:
            if self._stopped():
                return
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
                delivered = 0
            buffer = self.proc.stdout.read(hop_bytes)
            if len(buffer) < hop_bytes:
                self.proc.stdout.close()
                self.proc.wait()
                self.proc = None
                if not self.network:
                    return  # a file ended
                if delivered == 0:
                    # ffmpeg found no audio track (or the relay closed at
                    # once): not an error, the stream is silent to us.
                    self.telemetry.count("audioNoTrack")
                    self._wait(NO_TRACK_RETRY_S)
                else:
                    self.telemetry.count("audioReconnects")
                    self._wait(1.0)
                continue
            delivered += len(buffer)
            yield np.frombuffer(buffer, dtype="<i2").astype(np.float32) / 32768.0

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stopped():
                return
            self.sleep(min(0.25, deadline - time.monotonic()))

    def _stopped(self) -> bool:
        return self.closed or (self.stopping is not None and self.stopping.is_set())

    def stop(self) -> None:
        """No more chunks: the loop ends instead of reconnecting once the
        killed ffmpeg's short read comes back."""
        self.closed = True
        if self.proc is not None:
            self.proc.kill()


def _round_scores(raw: dict) -> dict[str, float]:
    return {label: round(float(raw.get(label, 0.0)), 6) for label in TARGET_LABELS}


class AudioStage(threading.Thread):
    """Consumes PCM chunks, emits `audio` messages, answers `classify`.

    `classifier` has `classify(wav) -> {label: score}` (ced.CedEngine, or a
    stand-in); `emit` receives every outgoing message; `stream_clock` returns
    the producer's current stream time so the analysis can line the audio
    clock up with the frames.
    """

    def __init__(self, source, classifier, emit: Callable[[dict], None],
                 telemetry, stream_clock: Callable[[], float | None] = lambda: None,
                 pitch_estimator=estimate_pitch, hop_s: float = HOP_S,
                 window_s: float = WINDOW_S, history_s: float = HISTORY_S):
        super().__init__(daemon=True, name="audio")
        self.source = source
        self.classifier = classifier
        self.emit = emit
        self.telemetry = telemetry
        self.stream_clock = stream_clock
        self.pitch_estimator = pitch_estimator
        self.hop_s = hop_s
        self.window_samples = int(window_s * SAMPLE_RATE)
        self.history_samples = int(history_s * SAMPLE_RATE)
        self.history = np.empty(0, dtype=np.float32)
        self.total_samples = 0
        self.last_frame_time_s = -1.0
        self.hops = 0
        self.requests: queue.Queue = queue.Queue(maxsize=REQUEST_QUEUE_DEPTH)
        self.stopping = threading.Event()

    # -- the thread ---------------------------------------------------------

    def run(self) -> None:
        try:
            for pcm in self.source.chunks():
                if self.stopping.is_set():
                    break
                self.feed(pcm)
        finally:
            # The session is over: pending requests are told so, and the
            # samples go with it.
            self._serve(drain=True)
            self.history = np.empty(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> None:
        """One chunk in: its hop message out, then whatever requests are
        waiting. What `run` does per chunk; tests call it directly."""
        try:
            self._step(pcm)
        except Exception as error:  # noqa: BLE001 - said aloud, survived
            self.telemetry.count("audioErrors")
            print(f"audio: hop failed at {self.end_s:.1f}s: {error!r}",
                  flush=True)
        self._serve()

    def stop(self) -> None:
        self.stopping.set()
        self.source.stop()

    # -- per hop ------------------------------------------------------------

    @property
    def end_s(self) -> float:
        return self.total_samples / SAMPLE_RATE

    def _step(self, pcm: np.ndarray) -> None:
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        start_s = self.end_s
        self.total_samples += len(pcm)
        end_s = self.end_s
        self.history = np.concatenate((self.history, pcm))[-self.history_samples:]
        window = self.history[-self.window_samples:]
        with self.telemetry.time_stage("audio"):
            scores = _round_scores(self.classifier.classify(window))
            pitch = self.pitch_estimator(window, SAMPLE_RATE)
            frames = self._fresh_frames(start_s, end_s)
        self.hops += 1
        self.telemetry.count("audioHops")
        self.telemetry.gauge("audioS", end_s)
        # The level of the window the classifier just saw: what the stage
        # is hearing at all, in the telemetry line. -90 is digital silence.
        self.telemetry.gauge("audioDbfs", float(pitch.get("loudnessDbfs") or -90.0))
        stream_s = self.stream_clock()
        if stream_s is not None:
            self.telemetry.gauge("audioLagS", float(stream_s) - end_s)
        self.emit({
            "kind": "audio",
            "atS": round(end_s, 3),
            "streamS": round(float(stream_s), 3) if stream_s is not None else None,
            "hopS": round(len(pcm) / SAMPLE_RATE, 3),
            "windowS": round(len(window) / SAMPLE_RATE, 3),
            "scores": scores,
            "pitch": {
                "pitchHz": pitch.get("pitchHz"),
                "pitchConfidence": pitch.get("pitchConfidence"),
                "voicedFramePct": pitch.get("voicedFramePct"),
                "loudnessDbfs": pitch.get("loudnessDbfs"),
            },
            "frames": frames,
        })

    def _fresh_frames(self, start_s: float, end_s: float) -> list[list]:
        """The frame contour of the new samples: `[time, dBFS, Hz | null]`
        per 16 ms frame, timestamped at the frame centre on the audio
        clock. Frames already sent (their centre at or before the last
        one's) are not repeated."""
        history_start_s = end_s - len(self.history) / SAMPLE_RATE
        window_start_s = max(history_start_s, start_s - FRAME_CONTEXT_S)
        window = self.history[int((window_start_s - history_start_s) * SAMPLE_RATE):]
        fresh = []
        for frame in pitch_frames(window, SAMPLE_RATE, min_dbfs=FRAME_MIN_DBFS):
            at = window_start_s + frame.time_s
            if at <= self.last_frame_time_s + 1e-6:
                continue
            self.last_frame_time_s = at
            fresh.append([
                round(at, 3),
                round(float(frame.dbfs), 1),
                round(float(frame.pitch_hz), 1) if frame.pitch_hz is not None else None,
            ])
        return fresh

    # -- segments -----------------------------------------------------------

    def request(self, message: dict) -> None:
        """A `classify` message from the analysis, on any thread. Served on
        the stage thread after the current hop; refused when the queue is
        full."""
        try:
            self.requests.put_nowait(message)
        except queue.Full:
            self.telemetry.count("audioSegmentErrors")
            self.emit({"kind": "segment", "id": message.get("id"),
                       "error": "busy"})

    def _serve(self, drain: bool = False) -> None:
        while True:
            try:
                message = self.requests.get_nowait()
            except queue.Empty:
                return
            if drain:
                self.emit({"kind": "segment", "id": message.get("id"),
                           "error": "closed"})
                continue
            self.emit(self.segment(message))

    def segment(self, message: dict) -> dict:
        """Measure one span of the history: `{"id", "fromS", "toS"}` in.
        Numbers only out; an `error` field instead when the span is not
        available (`expired`: older than the history; `span`: malformed or
        too long; `empty`: nothing decoded there yet)."""
        request_id = message.get("id")
        try:
            from_s = float(message["fromS"])
            to_s = float(message["toS"])
        except (KeyError, TypeError, ValueError):
            return self._refuse(request_id, "span")
        if not (to_s > from_s) or to_s - from_s > MAX_SEGMENT_S or from_s < 0:
            return self._refuse(request_id, "span")
        history_start_s = self.end_s - len(self.history) / SAMPLE_RATE
        if from_s < history_start_s - 1e-6:
            return self._refuse(request_id, "expired")
        to_s = min(to_s, self.end_s)
        lo = int((from_s - history_start_s) * SAMPLE_RATE)
        hi = int((to_s - history_start_s) * SAMPLE_RATE)
        segment = self.history[lo:hi]
        if not segment.size:
            return self._refuse(request_id, "empty")
        # The classifier wants at least CED_MIN_CONTEXT_S; a shorter sound is
        # centred in its surroundings.
        min_samples = int(CED_MIN_CONTEXT_S * SAMPLE_RATE)
        ced_segment = segment
        if ced_segment.size < min_samples:
            pad_s = (min_samples - ced_segment.size) / 2 / SAMPLE_RATE
            pad_lo = max(0, int((max(history_start_s, from_s - pad_s)
                                 - history_start_s) * SAMPLE_RATE))
            ced_segment = self.history[pad_lo:pad_lo + min_samples]
        with self.telemetry.time_stage("audioSegment"):
            scores = _round_scores(self.classifier.classify(ced_segment))
            top = max(scores.items(), key=lambda kv: kv[1])
            measured = {
                "kind": "segment",
                "id": request_id,
                "fromS": round(from_s, 3),
                "toS": round(to_s, 3),
                "ced": {"topLabel": top[0], "topScore": top[1], "scores": scores},
                "pitch": pitch_stats(segment),
                "loudness": loudness_stats(segment),
                "spectral": spectral_stats(segment),
            }
        self.telemetry.count("audioSegments")
        return measured

    def _refuse(self, request_id, reason: str) -> dict:
        self.telemetry.count("audioSegmentErrors")
        return {"kind": "segment", "id": request_id, "error": reason}
