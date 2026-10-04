"""The audio stage: the only code in the enclave that reads audio samples.

The stream the producer decodes for its frames may carry an audio track
(the phone's microphone published next to its camera, or a home camera's
own microphone). This module pulls that track through ffmpeg as 16 kHz mono
PCM, keeps the most recent `HISTORY_S` seconds of it in memory, and turns
it into numbers for the analysis process (analysis/protocol.md):

- every `HOP_S`, an `audio` message: the classifier's scores for the
  non-speech vocalization labels (`ced.TARGET_LABELS`) over the trailing
  `WINDOW_S` window, the whole class table over the same window (`all`, one
  score per class in the model's order, so the analysis can read the room:
  music, a fan, rain, a machine's hum), a pitch and level summary of the
  window, the per-frame level and pitch contour of the new samples, and,
  on a hop that ends on a whole second, the spectral shape and level of
  that second (`spectral`, audio_features.second_spectral);
- on request (`classify`, from the analysis), a `segment` message: the
  same measurements over one short span the analysis names, taken from the
  history, so that a sound it proposed from the frame contour can be typed
  and measured as a whole;
- with a second tagger (beats.py), every `TAGGER_HOP_S` of audio one
  `beats` message per window it reads (a session's are `TAGGER_WINDOWS_S`,
  one engine each): that model's scores for the 527 AudioSet classes over
  the trailing window, and its 768-value summary of the same window (the
  mean of its encoder's last layer over the window's patches). The tagger
  runs on its own thread (`TaggerWorker`), so a slow window never delays
  the hop's `audio` message.

Samples never leave this process: the history is discarded as it ages out
and when the session ends, and no sample is recorded. What crosses the
socket is listed exhaustively in the protocol document.

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
                            second_spectral, spectral_stats)
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
# The second tagger (beats.py): one window every TAGGER_HOP_S of audio, each
# the trailing TAGGER_WINDOW_S ending exactly on a multiple of the hop
# (atS 1.0, 2.0, ...), zeros on the left before the stream has that much.
TAGGER_HOP_S = 1.0
TAGGER_WINDOW_S = 2.0
# A session's tagger reads one window of each of these lengths at every
# hop, in this order. The first is the window it read alone before, the
# one `hello.beatsWindowS` names: its messages and its telemetry's names
# are unchanged.
TAGGER_WINDOWS_S = (2.0, 3.0)
# Its 527 scores to three significant digits rather than a fixed number of
# decimals: most of a window's classes score far below a thousandth, and
# what tells them apart is their relative size, which four decimals erase;
# the embedding's 768 values (of order one) to three decimals. The record
# carries one row a second, about 9 KB of JSON before gzip.
TAGGER_SCORE_DIGITS = 3
TAGGER_EMBEDDING_DECIMALS = 3


def _significant(value: float, digits: int) -> float:
    return float(f"{value:.{digits}g}")


class AudioSource:
    """ffmpeg's 16 kHz mono s16le output off the same stream the frames come
    from, in hop-sized chunks, with reconnect for a stream and one pass, to
    its end, for a file. `-vn` drops the video: the decode for frames is the
    producer's own ffmpeg, this one reads only the audio track."""

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
        """Yields float32 arrays of `hop_samples` samples in [-1, 1].

        One ffmpeg is read to the end of its pipe before anything is decided
        about it: the short read is the end, whatever `poll()` says
        meanwhile. ffmpeg exits with its last seconds still in the pipe,
        and a loop that respawned on exit status read a stream's tail into
        the void on every reconnect and started a file over from its first
        second, forever (the ended child never got to say so). A file is
        read once and the generator ends; a stream reconnects after its
        short read, as before.
        """
        hop_bytes = self.hop_samples * BYTES_PER_SAMPLE
        delivered = 0
        while True:
            if self._stopped():
                return
            if self.proc is None:
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


# The whole class table is rounded to this many decimals: 527 scores a hop
# stay near 3 KB of JSON, and a fourth decimal of a class probability is
# below anything the analysis reads.
ALL_SCORES_DECIMALS = 4


def classify_scores(classifier, wav: np.ndarray) -> tuple[dict[str, float], list[float] | None]:
    """One classifier call: the TARGET_LABELS scores, and the whole table
    when the classifier has one (`classify_all` and `target_scores`, as
    ced.CedEngine; a stand-in with only `classify` gives no table)."""
    classify_all = getattr(classifier, "classify_all", None)
    if classify_all is None:
        return _round_scores(classifier.classify(wav)), None
    vector = classify_all(wav)
    scores = _round_scores(classifier.target_scores(vector))
    return scores, [round(float(value), ALL_SCORES_DECIMALS) for value in vector]


class TaggerWorker(threading.Thread):
    """The second tagger's windows, one hop at a time, off the stage thread.

    `taggers` pairs each window length with what tags windows of that
    length (one beats.BeatsEngine each, or a stand-in). A hop has one
    window of each, tagged in that order, so a later window never delays
    the first's message. `offer` hands over a hop's windows and returns at
    once. A hop offered while the previous one is still being tagged
    waits; a newer one replaces a waiting one (its windows counted,
    `beatsDropped`), so the tagger never holds up the stage and never
    falls more than one hop behind. Each tagged window is one `beats`
    message: its end on the audio clock (`atS`), the stream clock when it
    was sent (`streamS`, and `lagS`, their difference: the window's whole
    delay, decode and wait and compute), its length (`windowS`), the
    compute time (`computeMs`), the scores and the embedding. The first
    window's compute is the `beats` stage and its `beatsMs` and
    `beatsLagS` gauges; a later window's carry its length (`beatsW3`,
    `beatsW3Ms`, `beatsW3LagS`). `beatsWindows`, `beatsDropped` and
    `beatsErrors` count windows of every length.

    `synchronous` tags each hop's windows inside `offer` instead, on the
    caller's thread: every window, in order, none dropped. That is the
    offline replay's mode and the tests'; a session never uses it.
    """

    def __init__(self, taggers, emit: Callable[[dict], None], telemetry,
                 stream_clock: Callable[[], float | None],
                 hop_s: float = TAGGER_HOP_S, synchronous: bool = False):
        super().__init__(daemon=True, name="audio-tagger")
        self.taggers = tuple((float(window_s), tagger) for window_s, tagger in taggers)
        if not self.taggers:
            raise ValueError("a tagger worker needs a tagger")
        self.names = tuple("beats" if index == 0 else f"beatsW{window_s:g}"
                           for index, (window_s, _) in enumerate(self.taggers))
        self.emit = emit
        self.telemetry = telemetry
        self.stream_clock = stream_clock
        self.hop_s = hop_s
        self.synchronous = synchronous
        self._ready = threading.Condition()
        self._pending: tuple[float, list[np.ndarray]] | None = None
        self._stopping = False
        self.windows = 0
        self.dropped = 0

    @property
    def windows_s(self) -> tuple[float, ...]:
        return tuple(window_s for window_s, _ in self.taggers)

    def start(self) -> None:
        if not self.synchronous:
            super().start()

    def offer(self, at_s: float, windows: list[np.ndarray]) -> None:
        """One hop's windows, one per tagger in `taggers`' order."""
        if self.synchronous:
            self._tag_hop(at_s, windows)
            return
        with self._ready:
            if self._stopping:
                return
            if self._pending is not None:
                replaced = len(self._pending[1])
                self.dropped += replaced
                self.telemetry.count("beatsDropped", replaced)
            self._pending = (at_s, windows)
            self._ready.notify()

    def run(self) -> None:
        while True:
            with self._ready:
                while self._pending is None and not self._stopping:
                    self._ready.wait()
                if self._stopping:
                    return
                at_s, windows = self._pending
                self._pending = None
            self._tag_hop(at_s, windows)

    def _tag_hop(self, at_s: float, windows: list[np.ndarray]) -> None:
        for index, window in enumerate(windows):
            try:
                self._tag(index, at_s, window)
            except Exception as error:  # noqa: BLE001 - said aloud, survived
                self.telemetry.count("beatsErrors")
                print(f"audio: tagger failed at {at_s:.1f}s "
                      f"({self.taggers[index][0]:g} s window): {error!r}", flush=True)

    def stop(self, timeout_s: float = 5.0) -> None:
        """No more windows: the hop waiting is dropped, the one being tagged
        finishes, and the thread ends."""
        with self._ready:
            self._stopping = True
            self._pending = None
            self._ready.notify()
        if self.is_alive():
            self.join(timeout=timeout_s)

    def _tag(self, index: int, at_s: float, window: np.ndarray) -> None:
        window_s, tagger = self.taggers[index]
        name = self.names[index]
        started = time.monotonic()
        with self.telemetry.time_stage(name):
            scores, embedding = tagger.tag(window)
        compute_ms = (time.monotonic() - started) * 1000.0
        stream_s = self.stream_clock()
        lag_s = float(stream_s) - at_s if stream_s is not None else None
        self.windows += 1
        self.telemetry.count("beatsWindows")
        self.telemetry.gauge(f"{name}Ms", round(compute_ms, 1))
        if lag_s is not None:
            self.telemetry.gauge(f"{name}LagS", round(lag_s, 3))
        self.emit({
            "kind": "beats",
            "atS": round(at_s, 3),
            "streamS": round(float(stream_s), 3) if stream_s is not None else None,
            "lagS": round(lag_s, 3) if lag_s is not None else None,
            "hopS": self.hop_s,
            "windowS": window_s,
            "computeMs": round(compute_ms, 1),
            "scores": [_significant(float(value), TAGGER_SCORE_DIGITS) for value in scores],
            "embedding": [round(float(value), TAGGER_EMBEDDING_DECIMALS) for value in embedding],
        })


class AudioStage(threading.Thread):
    """Consumes PCM chunks, emits `audio` messages, answers `classify`.

    `classifier` has `classify(wav) -> {label: score}` (ced.CedEngine, or a
    stand-in) and, when it can, `classify_all(wav)` for the whole class
    table; `emit` receives every outgoing message; `stream_clock` returns
    the producer's current stream time so the analysis can line the audio
    clock up with the frames. `taggers`, when given, pairs window lengths
    with what has `tag(window) -> (scores, summary)` over windows of that
    length (beats.BeatsEngine, or a stand-in), each run every
    `tagger_hop_s` of audio on one `TaggerWorker`; `tagger` with
    `tagger_window_s` is the same for one window.
    """

    def __init__(self, source, classifier, emit: Callable[[dict], None],
                 telemetry, stream_clock: Callable[[], float | None] = lambda: None,
                 pitch_estimator=estimate_pitch, hop_s: float = HOP_S,
                 window_s: float = WINDOW_S, history_s: float = HISTORY_S,
                 tagger=None, tagger_hop_s: float = TAGGER_HOP_S,
                 tagger_window_s: float = TAGGER_WINDOW_S,
                 tagger_synchronous: bool = False, taggers=None):
        super().__init__(daemon=True, name="audio")
        if tagger is not None and taggers:
            raise ValueError("tagger or taggers, not both")
        if taggers:
            taggers = tuple((float(length_s), engine) for length_s, engine in taggers)
        elif tagger is not None:
            taggers = ((float(tagger_window_s), tagger),)
        else:
            taggers = ()
        self.source = source
        self.classifier = classifier
        self.emit = emit
        self.telemetry = telemetry
        self.stream_clock = stream_clock
        self.pitch_estimator = pitch_estimator
        self.hop_s = hop_s
        self.window_samples = int(window_s * SAMPLE_RATE)
        # The history must hold the tagger's longest window as well as the
        # classifier's and the `classify` spans'.
        longest_s = max((length_s for length_s, _ in taggers), default=0.0)
        self.history_samples = int(max(history_s, longest_s) * SAMPLE_RATE)
        self.history = np.empty(0, dtype=np.float32)
        self.total_samples = 0
        self.last_frame_time_s = -1.0
        self.hops = 0
        self.requests: queue.Queue = queue.Queue(maxsize=REQUEST_QUEUE_DEPTH)
        self.stopping = threading.Event()
        self.tagger_hop_samples = max(1, int(round(tagger_hop_s * SAMPLE_RATE)))
        self.tagger_window_samples = tuple(int(round(length_s * SAMPLE_RATE))
                                           for length_s, _ in taggers)
        self.tagger = (TaggerWorker(taggers, emit, telemetry, stream_clock,
                                    hop_s=tagger_hop_s, synchronous=tagger_synchronous)
                       if taggers else None)

    # -- the thread ---------------------------------------------------------

    def run(self) -> None:
        if self.tagger is not None:
            self.tagger.start()
        try:
            for pcm in self.source.chunks():
                if self.stopping.is_set():
                    break
                self.feed(pcm)
        finally:
            # The session is over: pending requests are told so, and the
            # samples go with it (the tagger's waiting windows too).
            self._serve(drain=True)
            if self.tagger is not None:
                self.tagger.stop()
            self.history = np.empty(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> None:
        """One chunk in: its hop message out, the tagger's windows that end
        in it handed over, then whatever requests are waiting. What `run`
        does per chunk; tests call it directly."""
        start_samples = self.total_samples
        try:
            self._step(pcm)
        except Exception as error:  # noqa: BLE001 - said aloud, survived
            self.telemetry.count("audioErrors")
            print(f"audio: hop failed at {self.end_s:.1f}s: {error!r}",
                  flush=True)
        if self.tagger is not None and self.total_samples > start_samples:
            self._offer_tagger_windows(start_samples)
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
        # The hop's costs, each its own stage beside the whole: the
        # classifier over the window, the pitch summary over the window,
        # the contour of the new samples, and on a whole second the
        # second's spectrum. 2026-09-16 the whole read 750 ms (p50) per
        # 0.5 s hop and nothing said which part; the stage fell to half
        # real time and the pre-current baseline minute took two.
        spectral = None
        with self.telemetry.time_stage("audio"):
            with self.telemetry.time_stage("audioClassify"):
                scores, table = classify_scores(self.classifier, window)
            with self.telemetry.time_stage("audioPitch"):
                pitch = self.pitch_estimator(window, SAMPLE_RATE)
            with self.telemetry.time_stage("audioFrames"):
                frames = self._fresh_frames(start_s, end_s)
            if len(pcm) and self.total_samples % SAMPLE_RATE == 0:
                # The second [end_s - 1, end_s), on the hop that ends on it:
                # every other hop of the reader's.
                with self.telemetry.time_stage("audioSpectral"):
                    spectral = second_spectral(self.history[-SAMPLE_RATE:])
        self.hops += 1
        self.telemetry.count("audioHops")
        self.telemetry.gauge("audioS", end_s)
        # The level of the window the classifier just saw: what the stage
        # is hearing at all, in the telemetry line. -90 is digital silence.
        self.telemetry.gauge("audioDbfs", float(pitch.get("loudnessDbfs") or -90.0))
        stream_s = self.stream_clock()
        if stream_s is not None:
            self.telemetry.gauge("audioLagS", float(stream_s) - end_s)
        message = {
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
        }
        if table is not None:
            message["all"] = table
        if spectral is not None:
            message["spectral"] = spectral
        self.emit(message)

    def _offer_tagger_windows(self, start_samples: int) -> None:
        """The tagger's windows that end inside this hop: at each multiple of
        its hop, one of each of its lengths, each the trailing window ending
        exactly on that multiple, zero-padded on the left before the stream
        has that much audio."""
        hop = self.tagger_hop_samples
        history_start = self.total_samples - len(self.history)
        end = (start_samples // hop + 1) * hop
        while end <= self.total_samples:
            windows = []
            for size in self.tagger_window_samples:
                lo = end - size
                window = self.history[max(0, lo - history_start):end - history_start]
                if len(window) < size:
                    window = np.concatenate((np.zeros(size - len(window), dtype=np.float32), window))
                windows.append(np.array(window, dtype=np.float32))
            self.tagger.offer(end / SAMPLE_RATE, windows)
            end += hop

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
            scores, table = classify_scores(self.classifier, ced_segment)
            top = max(scores.items(), key=lambda kv: kv[1])
            ced = {"topLabel": top[0], "topScore": top[1], "scores": scores}
            if table is not None:
                ced["all"] = table
            measured = {
                "kind": "segment",
                "id": request_id,
                "fromS": round(from_s, 3),
                "toS": round(to_s, 3),
                "ced": ced,
                "pitch": pitch_stats(segment),
                "loudness": loudness_stats(segment),
                "spectral": spectral_stats(segment),
            }
        self.telemetry.count("audioSegments")
        return measured

    def _refuse(self, request_id, reason: str) -> dict:
        self.telemetry.count("audioSegmentErrors")
        return {"kind": "segment", "id": request_id, "error": reason}
