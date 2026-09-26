"""The session's record: the enclave's half of a masseuse.ai session record.

A leased session's capture does not stay on the slot. The trainer's lease
names a prefix under the attested capture bucket (`TEE_CAPTURE_BUCKET`),
`{account}/estim_sessions/{session}/enclave`, and every stream this
process produces is written there as write-once parts on a 30-second
wall-clock grid, uploaded by a background thread as each part closes, so a
slot that dies loses at most the part it was writing. The trainer writes
its own half next door (masseuse-trainer/server/session-record.js) on the
same grid and with the same part names, so the two halves line up by name.

What goes in (analysis/protocol.md names the messages):

    poses/, faces/     every keypoint the model emitted for the body and
                       face views - all 308 of them, with their scores -
                       one row per posed frame, plus keypoint-less rows
                       for dropped and errored steps; Parquet, one file per
                       window, x/y/score as fixed lists of 308 float32
                       (zstd, byte-stream-split), the layout in hello.json
                       and in the file's metadata
    frames/            the `frame` messages: the assembler's 21-point rows
                       at the stream's frame rate with the regional motion
                       descriptors
    audio/, segments/  the audio stage's measurements and the classified
                       spans it was asked for
    vocal/             the analysis's `vocal` rows, one per judgement
    onsets/, events/,  what the analysis decided, as the flat capture has
    payloads/, posts/  always kept them
    telemetry/         the process's counters and gauges, once a second
    hello.json         what this session was: the hello and ready of the
                       analysis link, the keypoint layout, the image
    summary.json       the session's summary when it ends

Everything but the Parquet is gzipped JSONL, one object per line, each
stamped `wallS` (unix seconds) at the append. Nothing here is a frame or
a sample: the pixels and the audio never leave the pipeline, and the
record is made of what the analysis and the models said about them.

Every write is best effort in the sink sense - a full disk, a refused
upload or a missing library is counted and said aloud, never raised into
the pose or descriptors threads - but it is not optional: a lease that
carries a record prefix on a slot without a capture bucket is refused
(tee_mode.Lease), so the trainer knows nothing was kept.
"""

from __future__ import annotations

import gzip
import json
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_PART_S = 30
KEYPOINT_COUNT = 308
# How long the uploader keeps trying one part (transient GCS errors) before
# it leaves the file where it is and moves on; the file is then listed as
# failed in the summary and dies with the instance.
UPLOAD_ATTEMPTS = 6
UPLOAD_BACKOFF_S = 2.0
# Streams closed with the session: how long close() waits for the queue.
CLOSE_TIMEOUT_S = 25.0

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# The trainer's name for a session's enclave half (session-record.js
# `enclavePrefix`): the account's uuid, then the session's, then `enclave`.
RECORD_PREFIX = re.compile(rf"^{UUID}/estim_sessions/{UUID}/enclave$")
BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")

JSONL_STREAMS = ("frames", "audio", "segments", "vocal", "onsets", "events",
                 "payloads", "posts", "telemetry")
KEYPOINT_STREAMS = {"body": "poses", "face": "faces"}


def window_start_s(wall_s: float, part_s: int = DEFAULT_PART_S) -> int:
    """The start (unix seconds) of the part window `wall_s` falls in."""
    part = max(1, int(part_s))
    return int(wall_s // part) * part


def part_name(window_start: int) -> str:
    """`part-20260915T051230Z` for the window starting at that UTC second:
    the trainer's name for its own parts, so the halves sort together."""
    return "part-" + datetime.fromtimestamp(
        int(window_start), tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_record(value, part_s: int = DEFAULT_PART_S) -> tuple[dict | None, str]:
    """The lease body's `record`: `{prefix, partSeconds}` validated, or
    (None, why). A lease without one is a session without a record."""
    if value is None:
        return None, ""
    if not isinstance(value, dict):
        return None, "record must be an object"
    prefix = str(value.get("prefix") or "").strip("/")
    if not RECORD_PREFIX.match(prefix):
        return None, ("record.prefix must be "
                      "{uuid}/estim_sessions/{uuid}/enclave")
    seconds = value.get("partSeconds", part_s)
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return None, "record.partSeconds must be an integer number of seconds"
    if not 5 <= seconds <= 600:
        return None, "record.partSeconds must be between 5 and 600"
    return {"prefix": prefix, "partSeconds": seconds}, ""


def keypoint_layout() -> dict:
    """Which index is which, for hello.json and the Parquet metadata: the
    body block by name (pixel/keypoints.py), the hands and the face by
    range. The face's 245 are in the model's own order (`LABEL_63` ..
    `LABEL_307`); the hands' internal order is the standard 21-point
    topology by hypothesis, not by check."""
    from keypoints import (BODY, KEYPOINT_NAMES, LEFT_HAND,  # noqa: PLC0415
                           MIN_KEYPOINT_SCORE, RIGHT_HAND)
    return {
        "count": KEYPOINT_COUNT,
        "coordinates": "pixels of the decoded frame (frameW x frameH), "
                       "origin top-left; score is the model's confidence",
        "body": {str(i): KEYPOINT_NAMES[i] for i in BODY},
        "leftHand": [LEFT_HAND[0], LEFT_HAND[-1]],
        "rightHand": [RIGHT_HAND[0], RIGHT_HAND[-1]],
        "handOrderVerified": False,
        "face": [RIGHT_HAND[-1] + 1, KEYPOINT_COUNT - 1],
        "faceOrder": "the model's (LABEL_63 .. LABEL_307)",
        "minKeypointScore": MIN_KEYPOINT_SCORE,
    }


# -- one stream's parts ---------------------------------------------------------------


class PartStream:
    """One stream of the record: rows into the part of the wall-clock window
    they arrive in, the part closed and handed to the uploader when the
    window ends (or on close). Thread-safe; appends never block on I/O
    beyond the local write."""

    extension = "jsonl.gz"
    content_type = "application/gzip"

    def __init__(self, name: str, directory: Path, part_s: int,
                 clock=time.time, on_part=None, on_error=None):
        self.name = name
        self.directory = directory / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.part_s = max(1, int(part_s))
        self.clock = clock
        self.on_part = on_part
        self.on_error = on_error or (lambda text: None)
        self.lock = threading.Lock()
        self.window: int | None = None
        self.rows = 0
        self.part_rows = 0
        self.parts = 0
        self.late = 0
        self.dropped = 0
        self.errors = 0
        self.closed = False

    # -- the grid ---------------------------------------------------------------

    def _place(self, wall_s: float) -> None:
        """Under the lock: the part `wall_s` belongs to is the open one."""
        window = window_start_s(wall_s, self.part_s)
        if self.window is None:
            self._open(window)
        elif window > self.window:
            self._roll()
            self._open(window)
        elif window < self.window:
            # Stamped before a roll it lost the race with: it rides in the
            # open part, its own wallS telling the truth.
            self.late += 1

    def _open(self, window: int) -> None:
        self.window = window
        self.part_rows = 0
        self.path = self.directory / f"{part_name(window)}.{self.extension}"
        self._begin()

    def _roll(self) -> None:
        """Under the lock: close the open part and hand it on."""
        if self.window is None:
            return
        window, path, rows = self.window, self.path, self.part_rows
        self.window = None
        try:
            self._finish()
        except Exception as error:  # noqa: BLE001 - counted, said, survived
            self.errors += 1
            self.on_error(f"record: {self.name} part {path.name} failed: {error!r}")
            return
        if rows == 0:
            path.unlink(missing_ok=True)
            return
        self.parts += 1
        if self.on_part is not None:
            self.on_part(self, path, window)

    def tick(self, now: float | None = None) -> None:
        """Close the open part once its window has ended: called every
        second by the record, so a sparse stream's part does not wait for
        its next row to reach the bucket."""
        now = self.clock() if now is None else now
        with self.lock:
            if (self.window is not None
                    and now >= self.window + self.part_s and not self.closed):
                self._roll()

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self._roll()

    def snapshot(self) -> dict:
        with self.lock:
            return {"rows": self.rows, "parts": self.parts, "late": self.late,
                    "dropped": self.dropped, "errors": self.errors}

    # -- the format ---------------------------------------------------------------

    def _begin(self) -> None:
        self._file = gzip.open(self.path, "wb", compresslevel=6)

    def _finish(self) -> None:
        self._file.close()

    def append(self, row: dict) -> bool:
        """One row, stamped `wallS` now unless it carries one."""
        with self.lock:
            if self.closed:
                self.dropped += 1
                return False
            wall_s = row.get("wallS")
            if not isinstance(wall_s, (int, float)):
                wall_s = round(self.clock(), 3)
                row = {"wallS": wall_s, **row}
            try:
                self._place(float(wall_s))
                self._file.write(
                    (json.dumps(row, separators=(",", ":")) + "\n").encode())
            except Exception as error:  # noqa: BLE001 - counted, said, survived
                self.errors += 1
                self.on_error(f"record: {self.name} append failed: {error!r}")
                return False
            self.rows += 1
            self.part_rows += 1
            return True


class KeypointStream(PartStream):
    """A view's keypoints as Parquet parts: one row per pose step, x, y and
    score as lists of 308 float32 (null when the step had none), the box,
    the frame size and the flags beside them. Rows are banked as numpy and
    written when the part closes."""

    extension = "parquet"
    content_type = "application/vnd.apache.parquet"

    def __init__(self, name: str, view: str, directory: Path, part_s: int,
                 metadata: dict | None = None, **kwargs):
        self.view = view
        # The file's own account of itself: the view, the keypoint layout,
        # whatever the record adds (the session, the image).
        self.metadata = {"view": view}
        try:
            self.metadata["keypoints"] = json.dumps(keypoint_layout(),
                                                    separators=(",", ":"))
        except Exception:  # noqa: BLE001 - the layout is a courtesy
            pass
        self.metadata.update({key: str(value) for key, value in (metadata or {}).items()})
        super().__init__(name, directory, part_s, **kwargs)

    def _begin(self) -> None:
        self._bank: list[tuple] = []

    def _finish(self) -> None:
        bank, self._bank = self._bank, []
        if not bank:
            return
        write_keypoint_part(self.path, self.view, bank, self.metadata)

    def append_step(self, frame_index: int, at_s: float, keypoints, scores,
                    box, box_score, people, unresolved: bool,
                    frame_size=None, *, dropped: bool = False,
                    error: str | None = None, wall_s: float | None = None) -> bool:
        """One pose step of this view: the full result (keypoints `(308, 2)`,
        scores `(308,)`, both None when nobody was found) or a gap."""
        if keypoints is not None:
            xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
            sc = np.asarray(scores, dtype=np.float32).reshape(-1)
            if xy.shape[0] != KEYPOINT_COUNT or sc.shape[0] != KEYPOINT_COUNT:
                # A model with another layout: the row is kept keypoint-less
                # and the mismatch is said once per part in the error column.
                xy, sc = None, None
                error = error or f"keypoints:{np.asarray(keypoints).shape}"
        else:
            xy, sc = None, None
        with self.lock:
            if self.closed:
                self.dropped += 1
                return False
            wall = float(wall_s) if isinstance(wall_s, (int, float)) else self.clock()
            try:
                self._place(wall)
            except Exception as error_:  # noqa: BLE001 - counted, said, survived
                self.errors += 1
                self.on_error(f"record: {self.name} part failed: {error_!r}")
                return False
            self._bank.append((
                int(frame_index), float(at_s), round(wall, 3),
                frame_size, box, box_score, people, bool(unresolved),
                bool(dropped), error, xy, sc))
            self.rows += 1
            self.part_rows += 1
            return True


def write_keypoint_part(path: Path, view: str, bank: list[tuple],
                        metadata: dict | None = None) -> None:
    """The Parquet for one closed part of a keypoint stream."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    n = len(bank)
    frame_w = [None] * n
    frame_h = [None] * n
    box_cols: list[list] = [[None] * n for _ in range(5)]
    people = [None] * n
    xy_all = np.full((n, KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
    sc_all = np.full((n, KEYPOINT_COUNT), np.nan, dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, (_, _, _, size, box, box_score, count, _, _, _, xy, sc) in enumerate(bank):
        if size is not None:
            try:
                frame_w[i], frame_h[i] = int(size[0]), int(size[1])
            except (TypeError, ValueError, IndexError):
                pass
        if box is not None:
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
                box_cols[0][i], box_cols[1][i] = x0, y0
                box_cols[2][i], box_cols[3][i] = x1 - x0, y1 - y0
            except (TypeError, ValueError):
                pass
        if box_score is not None:
            try:
                box_cols[4][i] = float(box_score)
            except (TypeError, ValueError):
                pass
        if count is not None:
            try:
                people[i] = int(count)
            except (TypeError, ValueError):
                pass
        if xy is not None:
            xy_all[i] = xy
            sc_all[i] = sc
            has[i] = True

    def lists(values: np.ndarray) -> "pa.Array":
        # list<float32> of 308 per posed row, null for a row without a
        # result: a fixed-size list would say the 308 in the type, but
        # Parquet stores either as a repeated group and reads a null
        # fixed-size entry back as an empty one, which does not load.
        flat = pa.array(values[has].reshape(-1), type=pa.float32())
        lengths = np.where(has, KEYPOINT_COUNT, 0).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), flat,
                                        mask=pa.array(~has, type=pa.bool_()))

    table = pa.table({
        "frame": pa.array([row[0] for row in bank], type=pa.int32()),
        "atS": pa.array([row[1] for row in bank], type=pa.float64()),
        "wallS": pa.array([row[2] for row in bank], type=pa.float64()),
        "view": pa.array([view] * n, type=pa.string()),
        "frameW": pa.array(frame_w, type=pa.int16()),
        "frameH": pa.array(frame_h, type=pa.int16()),
        "boxX": pa.array(box_cols[0], type=pa.float32()),
        "boxY": pa.array(box_cols[1], type=pa.float32()),
        "boxW": pa.array(box_cols[2], type=pa.float32()),
        "boxH": pa.array(box_cols[3], type=pa.float32()),
        "boxScore": pa.array(box_cols[4], type=pa.float32()),
        "people": pa.array(people, type=pa.int8()),
        "identityUnresolved": pa.array([row[7] for row in bank], type=pa.bool_()),
        "dropped": pa.array([row[8] for row in bank], type=pa.bool_()),
        "error": pa.array([row[9] for row in bank], type=pa.string()),
        "x": lists(xy_all[:, :, 0]),
        "y": lists(xy_all[:, :, 1]),
        "score": lists(sc_all),
    })
    if metadata:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}),
             **{key.encode(): value.encode() for key, value in metadata.items()}})
    pq.write_table(table, str(path), compression="zstd",
                   use_byte_stream_split=["x", "y", "score"],
                   use_dictionary=["view", "error"], write_statistics=True)


# -- the uploader -------------------------------------------------------------------


class Uploader(threading.Thread):
    """Closed parts to `gs://bucket/prefix/`, one thread, in order, each
    object created once (`ifGenerationMatch=0`): a part that already
    exists is left as it is. The client is Google's, on the workload
    identity the entrypoint federated (`GOOGLE_APPLICATION_CREDENTIALS`),
    made on the first upload so a session that never closes a part never
    touches the library."""

    def __init__(self, bucket: str, prefix: str, telemetry=None,
                 client_factory=None, log=print, attempts: int = UPLOAD_ATTEMPTS,
                 backoff_s: float = UPLOAD_BACKOFF_S, sleep=time.sleep):
        super().__init__(daemon=True, name="record-uploader")
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.telemetry = telemetry
        self.client_factory = client_factory
        self.log = log
        self.attempts = attempts
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.queue: queue.Queue = queue.Queue()
        self.uploaded: list[str] = []
        self.existed: list[str] = []
        self.failed: list[str] = []
        self.bytes = 0
        self._bucket = None
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def put(self, name: str, path: Path, content_type: str) -> None:
        self._idle.clear()
        self.queue.put((name, path, content_type))

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
        """Drain what is queued, bounded, then stop. True when everything
        queued left the machine or failed for good."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self.queue.empty() or not self._idle.is_set():
            if time.monotonic() >= deadline or not self.is_alive():
                break
            self.sleep(0.05)
        drained = self.queue.empty() and self._idle.is_set()
        self._stop.set()
        return drained

    def snapshot(self) -> dict:
        return {"bucket": self.bucket_name, "prefix": self.prefix,
                "uploaded": len(self.uploaded), "existed": len(self.existed),
                "failed": list(self.failed), "bytes": self.bytes,
                "queued": self.queue.qsize()}

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                self._idle.set()
                continue
            try:
                self._upload(*item)
            finally:
                if self.queue.empty():
                    self._idle.set()

    def _gcs_bucket(self):
        if self._bucket is None:
            factory = self.client_factory
            if factory is None:
                from google.cloud import storage  # noqa: PLC0415

                def factory():
                    return storage.Client(
                        project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
            self._bucket = factory().bucket(self.bucket_name)
        return self._bucket

    def _upload(self, name: str, path: Path, content_type: str) -> None:
        object_name = f"{self.prefix}/{name}" if self.prefix else name
        for attempt in range(1, self.attempts + 1):
            try:
                blob = self._gcs_bucket().blob(object_name)
                blob.upload_from_filename(str(path), content_type=content_type,
                                          if_generation_match=0)
                self.bytes += path.stat().st_size
                self.uploaded.append(name)
                if self.telemetry:
                    self.telemetry.count("recordPartsUploaded")
                path.unlink(missing_ok=True)
                return
            except Exception as error:  # noqa: BLE001 - retried, then reported
                if _already_exists(error):
                    self.existed.append(name)
                    if self.telemetry:
                        self.telemetry.count("recordPartsExisted")
                    path.unlink(missing_ok=True)
                    return
                if attempt >= self.attempts or self._stop.is_set():
                    self.failed.append(name)
                    if self.telemetry:
                        self.telemetry.count("recordPartsFailed")
                    self.log(f"record: upload of {object_name} failed for good: "
                             f"{error!r}", flush=True)
                    return
                self._bucket = None  # a fresh client for the next try
                self.sleep(min(self.backoff_s * (2 ** (attempt - 1)), 30.0))


def _already_exists(error: Exception) -> bool:
    """A 412 on `ifGenerationMatch=0`: the object is already there."""
    code = getattr(error, "code", None)
    if code == 412:
        return True
    name = type(error).__name__
    return name == "PreconditionFailed" or "412" in str(error)[:40]


# -- the record ---------------------------------------------------------------------


class Record:
    """A session's streams under its lease's prefix, with their uploader.

    `hello(...)` writes hello.json once the analysis link has answered;
    the stream methods take rows; `close(summary)` closes every part,
    writes summary.json and waits, bounded, for the uploads. `flush()` is
    close() for a SIGTERM: whatever is open leaves now.
    """

    def __init__(self, directory: Path, bucket: str, prefix: str,
                 session_id: str, *, part_s: int = DEFAULT_PART_S,
                 telemetry=None, clock=time.time, client_factory=None,
                 log=print, provenance: dict | None = None):
        self.directory = directory
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.session_id = session_id
        self.part_s = max(1, int(part_s))
        self.telemetry = telemetry
        self.clock = clock
        self.log = log
        self.provenance = dict(provenance or {})
        self.started_wall_s = round(clock(), 3)
        directory.mkdir(parents=True, exist_ok=True)
        self.uploader = Uploader(bucket, self.prefix, telemetry,
                                 client_factory=client_factory, log=log)
        self.streams: dict[str, PartStream] = {}
        for name in JSONL_STREAMS:
            self.streams[name] = PartStream(
                name, directory, self.part_s, clock=clock,
                on_part=self._on_part, on_error=self._say)
        metadata = {"sessionId": session_id,
                    **{f"producer.{k}": v for k, v in self.provenance.items()
                       if isinstance(v, (str, int, float))}}
        self.keypoints: dict[str, KeypointStream] = {}
        for view, name in KEYPOINT_STREAMS.items():
            stream = KeypointStream(name, view, directory, self.part_s,
                                    metadata=metadata,
                                    clock=clock, on_part=self._on_part,
                                    on_error=self._say)
            self.streams[name] = stream
            self.keypoints[view] = stream
        self.files: list[str] = []
        self.closed = False
        self._closed_result: dict = {}
        self._closing = threading.Lock()
        self._telemetry_source = None
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True,
                                        name="record-ticker")
        self._stop = threading.Event()
        self.uploader.start()
        self._ticker.start()

    # -- streams -----------------------------------------------------------------

    def append(self, stream: str, row: dict) -> bool:
        target = self.streams.get(stream)
        if target is None or isinstance(target, KeypointStream):
            return False
        return target.append(row)

    def step(self, view: str, frame_index: int, at_s: float, keypoints, scores,
             box, box_score, people, unresolved: bool, frame_size=None,
             **flags) -> bool:
        stream = self.keypoints.get(view)
        if stream is None:
            return False
        return stream.append_step(frame_index, at_s, keypoints, scores, box,
                                  box_score, people, unresolved, frame_size,
                                  **flags)

    def keypoint_listener(self, view: str, state=None):
        """A `share_full_result` listener for `view`, reading the frame size
        off the view's state (live_pose.ViewState.frame_size) when given."""
        def listen(frame_index, at_s, keypoints, scores, box, box_score,
                   people, unresolved) -> None:
            self.step(view, frame_index, at_s, keypoints, scores, box, box_score,
                      people, unresolved,
                      frame_size=getattr(state, "frame_size", None))
        return listen

    def gap(self, view: str, row: dict) -> bool:
        """A pose worker's dropped or errored step (`Capture.pose` rows with
        those flags): a keypoint-less row of the view's stream."""
        if not (row.get("dropped") or row.get("error")):
            return False
        error = row.get("error")
        return self.step(view, int(row.get("frame", -1)), float(row.get("atS", 0.0)),
                         None, None, None, None, None, False,
                         dropped=bool(row.get("dropped")),
                         error=(str(error) if error and error is not True else
                                ("error" if error else None)),
                         wall_s=row.get("wallS"))

    def outbound(self, message: dict) -> None:
        """A message on its way to the analysis process: the `frame`, `audio`
        and `segment` kinds are the record's frames/, audio/ and segments/."""
        kind = message.get("kind")
        stream = {"frame": "frames", "audio": "audio", "segment": "segments"}.get(kind)
        if stream is None:
            return
        row = {key: value for key, value in message.items() if key != "kind"}
        self.append(stream, row)

    def telemetry_from(self, source) -> None:
        """Snapshot `source.snapshot()` into telemetry/ once a second."""
        self._telemetry_source = source

    # -- files -------------------------------------------------------------------

    def hello(self, **fields) -> None:
        """hello.json: what this session was, written once."""
        body = {
            "sessionId": self.session_id,
            "bucket": self.bucket, "prefix": self.prefix,
            "partSeconds": self.part_s,
            "startedWallS": self.started_wall_s,
            "producer": self.provenance,
            "keypoints": keypoint_layout_or_none(),
            "streams": {
                **{name: {"format": "jsonl.gz"} for name in JSONL_STREAMS},
                **{name: {"format": "parquet", "view": view}
                   for view, name in KEYPOINT_STREAMS.items()},
            },
            **fields,
        }
        self._file("hello.json", body)

    def _file(self, name: str, body: dict) -> None:
        path = self.directory / name
        try:
            path.write_text(json.dumps(body, indent=2, default=str) + "\n")
        except Exception as error:  # noqa: BLE001 - said, survived
            self._say(f"record: {name} failed: {error!r}")
            return
        self.files.append(name)
        self.uploader.put(name, path, "application/json")

    # -- lifecycle ---------------------------------------------------------------

    def _on_part(self, stream: PartStream, path: Path, window: int) -> None:
        self.uploader.put(f"{stream.name}/{path.name}", path, stream.content_type)

    def _say(self, text: str) -> None:
        if self.telemetry:
            self.telemetry.count("recordErrors")
        self.log(text, flush=True)

    def _tick_loop(self) -> None:
        last_telemetry = 0.0
        while not self._stop.wait(0.5):
            now = self.clock()
            for stream in list(self.streams.values()):
                try:
                    stream.tick(now)
                except Exception as error:  # noqa: BLE001 - said, survived
                    self._say(f"record: {stream.name} tick failed: {error!r}")
            source = self._telemetry_source
            if source is not None and now - last_telemetry >= 1.0:
                last_telemetry = now
                try:
                    snapshot = source.snapshot()
                    self.append("telemetry", {
                        "counters": snapshot.get("counters"),
                        "gauges": snapshot.get("gauges"),
                        "stages": snapshot.get("stages"),
                    })
                except Exception as error:  # noqa: BLE001 - said, survived
                    self._say(f"record: telemetry snapshot failed: {error!r}")

    def snapshot(self) -> dict:
        return {
            "bucket": self.bucket, "prefix": self.prefix,
            "partSeconds": self.part_s,
            "streams": {name: stream.snapshot()
                        for name, stream in self.streams.items()},
            "files": list(self.files),
            "upload": self.uploader.snapshot(),
        }

    def close(self, summary: dict | None = None,
              timeout_s: float = CLOSE_TIMEOUT_S) -> dict:
        """Every open part closed and queued, summary.json written, the
        uploader drained (bounded). Idempotent; the second call returns
        the first's result."""
        with self._closing:
            if self.closed:
                return self._closed_result
            self.closed = True
            self._stop.set()
            for stream in self.streams.values():
                stream.close()
            if summary is not None:
                self._file("summary.json", {
                    **summary,
                    "record": {**self.snapshot(), "endedWallS": round(self.clock(), 3)},
                })
            drained = self.uploader.close(timeout_s)
            result = self.snapshot()
            result["drained"] = drained
            if self.telemetry:
                self.telemetry.gauge("recordPartsQueued", float(self.uploader.queue.qsize()))
            self._closed_result = result
            return result

    def flush(self, timeout_s: float = 10.0) -> dict:
        """The SIGTERM path: close now with what there is, no summary
        beyond the record's own counts."""
        return self.close({"ended": "flush"}, timeout_s=timeout_s)


def keypoint_layout_or_none() -> dict | None:
    try:
        return keypoint_layout()
    except Exception:  # noqa: BLE001 - a courtesy
        return None
