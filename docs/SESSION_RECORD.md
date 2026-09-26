# The session record: what the enclave writes, and its layout

A session run for a signed-in account is written down as it happens
(README, "Session records"). The trainer names, in the lease it grants a
slot, the prefix the enclave's half goes under; the slot writes there and
nowhere else, as its attested identity, into the bucket its attested
environment names. This page is the layout of that half, for whoever reads
it back: the paths, the part grid, each stream's rows, and what is not in
it. The code is `workload/producer/record.py`.

## Paths

```
gs://<TEE_CAPTURE_BUCKET>/<account uuid>/estim_sessions/<session uuid>/enclave/
    hello.json
    summary.json
    runs/20260915T221554Z/hello.json
    runs/20260915T221554Z/summary.json
    runs/20260915T221618Z/…
    poses/part-20260915T051230Z.parquet
    faces/part-20260915T051230Z.parquet
    frames/part-20260915T051230Z.jsonl.gz
    audio/…      segments/…    vocal/…
    onsets/…     events/…      payloads/…    posts/…
    telemetry/…
```

The two uuids are opaque identifiers the trainer minted; the trainer's own
half of the same session is written beside `enclave/` (its streams,
manifest and summary), on the same grid with the same part names. The
lease's `record.prefix` must match
`^{uuid}/estim_sessions/{uuid}/enclave$` (lowercase hex) or the lease is
refused; a slot without `TEE_CAPTURE_BUCKET` refuses any lease naming a
record.

The record is the lease's, not a production run's. A session usually has
more than one run: the trainer opens a new `/produce` when the camera
changes (the phone's own picture first, then a fixed camera once its
connector is up, then again with the phone as the face inset), and each
run is a fresh pipeline on the slot. All of them write the same streams,
so a window has one part whichever runs fell in it, `hello.json` and
`summary.json` are written once, and each run has its own pair under
`runs/<start>/`, named by the UTC second it began (`-2`, `-3` should two
begin in the same second). The record closes when the lease ends: the
trainer's `/stop` or `/teardown`, the slot's idle exit or a lease expiry,
a SIGTERM, or a lease for another session.

## The grid

A part holds the rows written during one window of the wall clock:
`partSeconds` (30 by default, from the lease) long, aligned to multiples
of that length since the Unix epoch, named by the UTC second the window
starts, `part-YYYYMMDDTHHMMSSZ`. A row belongs to the window of the
moment it was appended, stamped as `wallS` (Unix seconds, three decimals);
a row that arrives after its window's part was closed rides in the open
part with its own `wallS` telling the truth, counted as `late` in the
summary. An empty window writes nothing. Parts are closed when the wall
clock leaves their window (a ticker closes a sparse stream's part within a
second of the window's end), uploaded as they close, and never
overwritten: each object is created with `ifGenerationMatch=0`, and an
object already there is left as it is (`existed` in the summary).

## `poses/` and `faces/` (Parquet)

One row per pose step of the view: every frame the pose cadence picked
(9 fps by default, `poseFps` in `hello.json`), whether or not a person was
found, plus a row for each step the worker dropped (queue full) or that
errored. Column order as written; `zstd` compression, `byte_stream_split`
encoding on the float columns, one row group per part.

| column | type | meaning |
| --- | --- | --- |
| `frame` | int32 | the frame index on the stream's 30 fps grid |
| `atS` | float64 | the frame's time on the stream clock (seconds; `frame / 30`) |
| `wallS` | float64 | Unix time the row was written |
| `view` | string | `body` or `face` (also the stream's name) |
| `frameW`, `frameH` | int16, nullable | the decoded frame's size in pixels; null when the step never saw a frame (a drop) |
| `boxX`, `boxY`, `boxW`, `boxH` | float32, nullable | the tracked person's box, pixels, origin top-left; null when nobody was found |
| `boxScore` | float32, nullable | the detector's score for that box |
| `people` | int8, nullable | how many person candidates the detector saw; null for a drop or error |
| `identityUnresolved` | bool | the tracker could not tell which candidate was the session's person |
| `dropped` | bool | the pose queue refused the frame; no inference ran |
| `error` | string, nullable | why the step produced no keypoints when it should have (`error`, or `keypoints:(K, 2)` for a result of another layout) |
| `x`, `y`, `score` | list<float32>, nullable | 308 values each, in model index order: the keypoint's pixel coordinates in the frame and the model's confidence. Null (not empty) when the step had no result: nobody found, dropped, or errored |

The keypoint layout is in `hello.json` (`keypoints`) and in each file's
metadata (`keypoints`, JSON), with `view`, `sessionId` and the image's
release stamp (`producer.imageVersion`, `producer.imageCommit`,
`producer.slot`):

| indices | block |
| --- | --- |
| 0–20 | body: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_big_toe`, `left_small_toe`, `left_heel`, `right_big_toe`, `right_small_toe`, `right_heel` (names in `keypoints.body`) |
| 21–41, 42–62 | left hand, right hand: 21 points each on the standard root-plus-four-joints-per-finger topology; the internal order is a hypothesis (`handOrderVerified: false`) until checked against stored tracks |
| 63–307 | face: 245 dense landmarks in the model's own order (`LABEL_63` .. `LABEL_307`) |

A score below `minKeypointScore` (0.3) means the model placed a point it
could not see; the coordinates are still written. Coordinates are in the
pixels of the view's decoded frame (`frameW` x `frameH`), which for the
face view is the phone's picture, mirrored or not as the phone sent it.

## The gzipped JSONL streams

One JSON object per line, `wallS` first. The rows are the messages of
`analysis/protocol.md` without their `kind`, kept whole.

| stream | one row per | fields |
| --- | --- | --- |
| `frames/` | decoded frame with a final pose decision (30 fps) | `frame`, `atS`, `keypoints` (the 21 body points by name, `[x, y, score]`, interpolated across the pose cadence; null where the person was absent), `fast`, `slow` (the regional motion descriptors, protocol.md `frame`) |
| `audio/` | half second of the audio track | `atS`, `streamS` and the measurements of protocol.md `audio`: the classifier's scores per label over the trailing window, `pitch` (pitch in Hz with its confidence, voiced share, loudness in dBFS), `frames` (a level and a pitch per 16 ms frame of the hop) |
| `segments/` | span the analysis asked to have typed | `id`, `fromS`, `toS`, the span's scores, level, pitch and duration, or `error` (`expired`, `span`, `empty`, `closed`, `no-audio`) |
| `vocal/` | judgement of the analysis's vocal side | `kind` (`activation`, `decision`, `segment_error`, `baseline`), `atS`, and the numbers it judged on (protocol.md `vocal`) |
| `onsets/` | onset the analysis found | `atS` |
| `events/` | paired event | `fromS`, `toS` |
| `payloads/` | analysis payload | the payload's own fields |
| `posts/` | reading posted to the trainer | the reading's own fields (`atS`, `modelVersion`, …) |
| `telemetry/` | second | `counters`, `gauges`, `stages` of the producer's telemetry snapshot |

## `hello.json`

Written when the record opens, with the first run:

| field | content |
| --- | --- |
| `sessionId`, `bucket`, `prefix`, `partSeconds`, `startedWallS` | the lease's terms and when the record opened |
| `producer` | `imageVersion`, `imageCommit` (the release stamp of the attested image), `slot` |
| `keypoints` | the layout above |
| `streams` | each stream's format |
| `runs` | where the runs' files are |

## `runs/<start>/hello.json`

One per production run, written once its analysis process has answered
`hello`:

| field | content |
| --- | --- |
| `sessionId`, `runId`, `run`, `startedWallS` | the record's session, the run's id (its start, UTC), the trainer's name for the run, and its start |
| `producer` | the image's release stamp and the slot, as in `hello.json` |
| `hello` | what the producer told the analysis: `fps`, `poseFps`, `facePoseFps`, `views`, `audio`, `audioModel`, `postIntervalS` |
| `ready` | the analysis's answer: `protocol`, `version` (the bundle in `analysis.lock`), `modelVersion`, `vocal` (the constants its `vocal` rows are judged against) |
| `sources` | which picture each view is (`poses`, `faces`, `audio`): the view's path on the enclave's own loopback relay (`rtsp://127.0.0.1:8554/…`, so a reader can tell the phone's camera from an external one) and the pose cadence; never a camera's address or link |

A run's rows are told apart in the streams by time: `startedWallS` and
`endedWallS` of each run are in both its summary and the record's.

## `runs/<start>/summary.json`

The run's summary as the producer prints it (`bootMs`, `counters`,
`gauges`, `stagesMs`, the analysis's summary, `overlay`) with `runId`,
`startedWallS`, `endedWallS`, and `record`: the record's per-stream
`rows`, `parts`, `late`, `dropped`, `errors` and its `upload` counts as
they stood when the run ended (`closed: false`: the parts of the open
window are still being written). A run that failed to start has `error`
and no counters.

## `summary.json`

Written when the lease ends, with `ended` saying how (`stop`, `teardown`,
`exit` for the slot's own idle exit or a drained teardown, `lease` for
another session's lease, `flush` for a SIGTERM, `run` when the record was
a single run's, outside a serving slot) and `record`: per-stream `rows`,
`parts`, `late`, `dropped`, `errors`; `files`; `runs` (`id`, `run`,
`startedWallS`, `endedWallS` each); `upload` (`uploaded`, `existed`,
`failed` by name, `bytes`) as they stood when the summary was written,
before the last parts left; `startedWallS`, `endedWallS`. Whether every
queued part then left before the close timed out is `drained` in the
slot's log line for the close, and the slot's `recordPartsQueued` gauge.

Records written before this layout (2026-09-15, image `v0.8.0`) have the
first run's `hello` and summary in `hello.json` and `summary.json`
themselves, no `runs/`, and where a later run of the same session fell in
the window the first ended in, that window's part holds the first run's
rows only.

## What is not in it

No frame, crop, or audio sample, at any resolution; no name, address,
account identifier other than the opaque uuids in the path, phone
identifier, or network address; no transcript (the conversation is the
trainer's half, from the voice service, never the enclave's, which runs no
speech recognition); no camera credential (an external camera's link
never leaves the process that dials it). The keypoints are coordinates,
the descriptors statistics of motion, the audio rows scores and a level
and pitch track: numbers about the sound, not the sound.
