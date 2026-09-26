# The analysis socket

The enclave runs two Python processes that matter for what happens to the
user's media:

- the **producer** (`workload/producer/producer.py`): decodes the stream,
  runs person detection and keypoint detection, computes regional motion
  descriptors, draws the annotated view. Every line that reads a frame or an
  audio sample is in this repository.
- the **analysis** process: receives keypoints and descriptors from the
  producer over a local Unix socket and turns them into the readings the
  trainer consumes. Its code is not published (it is the operator's
  interpretation of those numbers), but its bytes are pinned: the enclave
  fetches the bundle named in [`analysis.lock`](../analysis.lock) at boot and
  refuses to start it unless its SHA-256 matches. The lock file is part of
  the attested image, so the attestation digest covers which analysis bundle
  ran.

The analysis process runs under its own OS user (`analysis`), starts with
nothing but the socket, has no network access of its own (everything it
emits leaves through the producer, listed below), and never receives a
frame, a crop, or an audio sample. This document is the complete list of
what crosses the socket in both directions. Anything not listed here does
not cross it.

## Transport

- One Unix stream socket, path passed to the producer as
  `--analysis-socket` (the enclave uses `/run/tee/analysis/analysis.sock`).
- The analysis process listens; the producer connects once per session and
  closes when the session ends.
- Messages are JSON objects, one per line (`\n`-terminated, UTF-8). Floats
  are carried at full precision; `NaN` may appear (Python's `json` accepts
  it) where a statistic is undefined, for example a window with no rows.
- Every message has a `kind` field. Unknown kinds are ignored by both sides
  so the protocol can grow without a flag day; the `protocol` integer in
  `hello`/`ready` names the version of this document.

## Producer to analysis

### `hello`, once, first

```json
{"kind": "hello", "protocol": 1, "fps": 30.0, "poseFps": 6.0,
 "postIntervalS": 1.0, "run": null}
```

- `fps`: the decode cadence the `frame` messages arrive at.
- `poseFps`: the cadence the keypoint model runs at.
- `postIntervalS`: how often the analysis is expected to emit a `post`.
- `run`: the operator's name for a captured test session, or `null` in
  production.

### `pose`, every keypoint-model result, in order

```json
{"kind": "pose", "frame": 150, "atS": 5.0,
 "keypoints": {"left_hip": [812.4, 604.1, 0.93], "...": "..."},
 "dropped": false, "error": false, "frameSize": [1280, 720]}
```

- `frame`: decode frame index; `atS`: stream time in seconds.
- `keypoints`: name to `[x, y, score]` in source pixels, or `null` when no
  person was detected, when the pose slot was dropped under load
  (`dropped: true`), or when the model failed on that frame
  (`error: true`). The keypoint names are the model's
  (`workload/pixel/keypoints.py`).
- `frameSize`: the decoded frame's `[width, height]`, so framing can be
  judged against the picture edges. Nothing else about the picture is sent.

### `frame`, every decoded frame that has a final pose decision, in order

```json
{"kind": "frame", "frame": 151, "atS": 5.0333,
 "keypoints": {"...": "..."},
 "fast": {"...descriptor..."}, "slow": null}
```

- `keypoints`: the pose row at frame cadence: the model's row when the frame
  is on the pose grid, a linear interpolation between the two neighbouring
  rows when it is bridged, `null` when it is not
  (`workload/pixel/motion.py`, `RowAssembler`).
- `fast`: the descriptor for the pair (previous frame, this frame) at 30 fps,
  `null` when the two frames were not consecutive or either had no usable
  pelvis frame.
- `slow`: the same descriptor for the pair five frames apart, present only on
  every fifth frame.

A descriptor (`workload/pixel/motion.py`, `MotionSample.as_json`):

| field | meaning |
| --- | --- |
| `atS` | stream time of the pair's later frame |
| `profileRange` | per candidate axial window (key such as `m035` for centre -0.35 hip widths, `p025` for +0.25): `[left, right]` peak-to-trough range of the mean axial brightness profile of the hip-region strip, by half |
| `control` | the same statistic over windows that cannot carry the signal of interest: `back...` keys for two windows on the lower back, `bed...` keys for the candidate windows measured on the control strip, the identical warp slid sideways onto the support surface beside the body |
| `axialStrain` | `[left, right]` mean axial gradient of the axial optical flow over the hip-region rows, or `null` |
| `rigid` | least-squares similarity fit of the same flow: `translation` `[lateral, axial]`, `omega`, `scale`, and `deform` `[left, right]` (RMS residual per half), or `null` |
| `controlRigid` | the same fit on the control strip's flow, or `null` |
| `halfFlow` | `[[lateral, axial], [lateral, axial]]` mean flow per half, or `null` |
| `regionAxialFlow` | mean axial optical flow over the square hip-region patch |

Units: hip widths for positions, canonical pixels per frame for flow, grey
levels for brightness. All of these are statistics over a warped region of
the frame; none of them can be inverted into an image.

### `stop`, once, last

```json
{"kind": "stop", "atS": 187.4}
```

The producer then waits (bounded) for `summary` and closes the socket.

## Analysis to producer

### `ready`, once, in reply to `hello`

```json
{"kind": "ready", "protocol": 1, "version": "2026.09.09-1",
 "modelVersion": "analysis/v1"}
```

`version` is the bundle version from `analysis.lock`; `modelVersion` is what
the readings carry as their `modelVersion` field.

### `post`, at the post cadence

```json
{"kind": "post", "body": {"atS": 12.0, "modelVersion": "analysis/v1", "...": "..."}}
```

The reading. The producer treats `body` as opaque JSON: it appends it to the
session capture (`posts.jsonl`), emits it on the session's event stream, and
POSTs it to the trainer URL it was started with. This is the only path by
which anything derived from the user's media leaves the enclave, and it
carries numbers only.

### `onset`, `event`, `payload`

```json
{"kind": "onset", "atS": 12.34}
{"kind": "event", "fromS": 12.34, "toS": 13.1}
{"kind": "payload", "payload": {"...": "..."}}
```

Finer-grained records the analysis keeps alongside the readings; the
producer appends each to the capture (`onsets.jsonl`, `events.jsonl`,
`payloads.jsonl`) and emits `onset` and `payload` on the session's event
stream. Like `post`, they are numbers about the session, never media.

### `hud`

```json
{"kind": "hud", "lines": ["...", "..."]}
```

Text lines for the annotated view. The overlay draws whatever it is given
under its own status lines; the producer does not interpret them. The
annotated view is returned only to the user whose camera it came from.

### `gauges`

```json
{"kind": "gauges", "values": {"backlog": 0, "onsetsEmitted": 3}}
```

Numbers for the producer's telemetry line and `/status` snapshot.

### `log`

```json
{"kind": "log", "text": "calibrated at 41.2s"}
```

Printed by the producer to its own log.

### `summary`, once, in reply to `stop`

```json
{"kind": "summary", "summary": {"analysis": {"...": "..."}}}
```

Merged into the session summary the producer writes to the capture and
prints at exit. Keys the producer owns (`bootMs`, `counters`, `gauges`,
`stagesMs`, `run`, `captureDir`, `overlay`) are never overwritten.

## What this means for a user

Frames exist in the producer process and nowhere else. Between the
producer and the analysis process travel keypoints (named points in pixel
coordinates), the descriptors above (per-window statistics of brightness
and optical flow in a body-carried frame), and the analysis's own numbers
coming back. The analysis bundle can be shown to be the one the lock names
(its SHA-256 is checked before it starts, and the lock is inside the
attested image) even though its source is not published.
