# The analysis socket

The enclave runs two Python processes that matter for what happens to the
user's media:

- the **producer** (`workload/producer/producer.py`): decodes the stream,
  runs person detection and keypoint detection, computes regional motion
  descriptors, draws the annotated view, and, when the stream has an audio
  track, classifies it into non-speech vocalization labels with level and
  pitch (`workload/audio/`). Every line that reads a frame or an audio
  sample is in this repository.
- the **analysis** process: receives keypoints, descriptors and audio
  measurements from the producer over a local Unix socket and turns them
  into the readings the trainer consumes. Its code is not published (it is
  the operator's interpretation of those numbers), but its bytes are pinned:
  the enclave fetches the bundle named in [`analysis.lock`](../analysis.lock)
  at boot and refuses to start it unless its SHA-256 matches. The lock file
  is part of the attested image, so the attestation digest covers which
  analysis bundle ran.

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
 "postIntervalS": 1.0, "run": null,
 "audio": true, "audioModel": "ced.cpp-abi1:ced-small-f16.gguf"}
```

- `fps`: the decode cadence the `frame` messages arrive at.
- `poseFps`: the cadence the keypoint model runs at.
- `postIntervalS`: how often the analysis is expected to emit a `post`.
- `run`: the operator's name for a captured test session, or `null` in
  production.
- `audio`: whether the producer runs the audio stage for this session
  (`audio` messages may follow and `classify` requests are answered).
  `false` means the stream's sound is not read at all. `audioModel` names
  the classifier build and weights (`null` without the stage).
- `views` (absent in older producers, then `["body"]`): the camera views
  the session reads. `["body"]` is one camera. `["body", "face"]` is a
  fixed camera behind the user as the body view - everything below that
  is not marked otherwise - and the user's phone, pointed at their face,
  as a second view whose keypoints arrive as `facePose`; the audio stage
  then reads the phone's track. `facePoseFps` is the face view's keypoint
  cadence (`null` with one view).

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

### `facePose`, every keypoint-model result on the face view, in order

Only in a session whose `hello` listed a `face` view.

```json
{"kind": "facePose", "frame": 45, "atS": 1.5, "bodyAtS": 3.5,
 "keypoints": {"nose": [318.2, 411.7, 0.97], "...": "..."},
 "dropped": false, "error": false, "frameSize": [720, 1280]}
```

- `frame`, `atS`: the face view's own decode frame index and stream time;
  its timeline is not the body view's.
- `bodyAtS`: the same moment on the body view's timeline, from the two
  streams' sender clocks (`workload/producer/sync.py`), so a face row can
  be set beside the `pose` and `frame` rows around it; `null` before the
  two views are lined up.
- `keypoints`, `dropped`, `error`, `frameSize`: as in `pose`, for the face
  view's frame. The model is the same and so are the keypoint names; the
  body points of a face view are whatever of the body the phone sees.

Face rows carry no descriptors: the regional motion descriptors are the
body view's alone.

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

### `audio`, every half second of the audio track

```json
{"kind": "audio", "atS": 12.5, "streamS": 12.63, "hopS": 0.5, "windowS": 2.0,
 "scores": {"Wail, moan": 0.0021, "Groan": 0.0007, "Breathing": 0.31, "...": "..."},
 "pitch": {"pitchHz": 142.3, "pitchConfidence": 0.61, "voicedFramePct": 38.2,
           "loudnessDbfs": -31.4},
 "frames": [[12.032, -47.2, null], [12.048, -41.9, 139.8], "..."]}
```

Sent only when the stream has an audio track (`workload/audio/audio_stage.py`).

- `atS`: end of this hop on the audio clock (seconds of audio decoded since
  the session's audio began); `streamS`: the producer's frame clock at the
  moment the message was built, so the two can be lined up. `hopS` is the
  hop's length, `windowS` the length of the trailing window the scores and
  pitch summary were computed over (2 s once that much has been heard).
- `scores`: the CED classifier's score per label, for the labels in
  `workload/audio/ced.py` (`TARGET_LABELS`: the AudioSet classes Screaming;
  Crying, sobbing; Whimper; Wail, moan; Sigh; Groan; Grunt; Breathing; Gasp;
  Pant; and Speech, the last so that a voice in the room can be told apart
  from the others). Each is a probability-like number in [0, 1]. No other
  class of the model's 527 is sent.
- `pitch`: over the same window, the median fundamental frequency in Hz of
  the frames that had one (`null` if none did), the median periodicity
  confidence of those frames, the percentage of frames that had a pitch,
  and the RMS level in dBFS.
- `frames`: the contour of the new samples, one entry per 16 ms frame
  (64 ms window): `[time on the audio clock, level in dBFS, pitch in Hz or
  null]`. Every frame is sent exactly once. A frame is about 40 bytes; a
  second of audio becomes about 60 frames.

Nothing in an `audio` message can be turned back into sound: a level and a
pitch per 16 ms is not a waveform, and the labels are class scores.

### `segment`, in reply to `classify`

```json
{"kind": "segment", "id": 17, "fromS": 11.62, "toS": 12.31,
 "ced": {"topLabel": "Groan", "topScore": 0.412, "scores": {"...": "..."}},
 "pitch": {"medianHz": 118.4, "p10Hz": 109.0, "p90Hz": 131.2,
           "slopeHzPerS": -14.0, "voicedFraction": 0.72, "voicedFrames": 31,
           "pitchReliable": true, "confidence": 0.7, "frames": 43},
 "loudness": {"peakDbfs": -18.2, "meanDbfs": -26.9, "rmsDbfs": -25.1},
 "spectral": {"centroidHz": 812.5, "rolloff85Hz": 1890.6}}
```

The same kind of measurements over one span the analysis asked about
(`workload/audio/audio_features.py`): the classifier's scores over the span
(centred in at least one second of surrounding audio, which the classifier
needs), an F0 summary, a loudness summary, and the spectral centroid and 85%
rolloff. The span is taken from the producer's 20 s rolling history of the
audio; a span it no longer holds, or a malformed one, is answered with
`{"kind": "segment", "id": 17, "error": "expired" | "span" | "empty" |
"busy" | "closed" | "no-audio"}` instead.

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

### `classify`

```json
{"kind": "classify", "id": 17, "fromS": 11.62, "toS": 12.31}
```

A request to measure one span of the audio (at most 10 s long, on the audio
clock), answered with a `segment`. This is how the analysis types a sound
it noticed in the frame contour: it names the span, the producer measures
it. The request carries two timestamps and an id; it cannot ask for
samples, and the reply never contains any.

### `summary`, once, in reply to `stop`

```json
{"kind": "summary", "summary": {"analysis": {"...": "..."}}}
```

Merged into the session summary the producer writes to the capture and
prints at exit. Keys the producer owns (`bootMs`, `counters`, `gauges`,
`stagesMs`, `run`, `captureDir`, `overlay`) are never overwritten.

## What this means for a user

Frames and audio samples exist in the producer process and nowhere else.
Between the producer and the analysis process travel keypoints (named
points in pixel coordinates), the descriptors above (per-window statistics
of brightness and optical flow in a body-carried frame), the audio
measurements above (class scores, a level and a pitch per frame, and the
same over spans the analysis asks about), and the analysis's own numbers
coming back. The analysis bundle can be shown to be the one the lock names
(its SHA-256 is checked before it starts, and the lock is inside the
attested image) even though its source is not published.
