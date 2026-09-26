# The analysis socket

The enclave runs two Python processes that matter for what happens to the
user's media:

- the **producer** (`workload/producer/producer.py`): decodes the stream,
  runs person detection and keypoint detection, computes regional motion
  descriptors, draws the view returned to the user, and, when the stream has an audio
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
  closes when the session ends. A connection that breaks mid-session (a
  send fails, the analysis's end closes) is not the session's end: the
  producer connects again and greets with the same `sessionId` and
  `resume: true`, and the analysis, which keeps a session whose connection
  ended without `stop` for a grace period (two minutes), carries it on
  where it was (`ready.resumed`). Nothing sent while the link was broken
  is queued; the producer counts it (`analysisDropped` in its telemetry)
  and the record's `log` stream says what happened.
- Messages are JSON objects, one per line (`\n`-terminated, UTF-8). Floats
  are carried at full precision; `NaN` may appear (Python's `json` accepts
  it) where a statistic is undefined, for example a window with no rows.
- Every message has a `kind` field. Unknown kinds are ignored by both sides
  so the protocol can grow without a flag day; the `protocol` integer in
  `hello`/`ready` names the version of this document.

## Producer to analysis

### `hello`, once, first

```json
{"kind": "hello", "protocol": 1, "fps": 30.0, "poseFps": 9.0,
 "postIntervalS": 1.0, "run": null, "sessionId": "4ce25466-...",
 "audio": true, "audioModel": "ced.cpp-abi1:ced-small-f16.gguf",
 "audioLabels": ["Speech", "Male speech, man speaking", "..."]}
```

- `fps`: the decode cadence the `frame` messages arrive at.
- `poseFps`: the cadence the keypoint model runs at on the body view: the
  `pose` rows per second. It need not divide `fps`; the model's frames are
  the decode grid's slots nearest each ideal instant (at 9 of 30 fps, the
  frames 0, 3, 7, 10, 13, 17, ...: a 3-4-3 pattern of frame gaps, nine
  per thirty frames), so consecutive `pose` rows are three or four frames
  apart, not a fixed stride. A source slower than the grid arrives with
  repeated frames, and a slot that lands on a repeat of the frame last
  posed is posed on the next frame that differs instead, so a row may
  carry the frame one or two after its slot (its `frame` and `atS` are
  that frame's); a source whose picture stops changing keeps the cadence
  with repeats after one slot's wait. Rows stay in frame order either way.
- `postIntervalS`: how often the analysis is expected to emit a `post`.
- `run`: the operator's name for a captured test session, or `null` in
  production.
- `sessionId` (absent in older producers): the trainer's session id the
  slot is leased to (the record's), or the run's own id without a lease;
  an opaque string the analysis uses only as the key a `resume` names.
- `resume` (only on a reconnect, `true`): take up the session parked
  under `sessionId` if it is still there; otherwise this hello starts a
  new one, and `ready.resumed` says which happened.
- `audio`: whether the producer runs the audio stage for this session
  (`audio` messages may follow and `classify` requests are answered).
  `false` means the stream's sound is not read at all. `audioModel` names
  the classifier build and weights (`null` without the stage).
  `audioLabels` (absent in older producers) is the classifier's whole
  label table, the 527 AudioSet class names in the model's own order: the
  order the `all` vectors in `audio` and `segment` messages follow.
- `views` (absent in older producers, then `["body"]`): the camera views
  the session reads. `["body"]` is one camera. `["body", "face"]` is a
  fixed camera behind the user as the body view - everything below that
  is not marked otherwise - and the user's phone, pointed at their face,
  as a second view whose keypoints arrive as `facePose`; the audio stage
  then reads the phone's track. `facePoseFps` is the face view's keypoint
  cadence (`null` with one view): the body's unless the producer was
  started with a different one for the face, and picked from the face
  view's own decode grid the same way.
- `faceFrameIntervalS` (absent in older producers, `null` when the
  producer has nowhere to send them; then none are sent): asks for `face`
  messages (below) after the face view's rows, at most this many seconds
  of stream time apart. The enclave passes `0.2`. Only meaningful with a
  `face` view; a bundle from before 2026.09.25-6 ignores it.

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
  body points of a face view are whatever of the body the phone sees. A
  face row names the face block as well: the 238 face landmarks
  (`workload/pixel/keypoints.py`, `FACE`, indices 70-307: midline,
  eyebrows, eyelids, nose, lips, ears, iris, pupil) beside the 21 body
  points, so the analysis can measure the expression; a body row names the
  body points only.

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
  every fifth frame. Its spacing is its own (0.167 s, the interval the
  slow statistics were defined over), not `poseFps`'s.

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
| `midlineValley` | `[width, area, depth, level]` of the dark valley the brightness profile has where the two halves meet, measured along a per-session axis: half-depth width and darkness-deficit area in hip widths, depth and on-axis level as fractions of the surface brightness beside it; `null` until the axis is fitted (the first 3 s of a stream) |
| `midlineAxis` | the fitted axis the valley is measured along: `offset` (hip widths off the strip midline), `slope` (columns per row), `depth` and `spread` (its median relative depth and the scatter of the per-row floors about the line, in hip widths, on the running mean strip: the analysis ignores the valley when the axis is not credible); `null` with `midlineValley` |
| `regionAxialFlow` | mean axial optical flow over the square hip-region patch |

Units: hip widths for positions, canonical pixels per frame for flow, grey
levels for brightness. All of these are statistics over a warped region of
the frame; none of them can be inverted into an image.

### `audio`, every half second of the audio track

```json
{"kind": "audio", "atS": 12.5, "streamS": 12.63, "hopS": 0.5, "windowS": 2.0,
 "scores": {"Wail, moan": 0.0021, "Groan": 0.0007, "Breathing": 0.31, "...": "..."},
 "all": [0.0142, 0.0003, "..."],
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
  from the others). Each is a probability-like number in [0, 1].
- `all` (absent in older producers): the same classifier's score for every
  one of its 527 classes over the same window, rounded to four decimals,
  in the order of `hello.audioLabels`. The room's sounds are in it (music,
  a fan, rain, a machine's hum, a television), which is how the analysis
  tells a person's sound over a background from the background itself.
  Class scores are all it is: no class is a word, a voice or a sample.
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
 "ced": {"topLabel": "Groan", "topScore": 0.412, "scores": {"...": "..."},
         "all": [0.0142, 0.0003, "..."]},
 "pitch": {"medianHz": 118.4, "p10Hz": 109.0, "p90Hz": 131.2,
           "slopeHzPerS": -14.0, "voicedFraction": 0.72, "voicedFrames": 31,
           "pitchReliable": true, "confidence": 0.7, "frames": 43},
 "loudness": {"peakDbfs": -18.2, "meanDbfs": -26.9, "rmsDbfs": -25.1},
 "spectral": {"centroidHz": 812.5, "rolloff85Hz": 1890.6,
              "flatness": 0.0731, "lowShare300": 0.2145}}
```

The same kind of measurements over one span the analysis asked about
(`workload/audio/audio_features.py`): the classifier's scores over the span
(centred in at least one second of surrounding audio, which the classifier
needs; `all` is the whole table as in `audio`, absent in older producers),
an F0 summary, a loudness summary, and four spectral shape statistics: the
centroid, the 85% rolloff, the spectral flatness of the power spectrum (0
for a pure tone, 1 for white noise) and the share of power under 300 Hz
(where a machine's hum sits; the last two absent in older producers). The
span is taken from the producer's 20 s rolling history of the
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
 "modelVersion": "analysis/v1", "resumed": false, "vocal": {"...": "..."}}
```

`version` is the bundle version from `analysis.lock`; `modelVersion` is what
the readings carry as their `modelVersion` field. `resumed` (absent in
older bundles, then `false`) is whether this connection took up a parked
session (`hello.resume`) rather than starting one. `vocal`, when present, is
the constants the `vocal` rows below are judged against (thresholds, the
labels' names, the bundle's vocal version): numbers and names, kept in the
session record's `hello.json` so a row can be read back years later. From
bundle 2026.09.25-1 it also carries `head`: the version, threshold, feature
count and type classes of the fixed classifier that judges each measured
span from the `segment` and the trailing `audio.all` tables (`null` when
the producer sent no `audioLabels`, and the classifier's top label decides
as before), and `contextGroups`, the label groups the trailing background
is read as. Bundle 2026.09.25-4 carries a newer such classifier, fitted on
a third more reviewed spans than the one before it, over the same feature
list; `head.version` tells them apart, the fields are the same.
`face`, when present (a bundle from 2026.09.18-1 on), is the same for what
the readings carry about a `face` view: the keypoint indices and thresholds
it is computed from, the names of its channels, and that part of the
bundle's version. Numbers and names, kept in `hello.json` the same way.
From 2026.09.22-1 (`face/v4`) the reading's `face` object also carries the
head's speed (`headSpeed`, `headSpeed5s`: the rigid landmarks' centroid's
step between rows, in inter-ocular distances a second, over the last
second of face rows and the last five seconds of readings), `regions` (per face region of the keypoint
definition, the mean displacement of its landmarks in the baseline's pose
from their baseline-minute positions, inter-ocular units, `dy` downward
and `dx` outward, over the last second of face rows), and a signed channel, `browOuterZ` (the outer brows'
distance from the outer canthi against the baseline, in robust units); the
`face` constants name the regions' landmarks and the speed's rule. The
fields from before are computed as before. The per-second aggregates are
read by the newest face row's clock, not the reading's: a face row's
`bodyAtS` leads the body clock the readings are posted on by several
seconds (2026.09.22-2; 2026.09.22-1 read them by the reading's clock and
posted `regions` as `null`).
`stance`, when present (a bundle from 2026.09.24-1 on), is the same for the
posture word the readings carry: the names (`prone`, `knee_chest`, `wariza`,
`seated`, `standing`, `supine`, `side`, `unknown`), the body unit, the vote
and hold windows and the thresholds each word is judged by, kept in
`hello.json` the same way. From that bundle the reading's `positioning`
object carries `stance`, `stanceConfidence` (the two-second vote's share
behind the word) and `stanceSinceS`, with `orientation` derived from the
stance for readers from before, and its `posture` object carries `stance`
and `hipRiseBu` (the hips' height above their rest line, in body units)
beside the fields from before, which are `null` while the stance is
standing, seated or wariza (no lying reference applies) and computed as
before otherwise. The word is judged from one camera's 21 body keypoints in
the body's own units and against each landmark's rest line, so the camera
may be anywhere around him. From 2026.09.25-3 (`stance/v3`) the word is
`unknown` rather than a guess while the hips' rest line is not yet
credible and the row's shape fits both a lying posture and a kneeling one
(the same keypoints from a camera behind him), and a hips' line that
would put the hips' rest above the shoulders' rest is not used until a
lying stretch corrects it; the `stance` constants name the thresholds
(`rest.hipsAboveShouldersMaxBu`, `unresolved`, `kneeChest.rearKneeFlexDeg`).
The fields are the same; only the word's timing at the start of a run and
its reading of a fold seen from below the hips change.
From 2026.09.25-5 (`stance/v4`) the rest lines learn only from rows with a
hip and a knee in frame (a body passing close to the lens with the legs
out of frame taught them where it was, not where it lies), and when the
body has been still for 30 s with the hips and shoulders both more than a
body unit off their lines, the lines are re-seeded from the still rows;
the `stance` constants name both (`rest.contactInFrame`, `rest.reanchor`).
The same bundle's reading carries `rhythmSnr` and `rateConfidence` beside
the contraction rate: the best channel's in-band to out-of-band power
ratio, and a confidence in the rate from it (0 while the channel shows no
rhythm at the contraction period, when no onsets are emitted either; 1 at
twice the gate), so a reader weighs the rate by it; both `null` before the
detector is ready. Its `posture` object carries `buckPeakedness`, the
share of the movement band's power in its strongest bin, a diagnostic
beside `buckPowerRatio`; the fields from before are computed as before.
From 2026.09.24-2 (`face/v5`) the reading's `face` object also carries
`hazard`: the analysis's probability that the response peaks within 30 and
within 60 seconds (`p30`, `p60`, their logits, and `featuresPresent`, the
share of the model's inputs the object had), read from the `face` object's
own fields alone (the indices and trend clocks, the channels, the regions,
the head's speed) by a fixed model whose identity the `face` constants name
(`hazard`: its version, horizons, column count and rule). It is `null`
while the face is not present or its baseline is not ready. The passage of
time is not in it: the analysis does not know how long a session has run,
and the reader adds that. The fields from before are computed as before.

### `post`, at the post cadence

```json
{"kind": "post", "body": {"atS": 12.0, "modelVersion": "analysis/v1", "...": "..."}}
```

The reading. The producer treats `body` as opaque JSON: it appends it to the
session capture (`posts.jsonl`, or the record's `posts/` parts), emits it on
the session's event stream, and POSTs it to the trainer URL it was started
with. It carries numbers only. Besides the readings, a leased session's
record (`workload/producer/record.py`, described in the README under
"Session records") is the other path out of the enclave: the keypoints,
descriptors, audio measurements and the analysis's rows, written to the
attested capture bucket under the prefix the trainer's lease named. Frames
and samples are on neither path.

### `face`, after a face row, at the frame cadence

Only when `hello` carried `faceFrameIntervalS`, from bundle 2026.09.25-6.

```json
{"kind": "face", "body": {"atS": 41.7, "present": true, "confidence": 0.97,
 "baselineReady": true, "channels": {"eyeBlinkL": 12.3, "jawOpen": 0.0, "...": "..."}}}
```

A frame for the user's own display: the channel values the reading's
`face` object carries (the same names, read from the same smoothed rows),
sent between readings so the display can follow the face at the face
rows' pace rather than once a second. `body.atS` is the newest face row's
clock; `present` whether a face is in view (from 2026.09.26-1, one placed
within a second of the newest row: a row dropped under load or one the
fit could not place is not the face gone; in 2026.09.25-6 it was whether
the very last row was placed); `confidence` the model's on that row;
`baselineReady` whether the first-minute baseline the channels are
measured against is in; and `channels` the twenty-seven channel values
the display reads (the contract's rig channels and five more), or `null`
before the baseline or while no face is present. From 2026.09.26-1 the
frame's unit channels are on the display's scale, 100 at a full movement
in inner-canthal units (the `face` constants' `wire.displayRange`), where
the reading's `channels` are a detector's, 100 at three robust units of
the baseline minute's own jitter; the reading's `face` object carries the
same twenty-seven on the display's scale as `wire` from that bundle, so a
reading agrees with the frames around it, and `present` follows the same
rule. Nothing else of the reading's `face` object is
in it: no aggregate, clock, region, witness or hazard, and, like every
message on this socket, no keypoint. The producer treats `body` as opaque
JSON and sends it to the trainer alone, at the readings' URL with the last
path segment `face` in place of `readings`: it is not appended to the
capture or the record (the record's `posts` stream has the `face` object
once a second) and not emitted on the session's event stream. A frame that
is still unsent when the next arrives is replaced by it, and a frame never
displaces a reading.

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

### `vocal`, one per judgement of the audio measurements

```json
{"kind": "vocal", "row": {"kind": "decision", "atS": 12.31, "...": "..."}}
```

The analysis's account of one step of its vocalization typing: `row.kind`
is `activation` (a classifier activation opened or closed a span),
`decision` (a span was judged, with the scores, level, pitch and duration
it was judged on and the verdict), `segment_error` (a `classify` it asked
for was not answered) or `baseline` (the reference level or pitch it
compares against changed). A `decision` from bundle 2026.09.25-1 on also
carries the head's score for the span (`headScore`), the head's version,
the type it named (`type`, beside the classifier's `topLabel`) and its
probability per type class (`typeProbs`); `rejected` joins the verdicts
(the head scored the span under its threshold and neither the type head
nor the classifier's top label called it breath; a span under the
threshold that is breath by either stays `breathing`). Everything in `row` is a
number, a label name or a timestamp, derived from the `audio` and
`segment` messages above; no sample is anywhere near it. The producer appends each row to the session
record's `vocal/` stream (`workload/producer/record.py`) and does not emit
it on the session's event stream or send it anywhere else.

### `hud`

```json
{"kind": "hud", "lines": ["...", "..."]}
```

Accepted and not drawn. The view returned to the user carries no
lettering any more (`workload/producer/overlay.py`: the picture, with the
keypoints when the user asks), so the producer takes this message and
drops it; it stays in the protocol so an analysis bundle that still sends
it needs no change, and will be removed once none does. Nothing the
analysis says reaches the picture.

### `gauges`

```json
{"kind": "gauges", "values": {"backlog": 0, "onsetsEmitted": 3}}
```

Numbers for the producer's telemetry line and `/status` snapshot.

### `log`

```json
{"kind": "log", "text": "calibrated at 41.2s"}
```

Printed by the producer to its own log and kept in the session record's
`log` stream (`docs/SESSION_RECORD.md`) with `source: analysis`: a handler
that could not take a message says so here, once per kind, and the
message's failure is that message's alone; the session goes on.

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
