# v0.12.4 (bundle 2026.09.26-2; gateway masseuse-camlink v0.14.0): the
# first image built, signed and published at this repository's home in the
# Masseuse-ai organization (#2, the verifier's defaults following); the
# workload and the analysis bundle are v0.12.3's. Its provenance names
# github.com/Masseuse-ai/masseuse-video-tee and its public copy sits on
# ghcr.io/masseuse-ai/masseuse-video-tee, which the trainer's policy and
# masseuse-camlink's floors (v0.26.0) name from this roll on. Released
# 2026-09-26 21:34 UTC as
# sha256:51bbd1be16fb3c3db6cf32fc33e471d681b0e423ea29a25d1c29028a06008502;
# rolled 2026-09-26 with no slot running (this apply and the trainer's
# records.tf), the trainer's policy first. v0.12.3 goes first into
# candidate_image_digests as the pin-back, and pinning back to it means
# the trainer's policy and the connectors' floors go back to the previous
# home with it; v0.12.2 and the five before it stay, and v0.10.7 behind
# them.
# v0.12.3 (bundle 2026.09.26-2; gateway masseuse-camlink v0.14.0): the
# workload alone (#69), the analysis bundle unchanged. The live stream
# accepts rtmp:// destinations beside rtmps:// (anything else refused as
# bad-scheme; the private-address check as before), with a per-service
# video bitrate cap in the re-encode branch (onlyfans.com 2500k, 4500k
# otherwise); the HUD card's wells are composited on their own layer (the
# fills blend instead of replacing the ink), the audience row is gone, an
# idle card says paused, tiles whose state is none are skipped. Released
# 2026-09-26 14:13 UTC as
# sha256:bec37d947de59fb51e293714f5d389e113544a5ad1ee3958eca57b40a70f002c;
# rolled 2026-09-26 with no slot running (this apply and the trainer's
# records.tf), the trainer at revision 00187 first (the Live show section
# and the overlay pages). v0.12.2 goes first into candidate_image_digests
# as the pin-back (no session served on it); v0.12.1 and the five before
# it stay, and v0.10.7 behind them.
# v0.12.2 (bundle 2026.09.26-2; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone (#68), a correction to 2026.09.26-1: the reading's
# `face.present` is back to the rule the hazard model was fitted on (the
# newest placed row is the very last row); 2026.09.26-1 had held it for a
# second after the last placed row, as the `face` message does, and the
# hazard's trend windows moved with the filled seconds (p30 by over 0.05 on
# 13% of a replayed session's seconds). Replayed, the reading's `face`
# object is now identical to 2026.09.25-6's on every shared field; `wire`
# is the only addition. The `face` message keeps the hold. Nothing in the
# workload changes. Released 2026-09-26 04:30 UTC as
# sha256:a852b65747c3c14cd845ad1ba7066ebdd097e930d8cb5a101b551f69307a7a5a;
# rolled 2026-09-26 with no slot running (this apply and the trainer's
# records.tf). v0.12.1 goes first into candidate_image_digests as the
# pin-back (no session served on it); v0.12.0 and the four before it stay,
# and v0.10.7 behind them.
# v0.12.1 (bundle 2026.09.26-1; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone (#67), after the first live session on the face
# frames (2026-09-25). The frame's unit channels are on the display's
# scale, 100 at a full movement in inner-canthal units (the `face`
# constants' wire.displayRange), where the reading's `channels` stay the
# detector's (three robust units of the baseline minute's jitter, on which
# a still face's closed lips read 100 in 78% of that session's readings);
# the reading's `face` object carries the same twenty-seven as `wire`. A
# face is present while one was placed within a second of the newest row,
# in the frame and the reading alike (a row dropped under load or one the
# fit could not place is no longer the face gone for its instant). Nothing
# in the workload changes. Released 2026-09-26 03:48 UTC as
# sha256:7cd35102149869e0e670dc00418397524ed843f91a8f60f7020eed30fc003278;
# rolled 2026-09-26 with no slot running (this apply and the trainer's
# records.tf), the trainer at revision 00186 first. v0.12.0 goes first into
# candidate_image_digests as the pin-back (one session served on it, the
# live test); v0.11.4 and the three before it stay, and v0.10.7 behind them.
# v0.12.0 (bundle 2026.09.25-6; gateway masseuse-camlink v0.14.0): the
# face's frames for the user's own display (#66). The analysis sends a
# `face` message after each face row at most every 0.2 s (hello's
# faceFrameIntervalS): the channel values the reading's `face` object
# carries, the clock and three flags, nothing else and never a keypoint;
# the producer forwards them to the trainer alone, at the readings' URL
# with `face` for its last segment, over one connection the Poster now
# keeps open across sends, frames in a slot of their own that never
# displaces a reading (workload/producer/sinks.py). Not the capture, not the
# record, not the session's event stream. The reading and every field
# from before are the same. Released 2026-09-25 23:20 UTC as
# sha256:62b5735b544b7eeaffb150631c38fd80a29094b52e65f81aef28bdaf727d5770;
# rolled 2026-09-26 with no slot running (this apply and the trainer's
# records.tf), the trainer's /face route live first (revision 00184).
# v0.11.4 goes first into candidate_image_digests as the pin-back; no
# session has served on v0.11.4, v0.11.3, v0.11.2 or v0.11.1, so all stay,
# and v0.10.7 behind them.
# v0.11.4 (bundle 2026.09.25-5; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone (#65), stance/v4 and the contraction rate's
# confidence. The stance's rest lines learn only from rows with a hip and a
# knee in frame and re-seed from a body still 30 s a unit off them (a
# session whose first rows were a climb onto the bed read `unknown` for
# 2.7 minutes lying down); the reading carries `rhythmSnr` and
# `rateConfidence` beside the contraction rate, no onsets emitted while the
# best channel shows no rhythm at the contraction period; the posture object
# carries `buckPeakedness` as a diagnostic. The fields from before are the
# same. Released 2026-09-25 22:31 UTC as
# sha256:cdeb9b199d0fa2cff7eaeae3cb2f7b2c02e7683437e6912bdd536156b86a15a1;
# rolled 2026-09-25 with no slot running (this apply and the trainer's
# records.tf). v0.11.3 goes first into candidate_image_digests as the
# pin-back; no session has served on v0.11.3, v0.11.2 or v0.11.1, so all
# stay, and v0.10.7 behind them.
# v0.11.3 (bundle 2026.09.25-4; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone (#64). The fixed classifier that judges each
# measured vocal span is a newer fit over the corpus's four verification
# rounds (1,197 reviewed spans against 897; out of fold PR-AUC 0.763,
# precision 0.70 at recall 0.70; a logistic over the same 82 features,
# threshold 0.5893); `hello.vocal.head.version` tells it from the one
# before, the fields are the same. Released 2026-09-25 17:47 UTC as
# sha256:fad8cbc4c42094308b42a28204640446677c79135fbc46c5f08ac6df50e07a46;
# rolled 2026-09-25 with no slot running (this apply and the trainer's
# records.tf). v0.11.2 goes first into candidate_image_digests as the
# pin-back; no session has served on v0.11.2 or v0.11.1 either, so both
# stay, and v0.10.7 behind them.
# v0.11.2 (bundle 2026.09.25-3; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone, stance/v3 (#63). The posture word is `unknown`
# rather than a guess while the hips' rest line is not yet credible and the
# row's shape fits both a lying posture and a kneeling one (the same
# keypoints from a camera behind him; a run that opens lying down reads
# prone as before from the rows that fit one shape alone); a hips' line
# that would put the hips' rest above the shoulders' rest is not used until
# a lying stretch corrects it; the along-the-body kneeling shape wants the
# shank flat toward the camera. The fields are the same. Released 2026-09-25
# 15:24 UTC as
# sha256:8944a97194b96522e2a611e5eaf6f2c121b38981fa32339bd692d236567910e3;
# rolled 2026-09-25 with no slot running (this apply and the trainer's
# records.tf). v0.11.1 goes first into candidate_image_digests as the
# pin-back; no session has served on it either, so v0.10.7 stays too.
# v0.11.1 (bundle 2026.09.25-2; gateway masseuse-camlink v0.14.0): the audio
# stage sends the analysis the classifier's whole class table
# (hello.audioLabels, audio.all, segment.ced.all) and two more spectral
# statistics per measured span (#60), and the analysis bundle judges each
# span with a fixed classifier over them (vocal/v3, #61/#62) instead of the
# classifier's top label: the decision rows carry the head's score, type
# and probabilities, `rejected` joins the verdicts, breath under the
# threshold stays `breathing` for the pacing's resting-rate term; the
# reading's vocal block keeps its shape. v0.11.0 (bundle 2026.09.25-1,
# tagged 04:57 UTC) counted breath as rejected and is not rolled; its
# Release stands as the tag ruleset requires. Released 2026-09-25 05:11 UTC
# as
# sha256:1d7e5a50e7e417a3f2d72b787b9593e03e5e26bf83c4edf66baa0532053b7373;
# rolled 2026-09-25 with no slot running (this apply and the trainer's
# records.tf). v0.10.7 stays in candidate_image_digests as the pin-back
# until a session has served on this one.
# v0.10.7 (bundle 2026.09.24-2; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone. The readings' face object carries `hazard` (face/v5):
# the probability the response peaks within 30 and 60 s, read from the face
# object's own fields by a fixed model named in the ready message's face
# constants; null while the face is not present or its baseline not ready;
# every field from before computed as before, the stance of 2026.09.24-1
# unchanged. v0.10.6 is a tag made in error at the previous head (its release
# run cancelled, no Release; the tag ruleset keeps it). Released 2026-09-24
# 21:14 UTC as
# sha256:3ff6f9ea3445cc83b668c14a2d13ab368e0b7e02fe63961b77ced5c03ae3db37;
# rolled 2026-09-24 with no slot running (this apply and the trainer's
# records.tf). No session served on v0.10.5, so both v0.10.5 and v0.10.4
# stay in candidate_image_digests until the first session on this one has
# served: it proves the stance bundle and the hazard bundle at once.
# v0.10.5 (bundle 2026.09.24-1; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone. The readings' positioning block carries a stance
# (prone, knee_chest, wariza, seated, standing, supine, side, unknown)
# judged from the 21 body keypoints in the body's own units against each
# landmark's rest line, so the camera may be anywhere around him; the
# posture block reads it (its hip numbers null while he is upright,
# hipRiseBu beside them) and the trainer's HUD and controller say the
# hips are lifting only in knee_chest. Every field from before is computed
# as before; face/v4 unchanged. Released 2026-09-24 17:23 UTC as
# sha256:c5392a93c1bfda5975aa6c8545719b4a3cf6b403c9be86d87ae9af436d591863;
# rolled 2026-09-24 with no slot running (this apply and the trainer's
# records.tf); v0.10.4 stays the pin-back until a session has served on
# this one.
# v0.10.4 (bundle 2026.09.22-2; gateway masseuse-camlink v0.14.0): a fix
# to 2026.09.22-1's face block: its per-second aggregates (headSpeed,
# regions) are read by the newest face row's clock, since live a face
# row's bodyAtS leads the body clock the readings are posted on by
# several seconds (the first session on v0.10.3: a median 5.4 s), which
# left `regions` null on every reading. Released 2026-09-22 18:31 UTC as
# sha256:9e457ef9675163b5eb507467b0795c9abadb6920f1a64ddba9d75c202db056ca;
# rolled 2026-09-22 18:33 UTC with no slot running (this apply and the
# trainer's records.tf), slot-0 hand-booted at 18:33 and tee-verify
# -expect-release v0.10.4 passed at 18:37 (24 of 24). v0.10.3 and v0.10.1
# stay the pin-backs until a session has served on this one.
# v0.10.3 (bundle 2026.09.22-1; gateway masseuse-camlink v0.14.0): the
# analysis bundle alone. Its face block (face/v4) adds the head's speed,
# the face regions' displacement composites and a signed outer-brow
# channel, validated against the corpus study's own measures before the
# build; every field from before is computed as before. Released
# 2026-09-22 17:31 UTC as sha256:8adaddde12ae97b2dba0e8fd92b2ca56fe47760ebba1f9af3e91dbb19106e146
# (the Release's digest). Rolled 2026-09-22 17:44 UTC with no slot
# running (this apply and the trainer's records.tf), slot-0 hand-booted
# at 17:45 and tee-verify -expect-release v0.10.3 passed at 17:49 (24 of
# 24); the previous digest stays the pin-back until a session has run
# end to end on this image.
# v0.10.1 (bundle 2026.09.18-1; gateway masseuse-camlink v0.14.0): two
# fixes in the workload, nothing new to choose. The audio stage reads an
# ffmpeg child to the end of its pipe before deciding anything about it
# (it respawned on exit status, losing a stream's last seconds on every
# reconnect and starting a file over), and the keypoint parts' boxW/boxH
# are the box's width and height (they were w - x and h - y: the tracker's
# xywh box read as two corners; docs/SESSION_RECORD.md says how to read the
# parts written before). Rolled 2026-09-20; hand-booted and verified is the
# gate, then a session end to end.
container_image        = "us-central1-docker.pkg.dev/prod-masseuse-video-tee/masseuse-video-tee/masseuse-video-tee@sha256:51bbd1be16fb3c3db6cf32fc33e471d681b0e423ea29a25d1c29028a06008502"
container_image_digest = "sha256:51bbd1be16fb3c3db6cf32fc33e471d681b0e423ea29a25d1c29028a06008502"
capture_bucket         = "masseuse-ai-prod"
# v0.10.0 (bundle 2026.09.18-1; gateway masseuse-camlink v0.14.0): the
# phone's picture to the person's own connector for OBS, at their asking
# there and from their phone (PUT /ingest/share; ffmpeg -c copy through a
# TLS bridge pinned to the connector's leaf, via the gateway's own
# listener 7442), and the connector's front-facing camera as the face view
# (PUT /ingest/face-source, the face-ext path, shown and never analysed).
# With nothing chosen in the connector the slot runs as v0.9.0 did. The
# image before this roll, the pin-back until v0.10.1 has served a session;
# rolled 2026-09-19.
# v0.9.0 (bundle 2026.09.18-1): the view is the picture. No person box, no
# status lines; the keypoints drawn only when the phone asks (PUT
# /ingest/view {overlay: keypoints}), clean by default; /ingest/status says
# what is drawn (view). The image before this roll, the pin-back until
# v0.10.0 has served a session; rolled 2026-09-19 15:48 UTC, hand-booted
# and verified.
# v0.8.7 (bundle 2026.09.18-1): the live stream, opt in: the view to an
# rtmps:// destination the user's phone names (PUT /ingest/egress, the
# capability as the bearer), the HUD card the trainer words into it (PUT
# /overlay/hud), the trainer able to stop it (POST /egress/stop) and to
# read it in /ingest/status. The image before this roll, the pin-back until
# v0.9.0 has served a session; rolled 2026-09-19 05:26 UTC, hand-booted and
# verified, no session served yet.
# v0.8.6 (bundle 2026.09.18-1): the face view's pose rows carry the face
# block (#48) and the analysis bundle reads it (face/v3: the ready message
# carries the face constants). The image before this roll, the pin-back
# until v0.8.7 has served a session; rolled 2026-09-18.
# v0.8.5 (bundle 2026.09.17-1: keypoint names from the Sapiens2
# definition, hand sides corrected): the image before this roll, the
# pin-back until v0.8.6 has served a session; hand-booted and verified
# 2026-09-18 04:59 UTC, no session served.
# v0.8.4 (bundle 2026.09.17-1: a /stop or /teardown naming another
# session refused, a /stop leaving the lease and record standing): the
# pin-back before that.
# v0.8.3 (bundle 2026.09.17-1: v0.8.2 plus the telemetry rows' stage
# timings): the pin-back before that. It served the two sessions of 2026-09-17 10:09 and
# 10:15 (the enclave healthy in both: the audio hop 35 ms, the 6 fps
# buffer full; the phone's capture and the trainer's slot pool were the
# faults).
# v0.8.2 (bundle 2026.09.17-1: the analysis link resumes a broken session,
# the record's log stream, the pitch contour vectorised, the descriptor
# queue keeps the 6 fps grid): the pin-back before that; no session served
# it before v0.8.3 replaced it.
# v0.8.1 (the session record is the lease's, bundle 2026.09.15-1): the
# pin-back before that. It served the 2026-09-16 session whose readings
# stopped (its analysis's fault, not the record's).
# v0.8.0 (the session record, bundle 2026.09.15-1): the pin-back before
# that; it served one session on 2026-09-15 (two production runs; the
# second run's hello, summary and one window's parts were refused as
# already there: what v0.8.1 fixed).
# v0.7.0 (midline valley along a per-session axis, bundle 2026.09.14-1):
# the pin-back before that. Pinning back to it means the trainer's policy
# goes back too (tee_policy.require_capture_bucket = false, min_release =
# "v0.4.0"): it attests no capture bucket.
# v0.6.0 (four renditions, bundle 2026.09.10-2): dropped 2026-09-15 with
# this roll; v0.7.0 and v0.8.0 have both booted slots since.
# v0.5.1 (one rendition, 9 fps picks): dropped 2026-09-14; v0.6.0 booted
# slots on 2026-09-14 and is the pin-back now.
# v0.4.8, the last 6 fps image, kept as the fallback until the 9 fps
# end-to-end session gate passes (pin back here if it does not).
# v0.5.0 (detect stride 3, load gate busy 0.916) dropped 2026-09-12 once its
# load gate was superseded; no slot ever served a session on it.
candidate_image_digests = ["sha256:bec37d947de59fb51e293714f5d389e113544a5ad1ee3958eca57b40a70f002c", "sha256:a852b65747c3c14cd845ad1ba7066ebdd097e930d8cb5a101b551f69307a7a5a", "sha256:7cd35102149869e0e670dc00418397524ed843f91a8f60f7020eed30fc003278", "sha256:62b5735b544b7eeaffb150631c38fd80a29094b52e65f81aef28bdaf727d5770", "sha256:cdeb9b199d0fa2cff7eaeae3cb2f7b2c02e7683437e6912bdd536156b86a15a1", "sha256:fad8cbc4c42094308b42a28204640446677c79135fbc46c5f08ac6df50e07a46", "sha256:8944a97194b96522e2a611e5eaf6f2c121b38981fa32339bd692d236567910e3", "sha256:1d7e5a50e7e417a3f2d72b787b9593e03e5e26bf83c4edf66baa0532053b7373", "sha256:3ff6f9ea3445cc83b668c14a2d13ab368e0b7e02fe63961b77ced5c03ae3db37", "sha256:9e457ef9675163b5eb507467b0795c9abadb6920f1a64ddba9d75c202db056ca", "sha256:c5392a93c1bfda5975aa6c8545719b4a3cf6b403c9be86d87ae9af436d591863", "sha256:8adaddde12ae97b2dba0e8fd92b2ca56fe47760ebba1f9af3e91dbb19106e146", "sha256:3df3052afd78691eb42f7f15e12bb5d968401afed21e358aa17bd17d130f8062", "sha256:5228d9f52290ae72f8990f2a8b8d489880f53ec2bb00507c7d808a630826b029", "sha256:b41ad28fb1b294390fa396784a3e6fce8a8e0acc07d5f6ddca8fb9827924ff30", "sha256:7260ec095a1823a9f7aa43123d538346d705ac565ae36fb8170418cc2c4e62dc", "sha256:9659d3010c401ee1e0a4f3227b9c1ac5ae89cba365509b6a94b1ee227bb68f95", "sha256:1d5b1f00224dd4594a4698c1b242815c2ba29aab07f0513de6a710a9d056dfe0", "sha256:2014e9900ee1989f05e28590274add3cd3311023dd25a9e86e16b54d72506f17", "sha256:8c8535d6c264bcaed571f24645bb12436587bf1b45783499452e7c3840dd9262", "sha256:32ef8da505006d06a4f9a8e856319101122c56dc185d1bd0c2ab5b87f83949c3", "sha256:29460f10459ecfc8b585b1565b4a33b2922b7df042d5dfd3a40cb7f5e8c40f89", "sha256:8ba3be993c73eb594829636eef7c47fe857b67944e510fb1eb60d8c783217090", "sha256:37d9d67333985d1061801a78ab9786a767574f691e087a3fd623ca9a0d2523ac"]
debug_mode              = false
vm_running              = false
acme_contact_email      = "ops@femled.ai"
operator_members        = ["user:justin.wickett@gmail.com"]
sweeper_enabled         = true
require_signed_image    = true
