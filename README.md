# masseuse-video-tee

The enclave that every masseuse.ai camera stream is sent to, published so
that anyone can read what happens to their video and check that the machine
they are connected to is running exactly this.

masseuse.ai looks at a camera while you use it. The camera is either the
phone's own, a network camera behind you that the phone names, or a camera
on your home network carried by the open-source connector
[masseuse-camlink](https://github.com/FemLed/masseuse-camlink). Whichever it
is, the video is decrypted in two places only: on your device and inside a
Google Cloud Confidential Space VM (`a3-highgpu-1g`: Intel TDX, one H100 in
confidential-computing mode) in the dedicated project
`prod-masseuse-video-tee`. This repository is that VM's infrastructure, the
container image it runs, and the tools to verify both.

## What happens to your video and audio

Inside the enclave, in code that is in this repository:

- The stream is decoded into frames.
- A person detector (RT-DETRv4) finds the person in each frame.
- A keypoint detector (Sapiens2-1B) estimates body keypoints on the person.
- Regional motion descriptors are computed from the pixels around a few
  keypoint-anchored regions (optical-flow statistics: how much, how fast, in
  which direction a region moves).
- An annotated view (skeleton, boxes, a status line) is drawn on the frames
  and streamed back to the same device that sent the video, and to nothing
  else.
- When the stream carries an audio track (the phone's microphone travels
  with its camera unless you turn it off with `?mic=0`; a network camera's
  microphone, when it has one), the audio is decoded to 16 kHz mono and,
  every half second, a two-second window is classified into non-speech
  vocalization categories (the CED audio tagger, AudioSet labels such as
  breathing, groan, gasp, sigh, and "speech" so that talk can be told apart
  and set aside) along with its level and pitch. No speech recognition or
  transcription runs; nothing identifies the voice.
- Frames and audio are then discarded. Nothing is written to disk; the VM
  has no persistent storage and no operator can read its memory.

The keypoints, descriptors and vocalization labels, which identify nobody,
go to an analysis module that turns them into a few numbers per second for
the trainer (the masseuse.ai service on Cloud Run). The analysis module
runs inside the same enclave as a separate process, receives no frames and
no audio, and its logic is not published; its exact version is pinned by
hash in this repository so the attested image says which one is running.
The numbers, never frames or sound, are what leaves the enclave, over TLS to
the trainer's address that is itself part of the attestation.

All of that is `workload/`: `pixel/` is the decode geometry, person
detection, keypoint detection and motion descriptors; `audio/` is the audio
decode, the classifier binding and the level and pitch measurements;
`producer/` is the session shell that runs them, draws the overlay and
speaks to the analysis module over a local socket; `tee/` is the container
image and its entrypoint. `analysis/protocol.md` names every field that
crosses to the analysis module and every field that comes back.

Every published image is built from a tagged commit of this repository by
its release workflow (`.github/workflows/release.yml`) on GitHub Actions,
with SLSA provenance and a keyless signature, and the digest the enclave
attests is the digest that workflow built. `VERIFY.md` says how to check
that for the digest your session attested, and which digests are current.

## Trust boundary

- Cloudflare fronts `masseuse.ai` (HTML/JS, session API) but never carries
  SDP, media or the media capability, only the capability's SHA-256.
- The trainer (Cloud Run, operator-run) orchestrates leases and receives the
  readings. It does not proxy WHIP/WHEP for enclave slots.
- coturn is a TURN fallback the phone may use; it relays SRTP ciphertext.
- The operator (the Google Cloud project owner) cannot read enclave memory,
  SSH into the production image, redirect its logs, change the image without
  changing the digest every client checks, or obtain the DTLS keys or the
  capability.
- Residual: the JavaScript bundle and the allowed-digest policy are
  operator-served. Published digests with provenance, this verifier, and the
  connector (which runs the checks itself, outside the browser) are the
  mitigations.

The two kinds of external camera keep the same boundary:

- **A camera the user names (RTSPS, reachable from the internet).** The phone
  hands the enclave an `rtsps://` link (`PUT /ingest/source`, capability
  bearer, only after the phone has verified the attestation). The link is a
  credential and stays in the enclave: the trainer and Cloudflare learn only
  `source: external`, the enclave never logs it, nothing of it persists. The
  enclave probes the camera once for reachability and its certificate's
  SHA-256, then MediaMTX pulls RTSPS with that fingerprint pinned. Plain
  `rtsp://`, private, loopback, link-local, multicast, CGNAT and unspecified
  addresses and the VM's own address are refused before any connection. The
  path is cleared on teardown and by a lease for a different session.
- **A camera on the user's home network (masseuse-camlink).** The connector
  carries the camera's RTSPS *ciphertext* to the enclave, where
  `masseuse-camlink-gateway` (the second binary of that repository, running
  as one of the processes in this image) hands it to MediaMTX on loopback as
  if it were the camera. Nothing decrypts in between: the camera's TLS still
  terminates in MediaMTX with the certificate fingerprint pinned, so neither
  the connector nor the gateway holds the stream or can substitute one.
  Before dialing, the connector verifies the slot's attestation the way the
  phone does and pins the slot's TLS key to the SPKI hash in that
  attestation. It dials only the single private-network `host:port` the
  session names. The gateway binary in this image is pinned by release tag
  and checksum (`camlink.lock`) and rebuilt from the Go module proxy by the
  image build, which refuses to build unless the bytes match the signed,
  SLSA-attested release.

**One session per slot.** A slot serves one session at a time: the lease
holds a single capability hash, and the publish and overlay routes accept
only that capability as a bearer, so a second person cannot join a slot
that is someone else's.

**Hardware-signed evidence.** Each session is gated on a Google Cloud
Attestation OIDC token (TDX quote + GPU report → Google-signed JWT) whose
`eat_nonce` binds the in-enclave Ed25519 evidence key and the TLS leaf's
SPKI hash. The same kind of token is exchanged at STS for the only credential
that can read the model weights; the VM's service account has no data
access.

## Verify it yourself

`VERIFY.md` walks through it: build `verifier/` (Go), point it at a live
slot, and every claim above is checked against a token minted seconds ago
for your nonce, including the two things a browser cannot check (that the
TLS endpoint is the enclave, and the image signature). It also lists the
published digests and the signing key. `SECURITY.md` says how to report a
check that should hold and does not.

## What is in this repository

| Path | What |
| --- | --- |
| `workload/pixel/` | Everything that reads frames: decode geometry and stream handling (`live_pose.py`), RT-DETRv4 person detection (`vendor/rtdetrv4/`, `pose_track.py`), Sapiens2-1B keypoints (`pose_rows.py`, `keypoints.py`), regional motion descriptors (`motion.py`) |
| `workload/audio/` | Everything that reads sound: the audio track decoded to 16 kHz PCM (`audio_stage.py`), the CED non-speech vocalization classifier binding (`ced.py`; the library and weights are built into the image, `Dockerfile.tee`), level and pitch (`pitch.py`, `audio_features.py`) |
| `workload/producer/` | The session shell (`producer.py`), the overlay drawn back to the phone, the WHIP/WHEP relay proxy with the capability and evidence checks, the external-camera and connector ingest, the enclave's boot helpers (attestation, ACME, weights and analysis bundle fetch), the socket client to the analysis module |
| `workload/tee/` | The image: `Dockerfile` (base: Python, torch, ffmpeg), `Dockerfile.tee` (MediaMTX, Caddy, the connector gateway, the launch policy), `entrypoint.sh`, `Caddyfile`, `mediamtx.tee.yml` |
| `workload/tests/` | The workload's tests, CPU only |
| `analysis/protocol.md` | What crosses the socket to the analysis module and what comes back |
| `analysis.lock` | The analysis bundle (version and SHA-256) the image will run; copied into the image |
| `camlink.lock` | The `masseuse-camlink-gateway` release (tag and checksum) the image carries |
| `.github/workflows/release.yml` | The build: images to `ghcr.io`, SLSA provenance, keyless signature, promotion by digest into the enclave's registry, KMS signature |
| `terraform/` | The project: service account, Workload Identity Federation pool and providers keyed to the attestation, models bucket, Artifact Registry, static IP, firewall, the slot VM(s), the trainer's start/stop role, the KMS signing key, the release workflow's identity, the sweeper, VPC Service Controls (off until the project has an organization) |
| `verifier/` | `tee-verify`, the standalone attestation checker |
| `tools/` | `gts-acme-test.sh`: the rig that established Google Trust Services tolerates a fresh certificate on every boot |
| `build.sh` | The same two image builds, locally, without a push |
| `VERIFY.md` | How to verify a slot and the source of its image; published digests; signing key |
| `docs/OPERATIONS.md` | Running it: infrastructure, build, boot, the on-demand lifecycle, signing, the sweeper, certificates |

## How it runs

A slot exists only while a session needs it. When a masseuse.ai visitor taps
Enable camera the trainer starts the VM (the one thing its service account
may do here); about three minutes later the enclave holds a fresh TLS
certificate from Google Trust Services issued to a key generated inside it,
the model weights are in a tmpfs, and the slot attests and takes the lease.
A minute after the session ends the enclave exits on its own and the VM is
stopped. Every boot starts from nothing: no key, certificate, weight or
frame survives it. `docs/OPERATIONS.md` has the details, timings and the
runbook.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). Model weights are not in this repository;
each model's own license applies to it.
