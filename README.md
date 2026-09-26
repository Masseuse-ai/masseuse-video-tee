# masseuse-video-tee

The enclave that every masseuse.ai camera stream is sent to, published so
that anyone can read what happens to their video and check that the machine
they are connected to is running exactly this.

masseuse.ai looks at a camera while you use it. The camera is either the
phone's own, a network camera behind you that the phone names, or, through
the open-source connector
[masseuse-camlink](https://github.com/FemLed/masseuse-camlink) running on
the computer in the room, that computer's own camera and microphone or a
camera on your home network. Whichever it is, the video is decrypted in two
places only: on your own devices and inside a Google Cloud Confidential
Space VM (`a3-highgpu-1g`: Intel TDX, one H100 in confidential-computing
mode) in the dedicated project `prod-masseuse-video-tee`. This repository is that VM's infrastructure, the
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
  else. It is encoded in four renditions at once, from the one picture:
  the full view at 30 fps, the full view at 15 fps, and three quarters of
  its size at 15 fps at two bit rates. They give things up in that order -
  frame rate first, then resolution, then bits - and the device picks one
  and moves between them, stepping down when it drops frames and back up
  when it stops; which rendition it watches changes nothing about what is
  analysed.
- A device that cannot keep up also sends its own camera at fewer frames
  a second (never at a lower resolution or bit rate). The frames it sends
  are laid on a fixed 30 fps grid with repeats where it sent none, and the
  keypoint detector's picks skip a repeat of the frame it last posed for
  the next frame that differs, so it keeps working from distinct, sharp
  pictures at any upload rate down to ten a second.
- With a second camera - a fixed one behind the user, through a connector
  or named directly (the trust boundary below) - the phone's own camera
  stays live too. The fixed camera's picture is the session's: the body
  view, which everything above runs on. The phone's picture is decoded
  beside it, its person and keypoints found by the same detectors at the
  same cadence, and drawn as an inset in the top-right corner of the
  annotated view with those keypoints, cropped to follow the face and
  mirrored the way the phone's own preview was when its camera faces the
  user (the phone says which, `PUT /ingest/view`). The two pictures are
  lined up by the time each frame was taken - both senders time their
  frames with their own clocks (RTCP sender reports), the relay keeps
  those times, and `workload/reader/` reads them beside the frames - so
  the inset shows the same moment as the body view. Only the phone's
  microphone is listened to then, being the one near the user's face; the
  fixed camera's sound, if it has any, is not read.
- When the stream carries an audio track (the phone's microphone travels
  with its camera unless you turn it off with `?mic=0`; a network camera's
  microphone, when it has one), the audio is decoded to 16 kHz mono and,
  every half second, a two-second window is classified into non-speech
  vocalization categories (the CED audio tagger, AudioSet labels such as
  breathing, groan, gasp, sigh, and "speech" so that talk can be told apart
  and set aside) along with its level and pitch. No speech recognition or
  transcription runs; nothing identifies the voice.
- Frames and audio are then discarded. Nothing is written to disk; the VM
  has no persistent storage and nobody at masseuse.ai can read its memory.

The keypoints (the face view's included, as their own rows), descriptors
and vocalization labels, which identify nobody, go to an analysis module
that turns them into a few numbers per second for the masseuse (the
masseuse.ai service on Cloud Run). The analysis module
runs inside the same enclave as a separate process, receives no frames and
no audio, and its logic is not published; its exact version is pinned by
hash in this repository so the attested image says which one is running.
The numbers, never frames or sound, are what leaves the enclave, over TLS to
the masseuse's address that is itself part of the attestation.

All of that is `workload/`: `pixel/` is the decode geometry, person
detection, keypoint detection and motion descriptors; `audio/` is the audio
decode, the classifier binding and the level and pitch measurements;
`producer/` is the session shell that runs them, draws the overlay and
speaks to the analysis module over a local socket; `tee/` is the container
image and its entrypoint. `analysis/protocol.md` names every field that
crosses to the analysis module and every field that comes back.

Every image is built from a tagged commit of this repository by its
release workflow (`.github/workflows/release.yml`) on GitHub Actions, with
SLSA provenance and a keyless signature, then signed with a key only that
workflow can use and stamped with its release tag and commit. The
attestation the enclave gives your device names the digest, the signature
and the stamp, so which release is running is read off the running image,
never off a list kept here. `VERIFY.md` says how to check that for the
digest and release your session attested; the digest and validation
record of each release is on its GitHub Release.

## Trust boundary

- Cloudflare fronts `masseuse.ai` (HTML/JS, session API) but never carries
  SDP, media or the media capability, only the capability's SHA-256.
- The masseuse (Cloud Run, run by masseuse.ai) orchestrates leases and
  receives the readings. It does not proxy WHIP/WHEP for enclave slots.
- coturn is a TURN fallback the phone may use; it relays SRTP ciphertext.
- masseuse.ai itself, meaning any employee, contractor or administrator
  working on its behalf, including the owners of the Google Cloud project,
  cannot read enclave memory, SSH into the production image, redirect its
  logs, run an image the release workflow did not build and sign from a tag
  of this repository, or obtain the DTLS keys or the capability.
- Residual: the JavaScript bundle and the policy (which signing key and
  minimum release the clients accept) are served by masseuse.ai. The
  provenance on every release, this verifier, and the connector are the
  mitigations: the connector runs the checks itself, outside the browser,
  treats the served policy as something that can only tighten the anchors
  compiled into its reproducible build, and verifies each attested digest's
  signature and SLSA provenance against the public registry and the Sigstore
  transparency log before it sends a frame.

The external cameras keep the same boundary, with one difference the third
kind makes plain:

- **A camera the user names (RTSPS, reachable from the internet).** The phone
  hands the enclave an `rtsps://` link (`PUT /ingest/source`, capability
  bearer, only after the phone has verified the attestation). The link is a
  credential and stays in the enclave: the masseuse and Cloudflare learn only
  `source: external`, the enclave never logs it, nothing of it persists. The
  enclave probes the camera once for reachability and its certificate's
  SHA-256, then MediaMTX pulls RTSPS with that fingerprint pinned. Plain
  `rtsp://`, multicast, CGNAT and unspecified addresses and the VM's own
  address are refused before any connection; a private, loopback or
  link-local address is not dialed from the enclave at all but reached
  through the connector's tunnel, below.
- **A camera on the user's home network that the session names
  (masseuse-camlink, relaying).** The connector carries the camera's RTSPS
  *ciphertext* to the enclave, where `masseuse-camlink-gateway` (the second
  binary of that repository, running as one of the processes in this image)
  hands it to MediaMTX on loopback as if it were the camera. Nothing
  decrypts in between: the camera's TLS still terminates in MediaMTX with
  the certificate fingerprint pinned, so neither the connector nor the
  gateway holds the stream or can substitute one. Before dialing, the
  connector verifies the slot's attestation the way the phone does, against
  a policy it only lets tighten the anchors compiled into it (the signing
  key, the minimum release, this repository, the project and registry the
  slot runs from), pins the slot's TLS key to the SPKI hash in that
  attestation, and confirms in the public registry and the Sigstore
  transparency log that the attested digest is what this repository's
  release workflow signed and built at the release the image is stamped
  with. It dials only the single private-network `host:port` the session
  names. The gateway binary in this image is pinned by release tag and
  checksum (`camlink.lock`) and rebuilt from the Go module proxy by the
  image build, which refuses to build unless the bytes match the signed,
  SLSA-attested release.
- **The computer's own camera, or a home camera the connector names
  (masseuse-camlink, serving).** The connector is itself an RTSPS server,
  reachable only through its tunnel: the phone hands the enclave the fixed
  link `rtsps://127.0.0.1:7443/camera`, the enclave treats the loopback
  address as a tunnel target like any private one, probes and pins the
  connector's certificate through the tunnel as it would a camera's, and
  MediaMTX pulls the stream the same way. What the connector serves is the
  computer's camera and microphone (captured with ffmpeg, only while a
  session is reading) or a camera on the home network that it pulls itself,
  terminating that camera's TLS on the computer and pinning its certificate
  there. Here the connector is not a relay of ciphertext: it is the camera,
  and holds the picture in the clear on the person's own computer, as any
  camera program does. What holds unchanged is that the stream leaves that
  computer only inside TLS this attested image terminates, that the
  connector sends it to nothing but the slot it verified, that the enclave
  is told of the camera only by the phone's capability, and that the
  masseuse.ai service never carries a frame: it learns the connector's name
  for the camera (`POST /api/camlink/source`, so the phone can show it) and
  nothing else. That the connector does what it says is checked the way
  this image is: it is open source, and its releases are reproducible and
  signed.

With any of the three, the phone keeps publishing its own camera to the
slot over WHIP as before, and the two pictures meet only inside this
image: the fixed camera's as the body view, the phone's as the face inset
drawn over it, both returned over the one WHEP leg to the same phone. The
phone's capability is what opens `PUT /ingest/view`, the one control over
how its picture is drawn (mirrored or not); the masseuse learns the layout
and the inset's place from the slot's status (`overlay.view`) and nothing
of either picture.

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
TLS endpoint is the enclave, and the image signature), and, with
`slsa-verifier`, that the running digest is this source at the release the
image says it is. It also gives the signing key. `SECURITY.md` says how to
report a check that should hold and does not.

## What is in this repository

| Path | What |
| --- | --- |
| `workload/pixel/` | Everything that reads frames: decode geometry and stream handling (`live_pose.py`), RT-DETRv4 person detection (`vendor/rtdetrv4/`, `pose_track.py`), Sapiens2-1B keypoints (`pose_post.py`, `pose_rows.py`, `keypoints.py`), the captured CUDA graphs both forwards replay (`gpu_graph.py`) and the debug-slot bench of the pose graph over batches of crops (`pose_bench.py`), regional motion descriptors (`motion.py`) |
| `workload/audio/` | Everything that reads sound: the audio track decoded to 16 kHz PCM (`audio_stage.py`), the CED non-speech vocalization classifier binding (`ced.py`; the library and weights are built into the image, `Dockerfile.tee`), level and pitch (`pitch.py`, `audio_features.py`) |
| `workload/producer/` | The session shell (`producer.py`), the overlay drawn back to the phone, the WHIP/WHEP relay proxy with the capability and evidence checks, the external-camera and connector ingest, the two views' clocks (`sync.py`), the enclave's boot helpers (attestation, ACME, weights and analysis bundle fetch), the socket client to the analysis module |
| `workload/reader/` | `stream-reader` (Go): reads a live stream's video track off the relay and hands the producer each decoded frame with the time its sender gave it, so two cameras' views line up on their senders' clocks; the frame record it writes is documented in `record/record.go` |
| `workload/tee/` | The image: `Dockerfile` (base: Python, torch, ffmpeg, `stream-reader`), `Dockerfile.tee` (MediaMTX, Caddy, the connector gateway, the launch policy), `entrypoint.sh`, `Caddyfile`, `mediamtx.tee.yml` |
| `workload/tests/` | The workload's tests, CPU only |
| `analysis/protocol.md` | What crosses the socket to the analysis module and what comes back |
| `analysis.lock` | The analysis bundle (version and SHA-256) the image will run; copied into the image |
| `camlink.lock` | The `masseuse-camlink-gateway` release (tag and checksum) the image carries |
| `.github/workflows/release.yml` | The build: images to `ghcr.io` stamped with the tag and commit, SLSA provenance, keyless signature, promotion by digest into the enclave's registry, KMS signature, the GitHub Release that records the digest |
| `terraform/` | The project: service account, Workload Identity Federation pool and providers keyed to the attestation, models bucket, Artifact Registry, a subnet per region and a static IP per slot, firewall, the slot VMs spread over `slot_zones` (even slots in us-central1-a, odd in us-east5-a), the masseuse's start/stop role, the KMS signing key, the release workflow's identity, the sweeper, VPC Service Controls (off until the project has an organization) |
| `verifier/` | `tee-verify`, the standalone attestation checker |
| `tools/` | `gts-acme-test.sh`: the rig that established Google Trust Services tolerates a fresh certificate on every boot |
| `build.sh` | The same two image builds, locally, without a push |
| `VERIFY.md` | How to verify a slot and the source of its image: what identifies the image (signature, release stamp, provenance), the signing key, where the release history lives |
| `docs/OPERATIONS.md` | Running it: infrastructure, build, boot, the on-demand lifecycle, signing, the sweeper, certificates |

## How it runs

A slot exists only while a session needs it. When a masseuse.ai visitor taps
Enable camera the masseuse starts the VM (the one thing its service account
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
