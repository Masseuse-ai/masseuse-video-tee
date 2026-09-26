# Security

## Reporting

Report vulnerabilities privately through GitHub's
[security advisory form](https://github.com/FemLed/masseuse-video-tee/security/advisories/new)
for this repository. Do not open a public issue for a vulnerability. Reports
are acknowledged within three business days.

Findings about a live slot (`slot-N.tee.masseuse.ai`) are in scope: the
verifier in `verifier/` and the steps in `VERIFY.md` are there to be run
against production, and a check that should hold but does not is a report.

## What the enclave can and cannot do

- It receives one camera stream per session: the phone's WebRTC publish
  (WHIP), or a camera the user names, either reachable directly (RTSPS) or
  through the home-network connector
  ([FemLed/masseuse-camlink](https://github.com/FemLed/masseuse-camlink)).
  TLS and DTLS-SRTP terminate inside the enclave; the connector, the
  trainer and Cloudflare carry ciphertext or metadata only.
- It runs a container image whose digest is in the attestation token every
  client checks before sending media, on a production Confidential Space
  image with debugging disabled since boot, on an Intel TDX VM with the GPU
  in confidential-computing mode. The operator cannot SSH into it, redirect
  its logs, read its memory, or change what runs without changing the digest.
- It keeps no media: frames and audio samples live in memory for the
  duration of a decode and are not written anywhere. The data that leaves
  are the readings the analysis derives (numbers, never frames or sound)
  posted to the trainer, the view streamed back to the same user's phone
  (the picture, with the keypoints drawn on it only when that user asks),
  the same view (with the microphone, when asked) to the one
  live-streaming destination the user names from their phone when they
  open a live stream (README, "The live stream"; the destination and its
  key reach the enclave alone, and the trainer may only stop the stream),
  the phone's own camera picture copied, unchanged, to that user's own
  connector on their computer when they ask for it there and from their
  phone (README, "Trust boundary": through the connector's verified tunnel
  alone, to a certificate the enclave pinned, for OBS to work on; what
  comes back is shown as their face and never analysed),
  and, for a signed-in session, the session record (README,
  "Session records"): the keypoints, motion descriptors, audio
  measurements and the analysis's rows, written to the one bucket the
  attested environment names (`TEE_CAPTURE_BUCKET`) under the prefix the
  trainer's lease named, by the enclave's attested identity, which can
  create objects there and cannot read, overwrite or delete any.
- The model weights and the ACME account credential are readable only by an
  attested image digest through Workload Identity Federation; no service
  account, and no person, holds that access. The same principal is the only
  writer of the session records; a record's destination can be changed only
  by changing the attested environment, which the trainer's policy and the
  verifier both check.
- Its TLS key, its Ed25519 evidence key and its certificate are generated at
  boot and die with the VM. The attestation token names both keys, which
  is how a verifier knows the endpoint it reached is the enclave.

## Supply chain

Container images are built and signed only by the release workflow of this
repository from a tag, with SLSA provenance and a release stamp baked in;
the deployment pins the digest in `terraform/terraform.tfvars`, and the
trainer's published policy (`https://masseuse.ai/api/tee-policy`) pins the
signing key the attestation reports and a minimum release. `VERIFY.md` says
how the running digest is tied to this source; each release's digest is on
its GitHub Release. Third-party binaries in the image (MediaMTX, Caddy, the
camera connector's gateway) are pinned by digest or by release checksum
(`camlink.lock`).
