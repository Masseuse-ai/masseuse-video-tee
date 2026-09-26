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
- It keeps nothing: frames live in memory for the duration of a decode and
  are not written anywhere. The only data that leaves are the readings the
  analysis derives (numbers, never frames) posted to the trainer, and the
  annotated view streamed back to the same user's phone.
- Its TLS key, its Ed25519 evidence key and its certificate are generated at
  boot and die with the VM. The attestation token names both keys, which
  is how a verifier knows the endpoint it reached is the enclave.
- The model weights and the ACME account credential are readable only by an
  attested image digest through Workload Identity Federation; no service
  account, and no person, holds that access.

## Supply chain

Container images are pinned by digest in `terraform/terraform.tfvars` and in
the trainer's published policy (`https://masseuse.ai/api/tee-policy`), signed
with a Cloud KMS key the attestation reports, and listed with their
provenance in `VERIFY.md`. Third-party binaries in the image (MediaMTX,
Caddy, the camera connector's gateway) are pinned by digest or by release
checksum (`camlink.lock`).
