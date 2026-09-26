# Verifying a masseuse.ai video slot

masseuse.ai's camera pipeline runs in a Google Cloud Confidential Space
enclave: an `a3-highgpu-1g` VM with Intel TDX on the CPU and an NVIDIA H100
in confidential-computing mode. This document is how anyone, without an
account at FemLed or Google, checks that the enclave your phone (or your
home camera, through [masseuse-camlink](https://github.com/FemLed/masseuse-camlink))
would send video to is the one described here, and what that does and does
not prove. If you are not technical, hand it to someone who is.

## What is being claimed

1. Your camera's video is decrypted in exactly two places: your device and
   the enclave. Cloudflare, the trainer service (Cloud Run), the TURN relay,
   the home-network connector and FemLed's operators see ciphertext or
   nothing.
2. The enclave runs a published container image, by digest, on a
   production Confidential Space image with debugging disabled since boot
   (no SSH, no log redirection, no memory monitoring), on hardware Google
   attests to, with the GPU attestation-bound.
3. TLS for the signalling origin (`slot-N.tee.masseuse.ai`) terminates
   inside the enclave: the certificate's key was generated there and is
   named in the attestation.
4. The WebRTC media itself (DTLS-SRTP) terminates inside the enclave: every
   SDP answer's DTLS fingerprint is signed by an Ed25519 key generated in
   the enclave and named in the attestation, and your browser's own DTLS
   handshake then verifies that fingerprint.
5. FemLed operators cannot change what runs without changing the digest
   your device checks, and cannot read the model weights themselves: the
   only credential that can is minted from the attestation of an approved
   image.
6. What the enclave sends onward is derived readings (numbers), never
   frames: `README.md`, "What happens to your video".

## What you need

- The published digest list (below) and signing key fingerprint.
- `go` 1.22+ to build the verifier in `verifier/`.
- Optionally `cosign` to check the image signature independently.

## Run the verifier

```sh
cd verifier
go build -o tee-verify .
./tee-verify -origin https://slot-0.tee.masseuse.ai \
    -allowed-digests sha256:<digest from the list below> \
    -signer-key-ids <signing key fingerprint below>
```

Exit code 0 and `"passed": true` means every check below held for a token
minted seconds ago for your nonce. Add `-v` to print the decoded
attestation claims; add `-cosign <path> -cosign-public-key signer.pub` to
have cosign re-verify the image signature from the registry.

Each check, and why it matters:

| check | meaning |
| --- | --- |
| `jwt.signature`, `jwt.issuer` | the token is signed by Google's Confidential Space attestation service (`https://confidentialcomputing.googleapis.com`), whose keys are at `https://www.googleapis.com/service_accounts/v1/metadata/jwk/signer@confidentialspace-sign.iam.gserviceaccount.com`. Only the launcher inside a real Confidential Space VM can obtain one. |
| `jwt.audience` | the token was minted for this slot's origin, not lifted from another service. |
| `cs.swname`, `cs.hwmodel`, `cs.secboot` | Confidential Space on Intel TDX with Secure Boot. |
| `cs.dbgstat`, `cs.support_attributes.STABLE` | the production image family, debugging disabled since boot: the operator cannot SSH in, redirect the container's output, or read memory metrics. (`-allow-debug` accepts the debug posture; production verification must not pass it.) |
| `gpu.cc_mode`, `gpu.hwmodel` | the H100 is in confidential-computing mode and its device attestation is part of the token: the models run on protected GPU memory, not a plain GPU next to a TDX CPU. |
| `image.digest`, `image.reference` | the running container is one of the published digests, pulled from FemLed's registry for this project. |
| `image.env.TRAINER_URL` | the enclave posts its readings (numbers, never frames) to the trainer service you expect and nowhere else (the whole workload environment is in the token; `-v` shows it). |
| `image.signature` | the image carries a cosign signature by the KMS key whose fingerprint is published below; the launcher verified it at boot. Rebuilding the image with different code changes the digest and voids the signature. |
| `nonce.fresh` | `eat_nonce` contains the random nonce this run sent: the token was minted now, not replayed. |
| `nonce.evidence-key` | `eat_nonce` contains `sha256(evidence key)`: the Ed25519 key that signs DTLS fingerprints lives in this enclave. |
| `nonce.tls-spki` | `eat_nonce` contains `sha256(SubjectPublicKeyInfo)` of the certificate this very TLS connection negotiated: the TLS endpoint is the enclave, not a proxy in front of it. |
| `cosign.verify` | (optional) cosign independently verifies the registry signature over the running digest with the published public key. |

## Do it by hand

```sh
HOST=slot-0.tee.masseuse.ai
NONCE=$(head -c 24 /dev/urandom | base64 | tr '+/' '-_' | tr -d =)
curl -s "https://$HOST/attestation?nonce=$NONCE" | jq -r .token | cut -d. -f2 | tr '_-' '/+' | base64 -d 2>/dev/null | jq
curl -s https://$HOST/evidence-key | jq -r .publicKey | tr '_-' '/+' | base64 -d 2>/dev/null | openssl dgst -sha256 -binary | base64 | tr '+/' '-_' | tr -d =
openssl s_client -connect $HOST:443 -servername $HOST </dev/null 2>/dev/null \
  | openssl x509 -pubkey -noout | openssl pkey -pubin -outform DER | openssl dgst -sha256 -binary | base64 | tr '+/' '-_' | tr -d =
```

The two hashes must both appear in the token's `eat_nonce`, alongside your
`$NONCE`. Verify the token's RS256 signature against the JWKS URL above
with any JWT library.

A slot exists only while a session needs it (it boots when a visitor taps
Enable camera and stops a minute or two after the session), so a
connection refused or a DNS name that resolves to a silent address means
no slot is up, not that a check failed. Verify during a session of your
own, or start one and run the verifier from another machine.

## What the clients check

The masseuse.ai web app does the same verification in the browser before it
sends any media: JWT signature against the same JWKS, the same claims
against the policy the trainer publishes at
`https://masseuse.ai/api/tee-policy`, and the evidence-key binding. Then,
on every WebRTC leg, it verifies the enclave's Ed25519 signature over the
SDP answer's DTLS fingerprint, the session, the leg and its own nonce
(`X-Masseuse-Evidence`), and refuses the connection if it does not hold.
The browser cannot read the TLS certificate it negotiated, so the
`nonce.tls-spki` check is the verifier's job; the browser's DTLS
fingerprint check gives the media the same property.

The home-network connector (`masseuse-camlink`) runs the same checks
before it dials a slot (`internal/attest` there is a port of `verifier/`)
and additionally pins the slot's TLS key to the SPKI hash in the token.

## Published digests

| digest | since | notes |
| --- | --- | --- |
| `sha256:a1e224492978c5a90c3c1faad81b2ebd83dfd302675291db6373210b5d95331e` | 2026-09-09 | **`v0.1.0`**, the first image built from this repository: tag [`v0.1.0`](https://github.com/FemLed/masseuse-video-tee/releases/tag/v0.1.0) by [`release.yml`](.github/workflows/release.yml) run `34383875573`, SLSA provenance and a keyless signature on `ghcr.io/femled/masseuse-video-tee@sha256:a1e22449…` (same digest), promoted by digest into the registry above and signed there with the KMS key. Debug image (`confidential-space-debug`, `dbgstat=enabled`); not a production posture. `ubuntu:24.04` base, zstd layers, live streams decoded at a pinned geometry, the directly reachable external camera (`PUT /ingest/source`) and the home-network camera connector's gateway (`masseuse-camlink-gateway` from the public [FemLed/masseuse-camlink](https://github.com/FemLed/masseuse-camlink) release `v0.1.0`, reproduced by the image build from `camlink.lock`, which refuses any other bytes). The producer is split: the pixel path (`workload/pixel`, `workload/producer`) is this tree, and the analysis of its keypoints and descriptors is the pinned bundle `2026.09.09-1` (`analysis.lock`, SHA-256 `d5c53436…`), run under its own user behind the local socket (`analysis/protocol.md`). `tee-verify -allow-debug` with `-cosign` passed every check on the pinned slot 2026-09-09 18:11 UTC; `slsa-verifier verify-image` and `cosign verify` passed on the `ghcr.io` digest in the release's own `promote` job before the copy. |
| `sha256:c4d5dbb5a6c3c9f209772c28ad6fb810e64543b3697f399a0f8df11fbf23cb8b` | 2026-09-09 | **`v0.2.1`**: tag [`v0.2.1`](https://github.com/FemLed/masseuse-video-tee/releases/tag/v0.2.1), `release.yml` run `34392988036`, provenance and keyless signature on `ghcr.io/femled/masseuse-video-tee@sha256:c4d5dbb5…`, promoted and KMS-signed the same way. Same posture and base as `v0.1.0`; adds the audio path (`workload/audio`): the stream's audio track decoded to 16 kHz inside the enclave and classified into non-speech vocalization categories by the CED tagger built into the image (`libced.so` from a pinned ced.cpp commit, `ced-small-f16.gguf` at a pinned revision and SHA-256), with level and pitch; the labels cross the local socket to the analysis bundle `2026.09.09-2` (`analysis.lock`, SHA-256 `e1253179…`), which consumes them. Frames and audio stay in the public code. `tee-verify -allow-debug` with `-cosign` passed every check on the pinned slot 2026-09-09 19:25 UTC; the boot log shows the bundle fetched, checked and started under its own user, and the release's smoke test loaded the classifier (`ced.cpp-abi1:ced-small-f16.gguf`, 527 labels). (`v0.2.0` was tagged but its build failed before any image was pushed; no digest carries it.) |
| `sha256:8b4a90f9b1442c1f00866b0c120babb86672f70c01563dced695d862a01f39d7` | 2026-09-09 | **`v0.2.2`**: tag [`v0.2.2`](https://github.com/FemLed/masseuse-video-tee/releases/tag/v0.2.2), `release.yml` run `34403106483`, provenance and keyless signature on `ghcr.io/femled/masseuse-video-tee@sha256:8b4a90f9…`, promoted and KMS-signed the same way. Same posture, base and analysis bundle (`2026.09.09-2`) as `v0.2.1`; the audio stage now reports the level of each window it scores (`audioDbfs` in the telemetry line, -90 being digital silence) and relays its ffmpeg's description of the track and its warnings into the log, after two `v0.2.1` sessions measured a phone's track as silence that a third did not (the cause is open; the phone's publisher now also checks its own microphone clone is heard). The trainer verified this digest on the pinned slot in a phone session 2026-09-09 21:07 UTC, in which the stage heard the room at -61 dBFS and a played sound at -20 dBFS, and the analysis bundle reported a vocalization event from it; `slsa-verifier verify-image` and `cosign verify` passed on the `ghcr.io` digest in the release's own `promote` job before the copy. |

Retired: the debug images from the first days of the enclave (2026-09-08
and 2026-09-09), built by Cloud Build from a tree that was not yet public
and carrying no provenance: `sha256:efe3d2b7…`, `sha256:3c3fa7f0…`,
`sha256:80df7800…`, `sha256:c95761f4…`, `sha256:4866753a…`,
`sha256:e7c80084…`.

Production digests are appended here with the flip to the production
Confidential Space image (`debug_mode = false`, `require_signed_image =
true`); a digest is removed when its image is retired, and the trainer's
policy (`/api/tee-policy`) mirrors this list, with `imageSources` naming
the release tag each digest was built from.

## How the image is built

From `v0.1.0` on, every published digest is built by GitHub Actions from a
tagged commit of this repository ([`.github/workflows/release.yml`](.github/workflows/release.yml)):
two Dockerfiles (an `ubuntu:24.04` base with Python, the pinned torch
2.14.0+cu130 set split into five layers and the model code; then the TEE
layer with Caddy and MediaMTX by digest, the ACME and attestation tooling
and the launch-policy labels), built with BuildKit through `buildx` so every
layer is zstd and the push is a single OCI manifest, which is what a digest
names. The workflow pushes to `ghcr.io/femled/masseuse-video-tee`, signs the
digest keyless with cosign (the certificate's identity is the workflow at
the tag), attaches SLSA provenance with the
[slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)
container generator, and then a separate job copies the digest, unchanged,
into the registry the attestation names and signs it there with the KMS key
below, so the launcher's signature check is what it was. To check a digest
against its source:

```sh
DIGEST=sha256:...   # from the attestation, the table above or /api/tee-policy
TAG=v0.2.2          # the release the table (or imageSources) names for it
slsa-verifier verify-image ghcr.io/femled/masseuse-video-tee@$DIGEST \
    --source-uri github.com/FemLed/masseuse-video-tee --source-tag $TAG
cosign verify ghcr.io/femled/masseuse-video-tee@$DIGEST \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp '^https://github.com/FemLed/masseuse-video-tee/.github/workflows/release.yml@refs/tags/v[0-9.]+$'
```

(`cosign` 3.x is needed: the signature is stored as a Sigstore bundle.
The digest on `ghcr.io` and in `us-central1-docker.pkg.dev` is the same
string, so the provenance verified on one is the provenance of the other.)
The trainer's `/api/tee-policy` carries the same mapping as `imageSources`,
which is what the home-camera connector reads to log the command for the
digest your session attested.

Digests before `v0.1.0` were built by Cloud Build in the enclave's project
from a source tree that was not yet public, with the same Dockerfiles but
no provenance; what they did is only as verifiable as this document.

Since `e7c80084…` the image carries one binary that is not built from the
workload tree: `masseuse-camlink-gateway`, the enclave half of the
home-network camera connector, from the public
[FemLed/masseuse-camlink](https://github.com/FemLed/masseuse-camlink)
repository. The TEE Dockerfile has a `golang` stage that runs `go install
github.com/FemLed/masseuse-camlink/cmd/masseuse-camlink-gateway@<tag>`
through the Go module proxy with the release's pinned toolchain and flags,
then `sha256sum -c` against the checksum in `camlink.lock`, which is the
`masseuse-camlink-gateway_<version>_linux_amd64` line of that release's
`checksums.txt` (signed keyless by its release workflow, covered by SLSA
provenance, and shown by its `reproduce` job to be what `go install`
yields; `VERIFY.md` there). A mismatch fails the image build, so the
gateway inside a published digest is byte for byte the published,
verifiable release, and `camlink.lock` in this repository says which one.

## Image signing key

Cosign signatures are made by a Cloud KMS `EC_SIGN_P256_SHA256` key
(`terraform/signing.tf`) that only the project's build identity may use;
every build signs its digest.

- key_id (hex SHA-256 of the DER public key, what the attestation reports
  in `submods.container.image_signatures[].key_id`):
  `cfb085b950e93abb8332cede62fa50df662ef9aebb1533b2ae0bf1403ea4f811`
- public key: `terraform output -raw image_signer_public_key_pem` in
  `terraform/`, reproduced here:

```
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEsZXAO7GfRoBojE2C+WgQX2ptlIlQ
FpCB6iIi0saIIRUf4khO033wXPEVc9NZj17GLaAXkPgtgNScFvem5HV6Jg==
-----END PUBLIC KEY-----
```

Check the fingerprint yourself: `openssl pkey -pubin -in signer.pub
-outform DER | openssl dgst -sha256`.

## What this does not prove

- That the source code is what it says, for the digests above: a digest
  pins an image, and tying the image to source needs a build attestation.
  That is what the provenance described under "How the image is built"
  adds for digests built from this repository.
- That the JavaScript your phone runs is the published one. It is served by
  FemLed through Cloudflare. A modified bundle could skip the checks above,
  which is why the checks are also documented for you to run from outside,
  and why the home-network connector, which is open source and reproducible,
  runs them itself.
- What the analysis module does with the readings. It consumes keypoints
  and descriptors, not frames; its logic is not published (`README.md`,
  "What happens to your video").
- Availability. Confidential Space GPU VMs are Spot instances that boot on
  demand (about three minutes after you tap Enable camera) and stop a
  minute or two after the session; while one boots, or when one is
  preempted, the app runs pose on the phone until the slot is back. There
  is no other server-side path.

## Residual trust

- Google: the hardware root of trust (TDX, the H100's attestation) and the
  Confidential Space launcher and attestation service.
- Google Trust Services (Cloud Public CA): issues the TLS certificate to
  the key in the enclave over TLS-ALPN-01 at every boot (chain: leaf →
  `WR1` → `GTS Root R1`). The ACME account is registered with an External
  Account Binding the enclave mints for itself with its attested identity;
  only a digest-pinned image on TDX with the GPU in CC mode may mint one.
  The CAA record on `tee.masseuse.ai` (`0 issue "pki.goog"`) allows no other
  CA. Each boot's certificate is public in the CT logs: a timestamp per
  session start, no identity.
- FemLed: the policy (which digests are allowed) and the JavaScript, as
  above. The published digest list and this verifier are how that trust is
  checked rather than assumed.
