# Verifying a masseuse.ai video slot

masseuse.ai's camera pipeline runs in a Google Cloud Confidential Space
enclave: an `a3-highgpu-1g` VM with Intel TDX on the CPU and an NVIDIA H100
in confidential-computing mode. This document is how anyone, without an
account at FemLed or Google, checks that the enclave your phone (or your
home camera, through [masseuse-camlink](https://github.com/Masseuse-ai/masseuse-camlink))
would send video to is the one described here, and what that does and does
not prove. If you are not technical, hand it to someone who is.

## What is being claimed

1. Your camera's video is decrypted in exactly two places: your device and
   the enclave. Cloudflare, the trainer service (Cloud Run), the TURN relay,
   the home-network connector and FemLed's operators see ciphertext or
   nothing.
2. The enclave runs an image that the release workflow of this public
   repository built from a tagged commit and signed, on a production
   Confidential Space VM image with debugging disabled since boot (no SSH,
   no log redirection, no memory monitoring: that is what `dbgstat`
   attests, of the VM image the launcher runs, not of the container),
   on hardware Google attests to, with the GPU attestation-bound. The
   attestation names the image's digest, the release it was built from and
   the signature.
3. TLS for the signalling origin (`slot-N.tee.masseuse.ai`) terminates
   inside the enclave: the certificate's key was generated there and is
   named in the attestation.
4. The WebRTC media itself (DTLS-SRTP) terminates inside the enclave: every
   SDP answer's DTLS fingerprint is signed by an Ed25519 key generated in
   the enclave and named in the attestation, and your browser's own DTLS
   handshake then verifies that fingerprint.
5. An image this repository's release workflow did not build cannot take
   a session, and a released image runs only the weights and the analysis
   bundle its release names. FemLed's operators own the enclave's Google
   Cloud project, and owning it they can sign an image with the key the
   launcher checks, boot it, give it the weights, mint a certificate for a
   slot's name, and set the variables the launch policy leaves to the
   deployment. That is not where the clients look for the image's
   identity: the trainer and the camera connector hold every slot to the
   keyless signature and the SLSA provenance the release workflow leaves
   for the digest in the public transparency log, at the tag and commit
   the image is stamped with, which no one holding the project can make,
   and the phone sends nothing to a slot the trainer has not vouched for
   that way. The trainer's ID token goes only over the TLS key the
   attestation binds, and the phone's requests are sealed to a key the
   attestation names, so a certificate for the name opens nothing.
   `models.lock` and `analysis.lock`, part of the image, fix by SHA-256
   what it loads, whatever the bucket holds.
6. What the enclave sends onward is derived readings (numbers), never
   frames: `README.md`, "What happens to your video".

## What identifies the image

Not a list of digests. A digest is known only after a build, so any list
kept in a repository is written after the tag that produced the image and
is always a release behind what runs: `main` could never both describe the
running image and be the tree it was built from. The identity that holds
at every moment is in the attestation token itself:

- `submods.container.image_signatures[].key_id`: the fingerprint of the KMS
  signing key (below). The launcher verified the signature before it
  started the image. Terraform grants the key to the release workflow
  alone (`terraform/github-release.tf`, `signing.tf`: a run of
  `.github/workflows/release.yml` on a `v*` tag of this repository, on a
  GitHub-hosted runner), but the project's owners can grant it to
  themselves, so the signature says the image went through the enclave's
  project, not who built it.
- `submods.container.env.TEE_IMAGE_VERSION` and `TEE_IMAGE_COMMIT`: the
  release tag and the commit the workflow baked into the image
  (`workload/tee/Dockerfile.tee`), attested with the rest of the workload
  environment. Images released before `v0.4.0` carry no stamp.
- `submods.container.image_digest`: the digest that is running. The same
  digest sits on the public registry its Release names (`ghcr.io/masseuse-ai/masseuse-video-tee`
  from `v0.12.4` on, `ghcr.io/femled/masseuse-video-tee` for images up to `v0.12.3`)
  with its SLSA provenance, which names the source commit; that commit must be the one
  the stamp names, and `slsa-verifier` checks the tag.
- The release workflow's keyless signature over the digest and the SLSA
  generator's provenance for it, both in the Sigstore transparency log:
  certificates that name `release.yml` of this repository at the tag (and
  the generator, for the provenance), a GitHub-hosted run, and the commit.
  This is what says who built the image, and it is not the enclave
  project's to write. The trainer and the connector check it for every
  slot, and `tee-verify` does.

The clients pin the signing key and a minimum release; they do not pin
digests. The trainer and the connector check the digest's public record
themselves (below), and the web app, which cannot read the transparency
log, requires the trainer's verdict on it (`requireProvenance` in the
policy). The connector goes further: the signing key, the minimum release,
the project and registry the enclave must run from and this repository as
the source are compiled into it as floors the served policy can only
tighten.
Every release's digest and validation record is on its GitHub Release
([releases](https://github.com/Masseuse-ai/masseuse-video-tee/releases)), written
by the workflow that built it and appended to by the operators when the
image has been verified on a slot and carried a session.

## What you need

- `go` 1.22+ to build the verifier in `verifier/`.
- [`slsa-verifier`](https://github.com/slsa-framework/slsa-verifier) on
  your `PATH` (or `-slsa-verifier <path>`) to tie the running digest to
  this source: a run without it does not pass. Optionally `cosign` to check
  the image signature independently.
- The signing key fingerprint (below) is the default of `-signer-key-ids`;
  pass it explicitly if you would rather not take the verifier's word.

## Run the verifier

```sh
cd verifier
go build -o tee-verify .
./tee-verify -origin https://slot-0.tee.masseuse.ai
```

Exit code 0 and `"passed": true` means every check below held for a token
minted seconds ago for your nonce. The report names the digest and the
release (`release.version`, `release.commit`) that are running. Add
`-expect-release vX.Y.Z` to insist on a particular release (the newest one
on the releases page, say) or `-min-release vX.Y.Z` for a floor. A run
that cannot make the signature check (no accepted key) or the provenance
check (no `slsa-verifier`, an image released before `v0.4.0` with no
stamp to hold the provenance to, a provenance naming no commit) does not
pass. Add `-v` to print the decoded attestation claims; add `-cosign
<path> -cosign-public-key signer.pub` to have cosign re-verify the image
signature from the registry. `-allowed-digests` pins the slot to a list if
you have a reason to; nothing here requires it.

Each check, and why it matters:

| check | meaning |
| --- | --- |
| `jwt.signature`, `jwt.issuer` | the token is signed by Google's Confidential Space attestation service (`https://confidentialcomputing.googleapis.com`), whose keys are at `https://www.googleapis.com/service_accounts/v1/metadata/jwk/signer@confidentialspace-sign.iam.gserviceaccount.com`. Only the launcher inside a real Confidential Space VM can obtain one. |
| `jwt.audience` | the token was minted for this slot's origin, not lifted from another service. |
| `cs.swname`, `cs.hwmodel`, `cs.secboot` | Confidential Space on Intel TDX with Secure Boot. |
| `cs.dbgstat`, `cs.support_attributes.STABLE` | the production VM image family, debugging disabled since boot: the operator cannot SSH in, redirect the container's output, or read memory metrics. It is the VM image's posture; what the container does is the image's, below. (`-allow-debug` accepts the debug posture; production verification must not pass it.) |
| `gpu.cc_mode`, `gpu.hwmodel` | the H100 is in confidential-computing mode and its device attestation is part of the token: the models run on protected GPU memory, not a plain GPU next to a TDX CPU. |
| `image.signature` | the image carries a cosign signature by the KMS key whose fingerprint is published below (`-signer-key-ids`, that key by default); the launcher verified it at boot. Rebuilding the image with different code changes the digest and voids it. Who can make the signature is whoever the enclave project's IAM lets use the key: the release workflow on a tag, and the project's owners if they grant themselves the role. It says the image went through the project; `provenance.source` says who built it. |
| `image.release` | the release stamp (`TEE_IMAGE_VERSION`, `TEE_IMAGE_COMMIT`) the workflow baked in: which release is running, held to `-expect-release` / `-min-release` when given. |
| `image.digest`, `image.reference` | the digest of the running container, reported (and pinned to `-allowed-digests` if you passed a list), pulled from FemLed's registry for this project. |
| `image.env.TRAINER_URL` | the enclave posts its readings (numbers, never frames) to the trainer service you expect and nowhere else (the whole workload environment is in the token; `-v` shows it). |
| `image.env.TEE_CAPTURE_BUCKET` | the one bucket a signed-in session's record (keypoints, descriptors, audio measurements, the analysis's rows; README "Session records") may be written to is the one you expect (`-capture-bucket`, default `masseuse-ai-prod`); the enclave refuses a lease naming a record when the variable is unset, and `-capture-bucket ""` requires a slot that keeps none. The trainer holds the same expectation before it leases. |
| `image.env.TEE_REQUEST_SEAL` | `v1` (`-request-seal`): requests under the phone's capability are sealed to a key the enclave attests, and a bare bearer is refused. Not settable at boot; checked because an image without it takes the capability in the clear. |
| `image.env.TRAINER_INVOKER_SERVICE_ACCOUNT` | exactly the trainer's runtime account (`-trainer-invoker`), read as the enclave reads it: the one identity the slot's control routes (`/lease`, `/produce`, ...) take an ID token from. The deployment can set it at boot, so an extra identity here is an extra caller that could re-lease the slot. The trainer and the phone hold it to the policy's `expectedTrainerInvokers` too. |
| `image.env.MODELS_BUCKET`, `WIF_AUDIENCE`, `ACME_DIRECTORY_URL`, `TEE_ALLOWED_ORIGINS` | reported, not checked: the deployment may set them at boot. What each can change: which bucket the weights come from (`models.lock` fixes what loads, by SHA-256), which identity pool the attestation is exchanged at, which CA issues the certificate (the TLS key is attested whatever the CA), which page origins the slot answers, `*` among them meaning any page's (the deployment's since v0.19.0; every request under the capability is sealed regardless). |
| `provenance.source` | the running digest, on the public registry its release names (`-image-repo`), carries SLSA provenance from this project's release workflow (`-source-uri`) at the release the stamp names, and the provenance's commit is the stamp's commit: the image is the public source at that tag, built by GitHub's runners. Required: without `slsa-verifier`, a stamp, or a commit in the provenance, the run fails. |
| `nonce.fresh` | `eat_nonce` contains the random nonce this run sent: the token was minted now, not replayed. |
| `nonce.evidence-key` | `eat_nonce` contains `sha256(evidence key)`: the Ed25519 key that signs DTLS fingerprints lives in this enclave. |
| `nonce.tls-spki` | `eat_nonce` contains `sha256(SubjectPublicKeyInfo)` of the certificate this very TLS connection negotiated: the TLS endpoint is the enclave, not a proxy in front of it. |
| `nonce.exact` | `eat_nonce` holds those three and nothing else (the detail lists it as the token has it). The enclave appends one caller's nonce to its own two and refuses a caller's nonce shaped like them (43 characters of base64url, a SHA-256), so a token can never name somebody else's key beside the enclave's. |
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

The token's `eat_nonce` must be exactly the two hashes and your `$NONCE`:
the evidence key's first, the TLS key's second, yours last. Verify the
token's RS256 signature against the JWKS URL above with any JWT library.
In `submods.container`, `image_signatures[].key_id` must be the
fingerprint below, and `env.TEE_IMAGE_VERSION` says which release is
running; take that and `image_digest` to "How the image is built" below
to tie them to this source.

A slot exists only while a session needs it (it boots when a visitor taps
Enable camera and stops a minute or two after the session), so a
connection refused or a DNS name that resolves to a silent address means
no slot is up, not that a check failed. Verify during a session of your
own, or start one and run the verifier from another machine.

## What the clients check

The masseuse.ai trainer, which leases a slot to a phone and drives it with
its own Google ID token, checks the slot before anything with that token
reaches it (`masseuse-trainer/server/`, "Attested before leased" in its
README). It fetches the slot's `/attestation` with no token, over a
connection of its own, and requires the token to bind the TLS key that
very connection presented; it holds the claims to its policy (the same
document the browser fetches, below); and it checks the image's public
record the way the connector does (`tee-provenance.js`, a port of the
connector's `internal/provenance`): the release workflow's keyless
signature over the attested digest at the attested tag and commit from a
GitHub-hosted run, and the SLSA generator's provenance naming this
repository at that tag and commit, both verified against Sigstore's trust
root and the transparency log, with the certificates matched exactly.
Only then does it pin every connection to the slot to the attested TLS
key, checked in the handshake, so its token reaches the enclave that was
verified and nobody holding a certificate for the slot's name. It reports
what it found to the phone (`attested.provenance`), which requires it.

The masseuse.ai web app does the same verification in the browser before it
sends any media: JWT signature against the same JWKS, the same claims
against the policy the trainer publishes at
`https://masseuse.ai/api/tee-policy` (the signing key it accepts, the
minimum release once one is set, the source repository and public image
repository the provenance must name, the debug and GPU posture, exactly
the trainer as the slot's control-route caller), and the evidence-key
binding; and it refuses a slot the trainer has not described (the same
instance, image and evidence key it verified itself) or, under
`requireProvenance`, whose provenance the trainer has not verified. The
browser asks for no nonce of its own, so it takes
the slot's boot token only: `eat_nonce` must be the evidence key's hash,
then the TLS key's once the slot has a certificate, and nothing else; a
token minted for anyone's nonce is refused. Then, on every WebRTC leg, it
verifies the enclave's Ed25519 signature over the SDP answer's DTLS
fingerprint, the session, the leg and its own nonce
(`X-Masseuse-Evidence`), and refuses the connection if it does not hold.
The browser cannot read the TLS certificate it negotiated, so the
`nonce.tls-spki` check is the verifier's job; the browser's DTLS
fingerprint check gives the media the same property. For the same reason
an image that attests `TEE_REQUEST_SEAL=v1` takes nothing under the
phone's capability in the clear: the browser holds the seal key the
attestation document names to the evidence key's statement for it, and
seals every such request to that key, its body with it (README "Sealed
requests"). A bare capability is refused by the enclave, so a certificate
for the slot's hostname yields neither the capability nor a camera's link
or a stream key.

The home-network connector (`masseuse-camlink`) runs the same checks
before it dials a slot (`internal/attest` there is a port of `verifier/`)
and additionally pins the slot's TLS key to the SPKI hash in the token. It
does not take the served policy at its word: the token issuer and key set,
the signing key, the minimum release, this repository and its public
registry, the Google Cloud project and the registry path the enclave must
have been pulled from, and the `.tee.masseuse.ai` suffix are floors
compiled into the connector (`internal/attest/floors.go`), and a served
policy that contradicts them is refused rather than applied. Then, before
the first frame leaves the machine, it does what the two commands above do
(`internal/provenance`, on sigstore-go): it reads from `ghcr.io` the
Sigstore bundle and the SLSA provenance attached to the attested digest and
verifies against the public Sigstore trust root that the signature's
certificate is this repository's `release.yml` at the very tag the image is
stamped with, and that the provenance, signed by the SLSA GitHub generator,
names this repository at that tag and commit. An image without both records
in the transparency log is refused. It logs the `slsa-verifier` and
`cosign` commands for the digest and release your session attested, and the
Rekor indexes of the entries it verified, so you can repeat the check.

## Release history

Every release since `v0.1.0` has a GitHub Release at
[github.com/Masseuse-ai/masseuse-video-tee/releases](https://github.com/Masseuse-ai/masseuse-video-tee/releases),
created by the release workflow run that built it. Its body says what the
release changed, names the image digest (the same on `ghcr.io` and in the
enclave's registry) and the base image digest, and gives the commands that
tie the digest to the tag and to the identity that signed it. That record
is the history this document used to hold as a table of digests, and it is
written by the release, not ahead of it.

Releases up to and including `v0.12.3` were built and signed by this
release workflow at the project's previous GitHub home,
`github.com/FemLed/masseuse-video-tee`, and published to
`ghcr.io/femled/masseuse-video-tee` (the base image to
`ghcr.io/femled/masseuse-video-tee-base`), where they stay; their
provenance and certificates name that identity, and this repository
carries each of their tags with the tree the image was built from. From
`v0.12.4` on, images are built and signed here and published to
`ghcr.io/masseuse-ai/masseuse-video-tee`. Each Release's Image and Verify
sections name the registry and identity that apply to it; `tee-verify`'s
`-source-uri` and `-image-repo` default to this repository and its
registry and are set on the command line for an image up to `v0.12.3`.
A tag, once created, cannot be moved or deleted by anyone: a
repository ruleset ("Tags are immutable", on every tag, with no bypass
actor) refuses updates, deletions and force pushes, so the tag a
provenance names is still the commit it named when the image was built.

Images before `v0.1.0` were built by Cloud Build in the enclave's project
from a source tree that was not yet public, with the same Dockerfiles but
no provenance, all in the debug posture; they were retired on 2026-09-09.
They were signed by the same key by the build system of the time, so the
signature alone does not exclude them: the production posture does
(`dbgstat`), the deployment's digest pin keeps the weights from them, and
so do the minimum release the clients pin and the provenance the trainer
and the connector check, since none of them carries a stamp or a
provenance.

Production posture since 2026-09-09 22:20 UTC (`v0.2.3`), except for two
debug windows on 2026-09-10 recorded on the `v0.3.0` and `v0.3.1`
Releases: the slot boots the STABLE `confidential-space` image with
debugging disabled since boot (`debug_mode = false`; `cs.dbgstat` =
`disabled-since-boot`, `cs.support_attributes` = `[LATEST STABLE USABLE]`),
the trainer's policy refuses a debug image (`allowDebug: false`,
`requireStable: true`), and the container's output is not redirected
anywhere (`tee.launch_policy.log_redirect` = `debugonly`): what the enclave
does is observable only through the attestation, the readings and this
repository.

## How the image is built

From `v0.1.0` on, every image is built by GitHub Actions from a tagged
commit of this repository ([`.github/workflows/release.yml`](.github/workflows/release.yml)):
two Dockerfiles (an `ubuntu:24.04` base with Python, the pinned torch
2.14.0+cu130 set split into five layers and the model code; then the TEE
layer with Caddy and MediaMTX by digest, the two audio taggers - ced.cpp
built at a pinned commit with its weights, and the BEATs checkpoint, each
file fetched at a pinned revision and checked against its SHA-256 - the
ACME and attestation tooling, the release stamp and the launch-policy
labels), built with BuildKit
through `buildx` so every layer is zstd and the push is a single OCI
manifest, which is what a digest names. The workflow pushes to
`ghcr.io/masseuse-ai/masseuse-video-tee`, signs the digest keyless with cosign
(the certificate's identity is the workflow at the tag), attaches SLSA
provenance with the
[slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)
container generator, and then a separate job copies the digest, unchanged,
into the registry the attestation names and signs it there with the KMS key
below, so the launcher's signature check is what it was; a last job writes
the GitHub Release. To check the digest a token names against its source
(as written, for an image released from `v0.12.4` on; for an image up to
`v0.12.3` read `ghcr.io/femled/masseuse-video-tee` and
`github.com/FemLed/masseuse-video-tee`, "Release history"):

```sh
DIGEST=sha256:...   # submods.container.image_digest in the token (tee-verify prints it)
TAG=vX.Y.Z          # submods.container.env.TEE_IMAGE_VERSION in the same token
slsa-verifier verify-image ghcr.io/masseuse-ai/masseuse-video-tee@$DIGEST \
    --source-uri github.com/Masseuse-ai/masseuse-video-tee --source-tag $TAG --print-provenance
cosign verify ghcr.io/masseuse-ai/masseuse-video-tee@$DIGEST \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --certificate-identity-regexp '^https://github.com/Masseuse-ai/masseuse-video-tee/.github/workflows/release.yml@refs/tags/v[0-9.]+$'
```

The printed provenance's source commit must be the token's
`TEE_IMAGE_COMMIT` (that is what `tee-verify -slsa-verifier` compares).
For an image released before `v0.4.0`, which carries no stamp, take the
tag from the Release whose body names the digest and leave `--source-tag`
off to verify only the repository. (`cosign` 3.x is needed: the signature
is stored as a Sigstore bundle. The digest on `ghcr.io` and in
`us-central1-docker.pkg.dev` is the same string, so the provenance verified
on one is the provenance of the other.)

Since `v0.1.0` the image carries one binary that is not built from the
workload tree: `masseuse-camlink-gateway`, the enclave half of the
home-network camera connector, from the public
[Masseuse-ai/masseuse-camlink](https://github.com/Masseuse-ai/masseuse-camlink)
repository. The TEE Dockerfile has a `golang` stage that runs `go install
github.com/Masseuse-ai/masseuse-camlink/cmd/masseuse-camlink-gateway@<tag>`
through the Go module proxy with the release's pinned toolchain and flags,
then `sha256sum -c` against the checksum in `camlink.lock`, which is the
`masseuse-camlink-gateway_<version>_linux_amd64` line of that release's
`checksums.txt` (signed keyless by its release workflow, covered by SLSA
provenance, and shown by its `reproduce` job to be what `go install`
yields; `VERIFY.md` there). A mismatch fails the image build, so the
gateway inside a released image is byte for byte the published,
verifiable release, and `camlink.lock` at the tag says which one.

## Image signing key

Cosign signatures are made by a Cloud KMS `EC_SIGN_P256_SHA256` key
(`terraform/signing.tf`) that terraform grants to the release workflow's
identity alone (`terraform/github-release.tf`: a Workload Identity
Federation pool that admits runs of `release.yml` on `v*` tags of this
repository, by name and by GitHub's ids, on GitHub-hosted runners, and
nothing else); every release signs its digest. The project's owners can
change that grant, and every use of the key is in the project's Data
Access audit log; the clients do not rely on either, they check the
keyless signature and the provenance as well.

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
-outform DER | openssl dgst -sha256`. The key is the one stable identity
in the chain: a new key would be a new fingerprint here, in the trainer's
policy and in the WIF provider, all in this repository's history.

## What this does not prove

- That the source is what it says, for images released before `v0.1.0`:
  they carry no provenance. For every release since, the provenance
  (`provenance.source`) is the tie between the digest and a commit of this
  repository, and the stamp (`image.release`) is the image's own statement
  of which; the two must agree.
- That the JavaScript your phone runs is the published one. It is served by
  FemLed through Cloudflare. A modified bundle could skip the checks above,
  which is why the checks are also documented for you to run from outside,
  and why the home-network connector, which is open source and reproducible,
  runs them itself, with the policy's anchors compiled in and the digest's
  provenance verified against the public registry and the transparency log.
- What the analysis module does with the readings. It consumes keypoints
  and descriptors, not frames; its logic is not published (`README.md`,
  "What happens to your video").
- That the launcher refused anything. It does not refuse an unsigned image:
  it logs the miss and mints a token with no `image_signatures`, which the
  clients and this verifier then refuse (`docs/OPERATIONS.md`, "Image
  signing").
- That nothing touches a disk. The VM has its boot disk and the container
  its writable layer; the workload keeps the weights, the TLS key, the
  credential and the session's capture rows on two tmpfs mounts (`/models`,
  `/run/tee`) and writes nothing of a session anywhere else, which is the
  image's code, not something the token attests.
- A camera the user names. Its certificate is pinned on first use, from
  the enclave's network: the fingerprint is whatever answered the enclave's
  first probe, and for a camera behind the home connector that probe goes
  through the connector, which is trusted not to answer it itself and
  substitute a camera.
- Anything this tool cannot see from outside: the variables it reports
  rather than checks (above), the policy and the script the phone is
  served, and whether the trainer's verdict on provenance, which the phone
  relies on, is the one it computed.
- Availability. Confidential Space GPU VMs are Spot instances that boot on
  demand (about three minutes after you tap Enable camera) and stop a
  minute or two after the session; while one boots, or when one is
  preempted, the app runs pose on the phone until the slot is back. There
  is no other server-side path.

## Residual trust

- Google: the hardware root of trust (TDX, the H100's attestation) and the
  Confidential Space launcher and attestation service.
- GitHub: the runners that build every release, the OIDC identity the
  signing key is granted to, and the Sigstore provenance chain. Sigstore's
  transparency log records every keyless signature the workflow makes.
- Google Trust Services (Cloud Public CA): issues the TLS certificate to
  the key in the enclave over TLS-ALPN-01 at every boot (chain: leaf →
  `WR1` → `GTS Root R1`). The ACME account is registered with an External
  Account Binding the enclave mints for itself with its attested identity;
  of the slots, only a signed image on TDX with the GPU in CC mode may mint
  one. The project's owners can grant themselves the same and get a
  certificate for a slot's name; that is why the trainer pins the attested
  TLS key, the phone seals its requests to an attested key, and the DTLS
  fingerprints are signed by the evidence key. The CAA record on
  `tee.masseuse.ai` (`0 issue "pki.goog"`) allows no other CA. Each boot's
  certificate is public in the CT logs: a timestamp per session start, no
  identity.
- FemLed: the policy (which signing key and minimum release the clients
  accept), the JavaScript, and the trainer's verdict on each slot's
  provenance, which the phone relies on; and, as owners of the enclave's
  project, the signing key, the weights bucket, the identity pools, the
  CA account and the variables set at boot. The provenance, the locks in
  the image, the TLS pin, the seal, the Releases and this verifier are how
  that trust is checked rather than assumed. For the home-network connector
  this residual is smaller: the policy can only tighten anchors fixed in
  its reproducible build, and it verifies each digest's signature and
  provenance against the public registry and the transparency log itself,
  so a served policy or a served script cannot send it to an image this
  repository's release workflow did not build.
