# Operations

How the enclave project is stood up and run. `README.md` says what it is;
`VERIFY.md` how anyone checks it from outside. Everything here is the
operator's side of the same boundary: none of it grants access to what runs
inside a slot.

## Prerequisites

- `gcloud` authenticated as an owner of `prod-masseuse-video-tee`
  (`operator_members` in `terraform.tfvars`).
- Terraform >= 1.6.
- DNS for `slot-N.tee.masseuse.ai` (A records, DNS-only, and a CAA record
  pinning `tee.masseuse.ai` to Google Trust Services, `pki.goog`) is managed
  with the masseuse.ai zone, outside this repository.

## 1. Infrastructure

```sh
cd terraform
cp terraform.tfvars.example terraform.tfvars   # placeholder digest is fine for now
terraform init
terraform apply -target=google_compute_address.slot -target=google_artifact_registry_repository.images \
    -target=google_iam_workload_identity_pool_provider.attestation_prod \
    -target=google_iam_workload_identity_pool_provider.attestation_debug \
    -target=google_storage_bucket.models
terraform apply    # everything except a running VM (vm_running = false)
```

The state bucket `gs://prod-masseuse-video-tee-tf-state` was created by
hand (`versions.tf`). `terraform output slot_hosts` prints the static IP per
hostname.

## 2. The image

Every image is built by `.github/workflows/release.yml` from a tag:

```sh
git tag vX.Y.Z && git push origin vX.Y.Z   # through the release identity
```

The workflow builds the base (`workload/Dockerfile`) and the TEE layer
(`workload/tee/Dockerfile.tee`) with BuildKit, every layer zstd, one OCI
manifest per image, pushes them to `ghcr.io/femled/masseuse-video-tee`
(and `-base`), runs the smoke test on the pushed image, signs it keyless,
attaches SLSA provenance (the container generator, an isolated job), and
then promotes: `crane copy` by digest into this project's Artifact Registry
and `cosign sign --key gcpkms://...` with the key in `signing.tf`, in the
layout the launcher reads. The promote job authenticates as the
`github-release` service account through the GitHub OIDC pool in
`terraform/github-release.tf`, which admits only tag refs of this
repository. Its step summary prints the `terraform.tfvars` and `tee_policy`
lines for the roll; the digest is identical on ghcr.io and in Artifact
Registry (`VERIFY.md`, "How the image is built").

`bash build.sh` runs the same two builds locally, without a push (gzip layers,
so a different digest), for iterating on the tree.

Put the digest into `terraform.tfvars` (`container_image` and
`container_image_digest`) and into the trainer's `tee_policy`
(`allowed_image_digests`, and `image_sources` with the tag so the policy
says where it was built). While rolling from one digest to another, list
the new one in `candidate_image_digests` first so its VM can read the
weights on its first boot.

## 3. The weights

The models bucket (`terraform output models_bucket`) holds the detector and
keypoint model weights under `detectors/` and `hf-bf16/`, and the pinned
analysis bundle under `analysis/`. The enclave copies them into the
`/models` tmpfs at boot with the credential its attestation earns (about
3.3 GB; the hub `blobs/` and `.locks/` directories the boot never reads are
skipped). Weights are copied in by an operator once; only an attested image
digest can read them afterwards.

## 4. Boot

```sh
cd terraform
terraform apply -var vm_running=true
```

Under three minutes later the launcher starts the container. Watch:

```sh
gcloud compute instances get-serial-port-output masseuse-video-tee-slot-0 \
    --zone us-central1-a --project prod-masseuse-video-tee | grep -E 'entrypoint|tee:|tee-models|caddy|launcher'
gcloud logging read 'resource.type="gce_instance"' --project prod-masseuse-video-tee --limit 50
```

Expected order: caddy up → mediamtx advertising the static IP → weights
copied (`tee-models: done`) → `tee: serving https://slot-0.tee.masseuse.ai`
→ `tee: attestation token refreshed ... tlsNonce=yes` once the certificate
lands. (On the production image only the launcher's own log reaches Cloud
Logging; the container's output does not leave the enclave.)

## 5. Verify

```sh
HOST=slot-0.tee.masseuse.ai
curl -sS https://$HOST/healthz
curl -sS https://$HOST/evidence-key | jq
curl -sS "https://$HOST/attestation?nonce=$(head -c 24 /dev/urandom | base64 | tr '+/' '-_' | tr -d =)" | jq .token -r | cut -d. -f2 | base64 -d 2>/dev/null | jq
openssl s_client -connect $HOST:443 -servername $HOST </dev/null 2>/dev/null | openssl x509 -pubkey -noout | openssl pkey -pubin -outform DER | openssl dgst -sha256 -binary | base64 | tr '+/' '-_' | tr -d =
```

The last line is the TLS SPKI hash; it must equal the second entry of the
attestation token's `eat_nonce`, and the first entry must equal
`sha256(evidence-key.publicKey)`. `verifier/` automates all of it
(`VERIFY.md`).

Then the trainer: with the slot's origin in its slot list and its policy
listing the digest, a phone session on masseuse.ai leases the slot,
verifies the attestation, publishes over WHIP and renders the overlay over
WHEP. `GET https://masseuse.ai/api/tee-policy` shows what the clients
enforce.

## On demand: boot on tap, stop after the session

An `a3-highgpu-1g` Spot VM costs real money every hour it is up, so a slot
exists only while a session needs it. Nothing in this project keeps it
running between sessions.

- **Start.** When a masseuse.ai visitor taps *Enable camera*, the page asks
  the trainer to wake a slot before it even asks for the camera permission;
  the trainer calls `instances.start` on the first free slot. The trainer's
  runtime service account is the one identity that may do this: a custom
  role with `compute.instances.get/start/stop` bound *per instance*
  (`terraform/trainer-iam.tf`, `masseuseVideoTeeSlotOperator`) plus `actAs`
  on the VM's own service account, which starting a VM that runs as one
  requires. It holds no other permission in this project. During the boot
  the phone runs an on-device skeleton. There is no Cloud Run fallback:
  video is processed in the attested enclave or on the phone, nowhere else.
- **Stop, part one: the enclave decides.** The producer's idle watchdog
  exits 0 when no lease is held and nothing is working for `TEE_IDLE_EXIT_S`
  (60 s; a page refresh reconnects inside that), or `TEE_BOOT_IDLE_S`
  (300 s) after a boot that never gets a lease at all. The trainer's
  `/teardown` on session release drains for the same 60 s. Either way the
  container ends, Caddy and MediaMTX with it, and the slot stops answering.
  `/warmup` polls count as interest while a lease could still follow; they
  never keep an idle slot alive past `TEE_BOOT_IDLE_S`.
- **Stop, part two: the VM.** What happens next is the launcher's
  (go-tpm-tools `launcher/launcher/main.go` `getExitCode` and the image's
  `exit_script.sh`). The **debug image always holds** the VM after any
  exit ("VM remains running"). The **production image** powers off two
  minutes after a clean exit under `Never` or `OnFailure`, and after a
  crash under `Never`; but a crash under `OnFailure`, or any exit under
  `Always`, is `shutdown --reboot +2`: a *guest reboot*, which breaks the
  H100's attestation binding until a full stop/start (the rule below). So
  production runs `Never`, and the trainer's reaper does the actual
  stopping on both images: for an unleased slot it polls the VM's state
  and, while the VM is RUNNING past its boot grace, probes a control route
  that does not count as interest to the idle watchdog; 90 s of silence
  means the enclave exited or the slot is wedged, and it calls
  `instances.stop` with `discardLocalSsd`. It never stops an enclave that
  answers and never touches a leased slot.
- **Backstop.** The sweeper (below) stops any slot RUNNING longer than
  `slot_max_uptime_hours` (7), in case all of the above fail.
- **Terraform and the power state.** `vm.tf` has `ignore_changes =
  [desired_status]`: the trainer's starts and the enclave's stops are not
  drift, and `vm_running` only sets the state at create. By hand:
  `gcloud compute instances start masseuse-video-tee-slot-0 --zone us-central1-a`
  and `gcloud compute instances stop ... --discard-local-ssd=true` (the a3
  has local NVMe and a plain stop is refused; the google provider never
  sends `discardLocalSsd`, hashicorp/terraform-provider-google#26173).
- **Preemption.** A Spot preemption leaves the VM `TERMINATED`
  (`instance_termination_action = STOP`). Mid-session the trainer starts it
  again and the same session waits on-device through the reboot, then
  attests the new boot afresh. A preemption between sessions costs nothing.
- **Bounds on starting.** Every start is a multi-minute A3 boot billed from
  its first second, so the trainer allows a few starts per slot per
  interval and a daily cap across the pool; a start the API accepts that
  leaves the VM `TERMINATED` shortly after is a Spot stockout, and after a
  few of those the slot is benched briefly and the session shows the busy
  copy.
- **Cost floor per session.** About three minutes of boot, the session,
  60 s of grace, the reaper's ~90-105 s to call the silence final and the
  ~45 s stop, all Spot.

## Boot time

Start request to a leased slot is about **2 min 50 s** when Spot places the
VM at once. The launcher logs each stage with a timestamp (Cloud Logging,
`confidential-space-launcher`), so the budget is known:

| from `instances.start` | stage | took |
|---|---|---|
| 0:00 | Compute finds a Spot H100 host and powers the TDX VM on | 34 s |
| 0:34 | firmware, COS kernel and systemd, measured boot, network; launcher "Boot completed" | 48 s |
| 1:22 | launcher reads the launch spec, starts `cos-gpu-installer` | 1 s |
| 1:23 | GPU driver: download, userspace install, `insmod` and persistenced | 39 s |
| 2:02 | the image pull, 2.8 GiB in 22 zstd layers | 25 s |
| 2:26 | GPU attestation binding measured, signature cache, first token, container created | 8 s |
| 2:35 | entrypoint: EAB minted (0.5 s), Caddy, MediaMTX and the producer serving | 1 s |
| 2:36 | 3.3 GB of weights GCS → tmpfs (about 760 MB/s), under the torch import that already runs | 4 s |
| 2:40 | weights loaded, CUDA context, model built | 8 s |
| 2:48 | trainer's warm-up poll sees `ready`, attests, leases | 3 s |

Half of it is the platform below the launcher (Spot placement, TDX VM
start, COS boot), and the part that varies between boots. A quarter is the
NVIDIA driver, installed on every boot because Confidential Space keeps
nothing between boots by design (the stateful partition is encrypted with
a per-boot key, so there is no driver cache and no image cache). The image
pull is the one part that is ours, and it is what the image layout is
for:

- Base `ubuntu:24.04` rather than a CUDA base image: torch loads the
  `nvidia-*` wheels it bundles, Ubuntu's ffmpeg does not link the toolkit,
  and `libcuda` / `nvidia-smi` come from the launcher's driver mount at
  `/usr/local/nvidia`.
- The torch set is downloaded once and partitioned greedily by size into
  five layers, because the launcher fetches layers concurrently but unpacks
  one at a time: five 500 MB layers fetch in parallel where one 3 GB layer
  was a single stream followed by a single-threaded unpack.
- Every layer is zstd (`compression=zstd,compression-level=3,force-compression=true`),
  which COS's containerd unpacks 2-3x faster than gzip.
  `--provenance=false --sbom=false` keeps the push a single OCI manifest,
  so `image_digest`, cosign and the WIF condition stay one digest.
- The boot disk is `pd-ssd`: containerd syncs each layer to disk as it
  commits it, and `pd-balanced` at 100 GB was the bottleneck for the unpack.

The enclave's own start overlaps what it can: the producer runs its
warm-up as soon as it serves rather than waiting for the trainer's first
poll, the entrypoint starts the producer before the weights have finished
copying (the torch import overlaps the copy), and the model build waits on
a `.complete` marker.

## The stop/start rule

Never *reboot* the guest (`sudo reboot`, `gcloud compute instances reset`).
On H100 + TDX the GPU's attestation binding is lost across a guest reboot
and every token afterwards fails GPU attestation until the VM is fully
stopped and started (the known issue on the Confidential Space deploy
page). `tee-restart-policy` is a trap here: `Always`, and `OnFailure` after a
crash, make the launcher exit 3 and the image run `shutdown --reboot +2`,
a guest reboot. Production therefore runs `Never` (clean exit and crash
both power off; the next tap does a real start), and the debug image holds
the VM after any exit for the trainer's reaper to stop. To recover a VM
with a broken binding: `gcloud compute instances stop
--discard-local-ssd=true` then `start`.

Launch-spec details that each end in "Workload completed" and a shutdown
two minutes later if wrong:

- `tee-mount`'s `size=` is plain bytes (`8589934592`), not `8G`.
- `tee.launch_policy.allow_mount_destinations` is colon-separated
  (`/models:/run/tee`); with a comma the launcher sees one path and refuses
  both mounts.
- `tee.launch_policy.monitoring_memory_allow` is deprecated in favour of
  `hardened_monitoring` / `debug_monitoring`; the image sets the new pair.
  Their values are `none`, `memoryonly` or `all`.
- The WIF `external_account` credential names no project, and
  `google-cloud-storage` will not build a client without one; the
  entrypoint exports `GOOGLE_CLOUD_PROJECT` from the metadata server.
- `/models` is its own tmpfs, so the weights are staged in
  `/models/.partial` and renamed into place; staging next to `/models` is a
  cross-device rename the kernel refuses (EXDEV).
- The container shares the VM's network namespace, so the producer binds
  `127.0.0.1:8080` and only Caddy (:443) and MediaMTX's ICE port (:8189)
  face the network. The firewall allows exactly those.
- `google_workflows_workflow` defaults to `deletion_protection = true` in
  the google provider; the sweeper sets it false.

## Debug conveniences (`debug_mode = true`)

Never for production; the debug WIF provider and the soak service account
exist only while this is on, and the clients' policy refuses the debug
posture.

- `confidential-space-debug` image: container stdout in Cloud Logging and
  on the serial console; memory metrics; SSH via IAP with OS Login
  (`gcloud compute ssh masseuse-video-tee-slot-0 --tunnel-through-iap`),
  then `sudo ctr -n k8s.io containers list` / `nvidia-smi` on the host.
- The producer's control plane through the same tunnel:
  `gcloud compute ssh ... --tunnel-through-iap -- -N -L 18080:127.0.0.1:8080`
  puts the producer's loopback port on the laptop. Control routes want a
  Google ID token for the slot's origin from a listed invoker; in debug
  mode the `masseuse-video-tee-soak` service account is one
  (`gcloud auth print-identity-token --impersonate-service-account=$(terraform output -raw soak_service_account) --audiences=https://slot-0.tee.masseuse.ai --include-email`).
  The verifier runs through the same tunnel with `-fetch-base
  http://127.0.0.1:18080 -allow-debug`; the two TLS checks report themselves
  unverifiable (a tunnel ends at the producer, not at Caddy), so the run
  cannot pass, which is the point of the flag being debug-only.
- The host key changes on every boot (the disk is ephemeral):
  `ssh-keygen -R compute.<instance id> -f ~/.ssh/google_compute_known_hosts`.
- The certificate is publicly trusted in debug mode too (Google Trust
  Services), so a phone can test the debug slot as is.
- The `attestation-debug` WIF provider admits `dbgstat == 'enabled'`.

## Image signing (`terraform/signing.tf`)

Every image digest is signed with the project's Cloud KMS key
(`masseuse-video-tee-cosign/image-signer`, `EC_SIGN_P256_SHA256`,
`prevent_destroy`), through `cosign sign --key gcpkms://...` with the two
annotations Confidential Space requires (`dev.cosignproject.cosign/sigalg`,
`dev.cosignproject.cosign/pub`), no Rekor upload. Only the release
workflow's identity (`github-release`, `terraform/github-release.tf`)
holds `cloudkms.signerVerifier` on the key, and it can only be assumed by a
tag build of this repository. The signature is a tag on the image's own
repository (`.../masseuse-video-tee:sha256-<digest>.sig`), which is
therefore what `tee-signed-image-repos` names. cosign 2.x writes that
layout; the keyless signature on ghcr.io is cosign 3's bundle format, which
the launcher does not read and does not need.

The key's fingerprint (hex sha256 over the DER public key, which is what a
token reports in `submods.container.image_signatures[].key_id`) is
`terraform output image_signer_fingerprint`; the PEM is
`image_signer_public_key_pem`. Both are in `VERIFY.md`.

Enforcement is `require_signed_image = true`: the VM gets
`tee-signed-image-repos` naming the image repository, which makes the
launcher pull `sha256-<digest>.sig` from it at every token mint and put the
verified signatures into the attestation (`submods.container.image_signatures`).
The launcher itself does not refuse an unsigned image (it logs the miss and
mints a token with no signatures); the refusals are downstream, where they
are checked: the production WIF provider's condition demands a signature by
the key before the image can read the weights, and the trainer's and the
clients' policy (`imageSignatures`) refuse to lease or send media to a slot
whose token carries none.

## The sweeper (`terraform/sweeper.tf`)

The backstop for a slot that did not end itself. `sweeper_enabled = true`
(the default) adds a Cloud Scheduler job (`sweeper_schedule`, default every
15 minutes) that runs a Cloud Workflow which stops any slot RUNNING longer
than `slot_max_uptime_hours` (7; the producer caps a lease at 6, so a
legitimate session can never hit it), passing `discardLocalSsd`. It never
starts a VM: only a visitor's tap does that, through the trainer. The
workflow runs as `masseuse-video-tee-sweeper`, whose only power is a custom
role with `compute.instances.get/stop` and `compute.zoneOperations.get`; it
cannot start, delete or reconfigure a VM and cannot read anything the
enclave holds. `gcloud workflows run masseuse-video-tee-sweeper --location
us-central1` runs it by hand and returns the list of instances it stopped.

## The verifier (`verifier/`, `VERIFY.md`)

`go build ./verifier` gives a standalone checker anybody can point at a slot
(`-origin https://slot-0.tee.masseuse.ai -allowed-digests sha256:...
-signer-key-ids <fingerprint>`); it fetches `/attestation` with a fresh
nonce and `/evidence-key`, verifies the RS256 signature against Google's
JWKS and checks every claim the clients do plus the TLS SPKI binding (the
browser cannot see its certificate; this can), and optionally re-runs
`cosign verify`. `go test ./verifier` covers the checks against a fake
slot.

## Certificate authority (`tools/gts-acme-test.sh`)

Every boot mints a new TLS key and certificate inside the enclave and both
die with it. That is not a convenience choice: the only storage that could
carry a key across boots lives in a project whose IAM owners are the
operators, and an operator holding the slot's TLS key could stand up a
look-alike endpoint. So there is no persisted key, no persisted certificate,
and no "renewal" in the ACME sense: each boot is a first issuance for
`slot-N.tee.masseuse.ai` from a brand-new account.

Let's Encrypt cannot serve that pattern: its limit of 5 certificates per
exact identifier set per 7 days counts every boot, and the renewal
exemption needs the previous certificate, which the slot no longer has.
Google Trust Services documents only per-project request quotas (`newOrder`
100/h, `newAccount` 25/min and 100/h, `newAuthz` 300/h on the ACME side;
`publicca.googleapis.com/requests` 120/min on the EAB-minting API), so
whether an undocumented repetition heuristic exists had to be tested.
`tools/gts-acme-test.sh` is the rig: a throwaway VM reproducing the
enclave's pattern (mint an EAB with `externalAccountKeys.create`, register a
fresh account, order over TLS-ALPN-01 on :443, fresh key and storage every
time), one JSON line per iteration. 72 issuances for one hostname, 64 of
them inside a single hour, no `rateLimited`, no 429; the production chain
is leaf (ECDSA P-256, 90 days) → `WR1` → `GTS Root R1`. Decision: GTS, in
both modes.

How it is wired:

- `main.tf`: `publicca.googleapis.com` enabled; `local.acme_directory_url`
  is `https://dv.acme-v02.api.pki.goog/directory` regardless of
  `debug_mode`. The `acme_directory_url` override still works; pointing it
  at GTS's `test-api` directory makes the entrypoint mint from
  `preprod-publicca`.
- `acme.tf`: `roles/publicca.externalAccountKeyCreator` on every
  digest-pinned `principalSet://.../attribute.image_digest/<digest>` (the
  same list `bucket.tf` grants the weights to) and on no service account.
  Only an attested image on TDX with the GPU in CC mode can mint an EAB, so
  no operator can register an ACME account in a slot's name.
- The entrypoint mints the EAB right before Caddy with the same retry the
  weights get: `externalAccountKeys.create` with the rendered
  `external_account` credential, `b64MacKey` decoded exactly once (the REST
  field is standard base64 of the base64url HMAC the ACME client wants),
  `EAB_KID` and `EAB_HMAC` handed to Caddy through a 0600 file on the tmpfs
  and the environment, never stdout. No EAB after five attempts exits 68.
- The Caddyfile: `eab {$EAB_KID} {$EAB_HMAC}` in the `issuer acme` stanza;
  `disable_http_challenge` stays (TLS-ALPN-01 only).
- DNS: `CAA 0 issue "pki.goog"` on `tee.masseuse.ai`.

Alerting: `serviceruntime.googleapis.com/api/request_count` for
`publicca.googleapis.com` and `quota/rate/net_usage` for
`publicca.googleapis.com/requests`. The ACME-side hourly quotas are not
exported anywhere; they surface only as `rateLimited` in Caddy's log, so a
slot whose token stays at `tlsNonce=no` is the signal.

Found on the way: the lego ACME client cannot pass GTS validation over
TLS-ALPN-01, because its challenge certificate sets `KeyUsage =
keyEncipherment` only and GTS's BoringSSL-based validators refuse to let
such a certificate sign a TLS 1.3 CertificateVerify. The script documents
the one-line patch. It does not affect the enclave (Caddy).

Not done: pinning a shorter `notAfter`. Caddy's `cert_lifetime` could ask
GTS for a 7-day certificate, which would shrink the window a leaked key is
useful for; with the sweeper's 7-hour uptime cap a renewal never runs, so
the SPKI nonce binding would stay safe.

## Rolling an image

Every change to what runs in the enclave is a digest roll, and the order
matters because the trainer refuses to lease, and the clients refuse to
send media to, a digest that is not in the policy:

1. Release the image (tag; the workflow builds, signs, attests and
   promotes). Note the digest.
2. Here: the new digest into `candidate_image_digests`, `terraform apply`
   (the VM's `tee-image-reference` and the digest's WIF bindings).
3. Trainer: the new digest into its policy beside the previous one (so a
   phone mid-session is not refused), apply, deploy.
4. Hand-boot a slot with no lease and run `tee-verify` against it; every
   check passes. Append the row to `VERIFY.md` (tag, digest, what changed).
5. Move the digest into `container_image` / `container_image_digest`,
   `terraform apply`; carry a session end to end from a phone.
6. Drop the previous digest from the trainer's policy and from
   `candidate_image_digests` once no slot runs it.

Rolling the gateway alone (a new `masseuse-camlink` release) is the same
with one step in front: `camlink.lock` gets the new tag and the
`masseuse-camlink-gateway_X.Y.Z_linux_amd64` checksum from that release's
signed `checksums.txt`; the image build reproduces the binary at that tag
and refuses any other bytes. A connector older than the gateway keeps
working as long as the frame protocol (`docs/PROTOCOL.md` in that
repository) is unchanged.

## The production flip

Preconditions: the debug posture has carried sessions end to end from a
phone, `slot-N.tee.masseuse.ai` resolves and holds a Google Trust Services
certificate, the on-demand lifecycle has been exercised (tap → boot → live
→ close → the VM reads `TERMINATED` on its own), and the trainer IAM and
the sweeper have applied.

1. In `terraform.tfvars`: `debug_mode = false`, `require_signed_image =
   true`. `terraform apply` replaces the VM (production
   `confidential-space` family, no log redirect, no memory monitoring, no
   IAP SSH rule, OS Login off, `tee-restart-policy=Never`,
   `tee-signed-image-repos` set), destroys the `attestation-debug` WIF
   provider and the soak service account, and tightens the production
   provider to signed images.
2. Trainer: `allow_debug = false`, `require_stable = true`,
   `require_gpu_cc = true`, `image_signatures = [the fingerprint]`.
3. `tee-verify` without `-allow-debug` exits 0 with `cs.dbgstat`,
   `cs.support_attributes.STABLE`, `image.signature` and `nonce.tls-spki`
   all passing. Note it in `VERIFY.md`.

What the flip forfeits: container stdout. The production image sends
nothing to Cloud Logging, so the producer's status route through the
trainer's control token and the attestation are the only telemetry; a slot
that fails to come up is diagnosed by rebuilding it with `debug_mode =
true`. What it gains: the launcher itself powers the VM off two minutes
after the workload exits, a second path beside the trainer's reaper.

## VPC Service Controls

`terraform/vpc-sc.tf` is the perimeter (restricted services: STS, IAM
credentials, KMS, Storage, Artifact Registry, Compute, Confidential
Computing; egress only to the Confidential Space image project and the
launcher's bucket; operators admitted through an access level). It needs
an organization-level Access Context Manager policy, and
`prod-masseuse-video-tee` has no organization parent, so the perimeter
stays disabled (`vpc_sc_enabled = false`). Moving the project under an
organization and setting `access_policy_id` turns it on; `vpc_sc_dry_run =
true` first, to see what it would have blocked.
