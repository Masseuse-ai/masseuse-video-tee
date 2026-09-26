#!/usr/bin/env bash
# The Confidential Space slot's one entrypoint: Caddy (TLS), MediaMTX (the
# DTLS-SRTP endpoint), the camlink gateway (the home-camera tunnel), the
# producer in --tee mode with its arguments baked here (frames and, with
# --audio, the stream's audio track), the weights and the analysis bundle,
# then the analysis process under its own user - the image's launch policy
# forbids a command override, so what this file says is what the enclave
# runs, and the attestation token's submods.container.image_digest vouches
# for it. Which analysis bundle may run is fixed by /app/analysis.lock, also
# inside the image (analysis/protocol.md).
#
# Environment (the launch policy's allow_env_override list, set as
# tee-env-* metadata by masseuse-video-tee/terraform):
#   TEE_PUBLIC_HOST                    slot-0.tee.masseuse.ai
#   TEE_PUBLIC_IP                      the VM's static IP (discovered from the
#                                      metadata server when unset)
#   SLOT_NAME                          tee-slot-0 (the trainer's per-slot ingest route)
#   TRAINER_URL                        https://masseuse.ai
#   TRAINER_INVOKER_SERVICE_ACCOUNT    the trainer's runtime SA(s), comma-separated
#   MODELS_BUCKET                      prod-masseuse-video-tee-models
#   WIF_AUDIENCE                       //iam.googleapis.com/projects/N/locations/global/workloadIdentityPools/P/providers/X
#   ACME_DIRECTORY_URL                 Google Trust Services (production, or the
#                                      test-api staging directory)
#   ACME_CONTACT_EMAIL                 the ACME account contact
#   ACME_EAB_IMPERSONATE               optional: mint the EAB through this service
#                                      account instead of directly (tee_eab.py)
#   TEE_IDLE_EXIT_S                    optional, 60: how long the slot stays up
#                                      with no lease and no session before it
#                                      exits and the VM stops (tee_mode.IdleExit)
#   TEE_BOOT_IDLE_S                    optional, 300: the same for a boot that
#                                      never gets a lease at all
#   POSE_GRAPH_BENCH                   optional, unset: batch sizes ("1,2,4,8")
#                                      for the boot-time pose graph bench
#                                      (pixel/pose_bench.py); a debug-slot
#                                      knob set by hand, never by Terraform
#   POSE_LOAD_BENCH                    optional, unset: "bodyFps,faceFps,seconds"
#                                      ("9,9,120") for the boot-time pose load
#                                      bench (pixel/pose_load.py); the same
#                                      kind of knob
set -euo pipefail

log() { printf 'entrypoint: %s\n' "$*"; }

for name in TEE_PUBLIC_HOST SLOT_NAME TRAINER_URL TRAINER_INVOKER_SERVICE_ACCOUNT \
            MODELS_BUCKET WIF_AUDIENCE ACME_DIRECTORY_URL ACME_CONTACT_EMAIL; do
    if [ -z "${!name:-}" ]; then
        log "missing $name"
        exit 64
    fi
done

metadata() {
    python3 - "$1" <<'EOF' || true
import sys, urllib.request
request = urllib.request.Request(
    "http://metadata.google.internal/computeMetadata/v1/" + sys.argv[1],
    headers={"Metadata-Flavor": "Google"})
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        print(response.read().decode().strip())
except Exception:
    pass
EOF
}

if [ -z "${TEE_PUBLIC_IP:-}" ]; then
    TEE_PUBLIC_IP="$(metadata instance/network-interfaces/0/access-configs/0/external-ip)"
fi
if [ -z "$TEE_PUBLIC_IP" ]; then
    log "no public IP: TEE_PUBLIC_IP unset and the metadata server gave none"
    exit 65
fi
export TEE_PUBLIC_IP

# Everything that must not outlive the boot lives on the /run/tee tmpfs
# (tee-mount): the TLS key, the capture rows, the rendered credential, the
# analysis socket. The analysis user gets the socket directory and nothing
# else under /run/tee.
mkdir -p /run/tee/caddy /run/tee/capture /run/tee/analysis /models
if [ ! -w /run/tee ]; then
    log "/run/tee is not writable: is the tee-mount tmpfs present?"
    exit 66
fi
chmod 0700 /run/tee/caddy /run/tee/capture
chown root:analysis /run/tee/analysis
chmod 0770 /run/tee/analysis

# The enclave's credential: an external_account file whose subject token
# is the launcher's attestation claims token. google-auth exchanges it at
# STS; the WIF provider's condition decides whether this image on this
# hardware may read the models bucket.
sed "s|\${WIF_AUDIENCE}|${WIF_AUDIENCE}|" /app/tee/wif-credential.json.tmpl \
    > /run/tee/wif-credential.json
chmod 0600 /run/tee/wif-credential.json
export GOOGLE_APPLICATION_CREDENTIALS=/run/tee/wif-credential.json
# An external_account credential names no project, and google-cloud-storage
# refuses to construct a client without one (it is only the billing/quota
# project for object reads). The VM's own project is the one that owns the
# bucket, so take it from the metadata server unless the operator set it.
if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
    GOOGLE_CLOUD_PROJECT="$(metadata project/project-id)"
fi
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    log "no project id: GOOGLE_CLOUD_PROJECT unset and the metadata server gave none"
    exit 67
fi
export GOOGLE_CLOUD_PROJECT

pids=()
shutdown() {
    log "stopping"
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
}
trap shutdown TERM INT

# The ACME account's External Account Binding, minted with the credential
# above: Google Trust Services admits no account without one, and every
# boot is a new account. The same claims-token race as the weights below,
# so the same retry; no EAB means no certificate, and exit 68 lets the
# launcher's restart policy try the boot again.
attempt=0
until python3 /app/workload/producer/tee_eab.py --directory "$ACME_DIRECTORY_URL" --out /run/tee/eab.env; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 5 ]; then
        log "no EAB after $attempt attempts"
        exit 68
    fi
    log "EAB attempt $attempt failed; retrying in 15s"
    sleep 15
done
set -a
# shellcheck disable=SC1091
. /run/tee/eab.env
set +a
rm -f /run/tee/eab.env

log "caddy for $TEE_PUBLIC_HOST via $ACME_DIRECTORY_URL (EAB $EAB_KID)"
caddy run --config /app/tee/Caddyfile --adapter caddyfile &
pids+=($!)
# Caddy has them; nothing else started below needs to.
unset EAB_KID EAB_HMAC

log "mediamtx advertising $TEE_PUBLIC_IP:8189"
MTX_WEBRTCADDITIONALHOSTS="$TEE_PUBLIC_IP" mediamtx /app/tee/mediamtx.tee.yml &
pids+=($!)

# The home-camera tunnel gateway (github.com/FemLed/masseuse-camlink): Caddy
# hands it /ingest/tunnel on 8090; MediaMTX dials 127.0.0.1:7441 as the
# camera when the producer puts an external source into tunnel mode; the
# producer drives it on 8091 (external_source.py, camlink_gateway.py). All
# three are loopback: nothing about the tunnel is reachable except through
# Caddy's TLS and the ticket the trainer brokered.
export CAMLINK_GATEWAY_CONTROL=http://127.0.0.1:8091
log "camlink gateway $(masseuse-camlink-gateway --version) (ws 8090, relay 7441, control 8091)"
masseuse-camlink-gateway -ws 127.0.0.1:8090 -relay 127.0.0.1:7441 -control 127.0.0.1:8091 &
pids+=($!)

# The slot's lifetime is one session: the trainer's /teardown after the
# session drains for TEE_IDLE_EXIT_S and exits (a /warmup or /produce inside
# the drain - a page refresh - cancels it), and tee_mode.IdleExit exits on
# its own when no lease and no session has been held for as long, or when
# a boot never gets a lease (TEE_BOOT_IDLE_S). Either way the exit is 0.
export TEE_IDLE_EXIT_S="${TEE_IDLE_EXIT_S:-60}"
export TEE_BOOT_IDLE_S="${TEE_BOOT_IDLE_S:-300}"

# The producer starts before the weights land: --tee prewarms (imports
# torch, builds the models) at once, and its boot waits on this marker,
# which tee_models.py below touches last, before it reads a weight. The
# two slowest things on the boot run side by side (docs/OPERATIONS.md
# "Boot time"). Its readings go to the analysis socket's owner and come
# back to be posted to the trainer; the socket is the only path between
# the two processes.
export POSE_MODELS_READY_FILE=/models/.complete
rm -f "$POSE_MODELS_READY_FILE"
ANALYSIS_SOCKET=/run/tee/analysis/analysis.sock

log "producer --tee for $SLOT_NAME -> $TRAINER_URL (idle exit ${TEE_IDLE_EXIT_S}s, boot idle ${TEE_BOOT_IDLE_S}s)"
python3 /app/workload/producer/producer.py \
    --tee \
    --pose gpu \
    --pose-fps 9 \
    --stream rtsp://127.0.0.1:8554/cam \
    --input-lost-after 90 \
    --teardown-drain-s "$TEE_IDLE_EXIT_S" \
    --post-url "${TRAINER_URL}/api/pose-signals/slot/${SLOT_NAME}/readings" \
    --post-interval-s 1.0 \
    --audio \
    --analysis-socket "$ANALYSIS_SOCKET" \
    --sink-dir /run/tee/capture \
    --capture-bucket "" \
    --overlay-publish rtsp://127.0.0.1:8554/overlay \
    --overlay-size 720x1280 \
    --overlay-fps 30 \
    --overlay-bitrate 6M \
    --overlay-delay-s 1.0 \
    --overlay-encoder x264 \
    --overlay-mirror \
    --port 8080 &
pids+=($!)

# Weights and the analysis bundle into the /models tmpfs, then the marker
# the producer's boot is waiting on. The bundle is the one /app/analysis.lock
# names, or nothing: tee_models.py refuses a bundle whose SHA-256 is not the
# lock's. The claims token the credential reads appears once the launcher
# has attested; the first attempt may race it, so retry. No weights, no
# slot: the producer goes down with the script.
attempt=0
until python3 /app/workload/producer/tee_models.py --bucket "$MODELS_BUCKET" --target /models \
        --analysis-lock /app/analysis.lock \
        --ready-file "$POSE_MODELS_READY_FILE"; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 5 ]; then
        log "weights did not land after $attempt attempts"
        shutdown
        exit 67
    fi
    log "weights attempt $attempt failed; retrying in 15s"
    sleep 15
done

# The analysis process (analysis/protocol.md), from the verified bundle,
# under its own user with a scrubbed environment: no credential (the WIF
# file is root-only), no capture directory, no TLS material, no network
# use of its own - what it emits goes back through the producer's socket.
# Its readiness marker tells the log the socket is up; the producer's
# connect retries for a few seconds anyway.
log "analysis $(cat /models/analysis/VERSION) on $ANALYSIS_SOCKET as analysis"
setpriv --reuid=analysis --regid=analysis --clear-groups \
    env -i PATH=/venv/bin:/usr/bin:/bin HOME=/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    python3 /models/analysis/main.py \
        --socket "$ANALYSIS_SOCKET" \
        --ready-file /run/tee/analysis/.ready &
pids+=($!)

# The producer is the workload: when it exits, so does the container, with
# its status, and the slot stops answering. The VM then stops: the
# trainer's reaper sees the enclave gone and calls instances.stop, and on
# the production image (tee-restart-policy=Never, vm.tf) the launcher
# powers off two minutes later as well; the debug image holds the VM for
# the reaper. The Spot meter stops until the trainer's next
# instances.start. Caddy, MediaMTX, the gateway or the analysis process
# dying takes the producer down with it via the wait below.
wait -n "${pids[@]}"
status=$?
log "a process exited with $status; stopping the rest"
shutdown
exit "$status"
