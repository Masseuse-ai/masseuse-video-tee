#!/usr/bin/env bash
# Sourced by .github/workflows/release.yml and by bash build.sh: the buildx
# invocations that produce the enclave image, in one place, so the release
# and a local build push (or load) the same bytes for the same tree.
#
# In the release, BuildKit runs in a pinned moby/buildkit container (the
# docker-container driver docker/setup-buildx-action creates), and every
# image is pushed as one OCI manifest whose layers are all one compression -
# zstd for the enclave, whose launcher pulls and unpacks the image on every
# boot (docs/OPERATIONS.md "Boot time"). Locally, bash build.sh uses the
# daemon's own builder ("default"), which can see the images it just built.
#
# The caller names the builder in BUILDX_BUILDER.
set -euo pipefail

BUILDX_HOME="${BUILDX_HOME:-${HOME:-/tmp}/.masseuse-video-tee-buildx}"
BUILDX_BUILDER="${BUILDX_BUILDER:-default}"
mkdir -p "$BUILDX_HOME"

# buildx_image <image ref> <compression> <level> <buildx build args...>:
# build and push, every layer recompressed to <compression> (the base
# images arrive gzip), one OCI manifest (no provenance/SBOM index, so the
# digest names the image itself), and print the digest.
buildx_image() {
    local image=$1 compression=$2 level=$3
    shift 3
    docker buildx build --builder "$BUILDX_BUILDER" --progress plain \
        --provenance=false --sbom=false \
        --output "type=image,name=${image},push=true,oci-mediatypes=true,compression=${compression},compression-level=${level},force-compression=true" \
        --metadata-file "$BUILDX_HOME/metadata.json" "$@"
    echo "pushed ${image} $(buildx_last_digest)"
}

# buildx_local <image ref> <buildx build args...>: the same build without a
# registry, loaded into the local daemon (gzip; zstd layers need a registry
# push), for bash build.sh.
buildx_local() {
    local image=$1
    shift
    docker buildx build --builder "$BUILDX_BUILDER" --progress plain \
        --provenance=false --sbom=false \
        --output "type=docker,name=${image}" \
        --metadata-file "$BUILDX_HOME/metadata.json" "$@"
}

# buildx_last_digest: the digest of the last buildx_image push.
buildx_last_digest() {
    sed -n 's/.*"containerimage.digest": *"\([^"]*\)".*/\1/p' "$BUILDX_HOME/metadata.json"
}

# The tests that need torch and transformers (the CPU-only CI skips them):
# the graph wrapper's contract, the packed post-processing against the
# processor's own (batch of one and batched), the batch bench's contract,
# the detector's host-side selection, and the pose step and parity check
# around stub models. Run in the smoke stage, where the libraries are the
# versions the enclave ships.
SMOKE_TESTS="tests/test_gpu_graph.py tests/test_pose_post.py tests/test_pose_bench.py tests/test_detector_post.py tests/test_live_pose.py"

# buildx_smoke <TEE image ref>: every import the enclave makes, on the
# built image, CPU only, run by BuildKit (the daemon's own image store may
# not take zstd layers). A system library the ubuntu base lacks fails here
# rather than on an A3 boot. Then SMOKE_TESTS, with pytest installed into
# this throwaway stage only (the image itself ships no test tooling; the
# tests are outside the build context, so they are copied in here).
buildx_smoke() {
    mkdir -p "$BUILDX_HOME/smoke"
    rm -rf "$BUILDX_HOME/smoke/tests"
    cp -R "$(dirname "${BASH_SOURCE[0]}")/../tests" "$BUILDX_HOME/smoke/tests"
    rm -rf "$BUILDX_HOME/smoke/tests/__pycache__"
    cat > "$BUILDX_HOME/smoke/Dockerfile" <<EOF
FROM $1
COPY tests /app/workload/tests
RUN /venv/bin/pip install --no-cache-dir pytest \\
 && cd /app/workload && python3 -m pytest -q -p no:cacheprovider ${SMOKE_TESTS}
RUN python3 -c "import sys; sys.path[:0] = ['/app/workload/audio', '/app/workload/pixel', '/app/workload/producer']; \\
import torch, torchvision, torchvision.ops, transformers, cv2, scipy, numpy, PIL, yaml, cryptography, google.cloud.storage; \\
import pose_track, live_pose, motion, tee_mode, tee_models, tee_eab, relay_proxy, analysis_link, camlink_gateway, external_source, overlay, sinks; \\
import audio_stage, audio_features, pitch, ced; \\
engine = ced.CedEngine(); scores = engine.classify(numpy.zeros(16000, dtype=numpy.float32)); \\
assert set(scores) == set(ced.TARGET_LABELS), scores; print('ced', engine.version, len(engine.labels), 'labels'); \\
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__, 'cuda', torch.version.cuda)" \\
 && python3 /app/workload/producer/producer.py --help > /dev/null \\
 && python3 /app/workload/producer/tee_models.py --help > /dev/null \\
 && test -f /app/analysis.lock && test -x /app/tee/entrypoint.sh \\
 && id analysis && setpriv --version && tar --zstd --help > /dev/null \\
 && caddy version && test -x /usr/local/bin/mediamtx && ffmpeg -version | head -1 \\
 && masseuse-camlink-gateway --version \\
 && test -f /etc/ssl/certs/ca-certificates.crt
EOF
    docker buildx build --builder "$BUILDX_BUILDER" --progress plain --no-cache \
        --output type=cacheonly "$BUILDX_HOME/smoke"
}
