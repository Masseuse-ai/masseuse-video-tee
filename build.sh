#!/usr/bin/env bash
# Build the enclave image locally, the way .github/workflows/release.yml
# does, without pushing anywhere: the base image (workload/Dockerfile) and
# the TEE layer (workload/tee/Dockerfile.tee) into the local Docker daemon,
# then the smoke test (workload/tee/buildx.sh) on the result.
#
#   bash build.sh                 # both images and the smoke test (~25 min, ~20 GB)
#   SMOKE=0 bash build.sh         # skip the smoke test
#
# Published digests are never built here: they come from the release
# workflow on a tag, with SLSA provenance and signatures, and are promoted
# into Artifact Registry by digest (VERIFY.md "How the image is built").
# The local images are gzip (zstd layers need a registry push), so their
# digests differ from a release's; the contents are the same tree.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# The daemon's builder: it sees the base image it just loaded, which a
# docker-container builder would not (FROM resolves through a registry).
export BUILDX_BUILDER=${BUILDX_BUILDER:-default}
BASE_IMAGE=${BASE_IMAGE:-masseuse-video-tee-base:local}
IMAGE=${IMAGE:-masseuse-video-tee:local}

# shellcheck disable=SC1091
. "$HERE/camlink.lock"
if ! [[ "${CAMLINK_VERSION:-}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] || ! [[ "${CAMLINK_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "camlink.lock needs CAMLINK_VERSION=vX.Y.Z and the 64-hex CAMLINK_SHA256 of" >&2
    echo "masseuse-camlink-gateway_X.Y.Z_linux_amd64 from that release's checksums.txt" >&2
    exit 64
fi

# shellcheck disable=SC1091
. "$HERE/workload/tee/buildx.sh"
docker buildx version

echo "== base: $BASE_IMAGE"
buildx_local "$BASE_IMAGE" -f workload/Dockerfile --build-arg CUDA_BASE=ubuntu:24.04 .

echo "== tee: $IMAGE (gateway $CAMLINK_VERSION)"
buildx_local "$IMAGE" -f workload/tee/Dockerfile.tee \
    --build-arg "BASE=$BASE_IMAGE" \
    --build-arg "CAMLINK_VERSION=$CAMLINK_VERSION" \
    --build-arg "CAMLINK_SHA256=$CAMLINK_SHA256" \
    .

if [ "${SMOKE:-1}" = "1" ]; then
    echo "== smoke"
    buildx_smoke "$IMAGE"
fi
echo "built $IMAGE"
