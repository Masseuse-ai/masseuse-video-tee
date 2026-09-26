# The base image: Python, the pinned torch set, ffmpeg, and the workload
# code. Build context is the repository root (.dockerignore keeps it to
# workload/ and the lock files):
#
#   docker buildx build -f workload/Dockerfile --build-arg CUDA_BASE=ubuntu:24.04 .
#
# Weights are deliberately NOT baked in: tee_models.py copies them into a
# tmpfs at boot, with the credential the attestation earns, so the image is
# torch + ffmpeg and nothing heavier, and the same digest serves every model
# version the models bucket holds.
#
# The base is a build arg. Nothing from a CUDA runtime image is loaded at run
# time: torch's cu130 wheels bundle their own CUDA libraries (nvidia-cublas,
# cudnn, ...), Ubuntu's ffmpeg does not link the toolkit, and
# libcuda/nvidia-smi come from the driver the Confidential Space launcher
# mounts at /usr/local/nvidia. The release build therefore passes
# CUDA_BASE=ubuntu:24.04 and saves the 1.6 GB of compressed CUDA layers that
# cost every enclave boot 20+ s.
#
# The CUDA line of the torch wheels is the build arg that actually selects
# the kernels: the Confidential Space H100 (595.x driver) takes CUDA 13.0.x.
# Versions are pinned so a rebuild reproduces the layer set the attested
# digest was measured on.
#
# Layer layout, for the enclave's boot: Confidential Space pulls the image
# on every boot (nothing persists between boots) and containerd fetches
# layers concurrently but applies them one at a time. One 3.3 GB layer is
# one TCP stream followed by one long decompression, so the torch set is
# installed from a wheels stage in WHEEL_BUCKETS size-balanced layers
# (bucket-N.txt, greedy by wheel size), each its own RUN, pip --no-deps, with
# a pip check at the end. triton (torch.compile) is left out: nothing here
# compiles, and it is 250 MB. This needs BuildKit (RUN --mount), which is
# what workload/tee/buildx.sh and the release workflow use.
ARG CUDA_BASE=ubuntu:24.04
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0

# -- the torch wheel set, downloaded once and partitioned ---------------------
FROM ubuntu:24.04 AS wheels
ARG TORCH_INDEX
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG WHEEL_BUCKETS=5
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Same interpreter and platform as the final stage (Ubuntu 24.04, CPython
# 3.12, manylinux x86_64), so the resolved set is exactly what installs.
RUN pip download --dest /wheels --only-binary=:all: --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX}" \
    && rm -f /wheels/triton-*.whl
RUN python3 - "$WHEEL_BUCKETS" <<'EOF'
import pathlib, sys
buckets = int(sys.argv[1])
wheels = sorted(pathlib.Path("/wheels").glob("*.whl"),
                key=lambda p: (-p.stat().st_size, p.name))
plan = [[] for _ in range(buckets)]
load = [0] * buckets
for wheel in wheels:                       # largest first, onto the lightest
    i = load.index(min(load))
    plan[i].append(wheel)
    load[i] += wheel.stat().st_size
for i, group in enumerate(plan):
    pathlib.Path(f"/wheels/bucket-{i}.txt").write_text(
        "".join(f"{w}\n" for w in group))
    print(f"bucket-{i}: {load[i] / 1e6:7.1f} MB  "
          + ", ".join(w.name.split("-")[0] for w in group))
EOF

# -- the image -----------------------------------------------------------------
FROM ${CUDA_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_OFFLINE=0

# ca-certificates for the ACME client and GCS on a plain Ubuntu base (the
# CUDA base carried it). The NVIDIA_* variables and LD_LIBRARY_PATH are what
# the CUDA base set: the driver mount is where libcuda lives at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64

RUN python3 -m venv /venv

# The torch set, one layer per bucket (see the header). The bind mount
# leaves nothing of the wheels stage in the image.
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    xargs -r -a /wheels/bucket-0.txt /venv/bin/pip install --no-cache-dir --no-index --no-deps
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    xargs -r -a /wheels/bucket-1.txt /venv/bin/pip install --no-cache-dir --no-index --no-deps
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    xargs -r -a /wheels/bucket-2.txt /venv/bin/pip install --no-cache-dir --no-index --no-deps
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    xargs -r -a /wheels/bucket-3.txt /venv/bin/pip install --no-cache-dir --no-index --no-deps
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    xargs -r -a /wheels/bucket-4.txt /venv/bin/pip install --no-cache-dir --no-index --no-deps
# Every requirement of the set is met except triton, dropped on purpose.
RUN /venv/bin/pip check > /tmp/pip-check.txt || true; \
    if grep -v -e 'requires triton' -e 'No broken requirements' /tmp/pip-check.txt | grep -q .; then \
        cat /tmp/pip-check.txt; exit 1; \
    fi; rm -f /tmp/pip-check.txt

# opencv headless because there is no display anywhere near this.
# transformers is pinned: pixel/pose_post.py builds the pose graph's
# post-processing from the Sapiens2 image processor's own functions, and
# the processor's on-device crop is the pose preprocessing, so the version
# the tests ran against is the version the enclave runs.
RUN /venv/bin/pip install --no-cache-dir \
        transformers==5.17.0 pillow opencv-python-headless numpy \
        pyyaml scipy google-cloud-storage

WORKDIR /app
# The workload: pixel/ (decode geometry, person detection, keypoints,
# regional motion descriptors), audio/ (the audio track's classification,
# level and pitch; the classifier's library and weights are added by the
# TEE layer) and producer/ (the session shell, the overlay, the relay proxy,
# the enclave's boot helpers). The tests are not copied (.dockerignore); the
# TEE layer copies the same tree again.
COPY workload/pixel /app/workload/pixel
COPY workload/audio /app/workload/audio
COPY workload/producer /app/workload/producer

ENV PATH="/venv/bin:$PATH"
EXPOSE 8080
ENTRYPOINT ["python3", "/app/workload/producer/producer.py"]
CMD ["--serve"]
