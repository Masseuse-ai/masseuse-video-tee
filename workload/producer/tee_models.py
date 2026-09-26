"""The model store into the enclave's /models tmpfs, with the enclave's own
credential and no mount - and the analysis bundle, pinned by analysis.lock.

The Confidential Space slot has no bucket mount and a VM service account
that can read nothing; the credential that can is minted from the
attestation token - `GOOGLE_APPLICATION_CREDENTIALS` points at an
`external_account` file whose `credential_source.file` is the launcher's
claims token, so google-auth exchanges it at STS through the WIF pool whose
provider only admits this image digest on TDX with the GPU in CC mode. The
weights therefore land only in an enclave Google's attestation service has
vouched for, and only in memory (`tee-mount` makes /models a tmpfs).

Run once by tee/entrypoint.sh, alongside the producer it has just started:

    python3 tee_models.py --bucket $MODELS_BUCKET --target /models \
        --analysis-lock /app/analysis.lock --ready-file /models/.complete

Copies the prefixes the boot reads (hf-bf16/, detectors/), skipping what
the pose boot never opens; fetches the analysis bundle the lock names,
refuses it unless its SHA-256 is the lock's, and unpacks it to
<target>/analysis/; then touches --ready-file. The producer's boot
(live_pose.wait_for_model_store, POSE_MODELS_READY_FILE) imports torch
meanwhile and waits for that file before it loads weights, so the two
slowest things on the boot overlap. Exit code non-zero on any failure so
the entrypoint can retry; a stale marker is removed first, so a retry's
partial store never reads as complete.

The lock is part of the image, so the attestation digest covers which
analysis bundle may run (analysis/protocol.md).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 64 * 1024 * 1024
WORKERS = 16
DEFAULT_PREFIXES = ("hf-bf16/", "detectors/")
ANALYSIS_DIR = "analysis"


def skip(name: str) -> bool:
    """Mirror of live_pose._prefetch_skip on object names."""
    parts = name.split("/")
    if "blobs" in parts or ".locks" in parts:
        return True
    return parts[-1] == "sapiens2_1b_pose.safetensors"


def plan(bucket, prefixes: list[str]) -> list[tuple[str, int]]:
    """(object name, size) for everything under the prefixes that the boot
    reads. A prefix without a trailing slash names one object."""
    jobs: list[tuple[str, int]] = []
    for prefix in prefixes:
        if prefix.endswith("/"):
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith("/") or skip(blob.name):
                    continue
                jobs.append((blob.name, int(blob.size or 0)))
        else:
            blob = bucket.get_blob(prefix)
            if blob is None:
                raise FileNotFoundError(f"gs://{bucket.name}/{prefix}")
            jobs.append((blob.name, int(blob.size or 0)))
    return jobs


def fetch(bucket, jobs: list[tuple[str, int]], target: Path) -> None:
    from google.cloud.storage import transfer_manager

    # Staged inside the target: /models is its own tmpfs, and a rename from
    # anywhere else is a cross-device link the kernel refuses.
    partial = target / ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for name, size in jobs:
        (partial / name).parent.mkdir(parents=True, exist_ok=True)
    large = [job for job in jobs if job[1] > CHUNK]
    small = [job for job in jobs if job[1] <= CHUNK]
    for name, _ in large:
        transfer_manager.download_chunks_concurrently(
            bucket.blob(name), str(partial / name), chunk_size=CHUNK,
            max_workers=WORKERS)

    def one(job) -> None:
        name, _ = job
        bucket.blob(name).download_to_filename(str(partial / name))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, small))
    # Whole prefixes appear at once: a producer that boots mid-copy sees
    # either nothing or everything.
    for child in list(partial.iterdir()):
        destination = target / child.name
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        os.replace(child, destination)
    shutil.rmtree(partial, ignore_errors=True)


# -- the analysis bundle -------------------------------------------------------

def read_lock(path: Path) -> dict[str, str]:
    """`key=value` lines; `version`, `object` and `sha256` are required."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    missing = [key for key in ("version", "object", "sha256") if not values.get(key)]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    digest = values["sha256"].lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{path}: sha256 is not 64 hex digits")
    values["sha256"] = digest
    return values


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(tarball: Path, destination: Path) -> None:
    """Extract `analysis/` from the tarball into `destination`'s parent,
    world-readable, so the analysis user can read it."""
    staging = destination.parent / ".analysis-partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    subprocess.run(
        ["tar", "--zstd", "-xf", str(tarball), "-C", str(staging),
         "--no-same-owner", "--no-same-permissions"],
        check=True)
    unpacked = staging / ANALYSIS_DIR
    if not (unpacked / "main.py").is_file():
        raise FileNotFoundError(f"{tarball.name} holds no analysis/main.py")
    for root, dirs, files in os.walk(unpacked):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(unpacked, destination)
    shutil.rmtree(staging, ignore_errors=True)


def fetch_analysis(bucket, lock: dict[str, str], target: Path) -> Path:
    """The bundle the lock names, verified, unpacked to <target>/analysis/."""
    partial = target / ".partial-analysis"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    tarball = partial / Path(lock["object"]).name
    blob = bucket.get_blob(lock["object"])
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket.name}/{lock['object']}")
    blob.download_to_filename(str(tarball))
    actual = sha256_of(tarball)
    if actual != lock["sha256"]:
        tarball.unlink(missing_ok=True)
        raise ValueError(
            f"analysis bundle {lock['object']} is sha256 {actual}, "
            f"analysis.lock pins {lock['sha256']}; refusing it")
    destination = target / ANALYSIS_DIR
    unpack(tarball, destination)
    (destination / "VERSION").write_text(lock["version"] + "\n")
    shutil.rmtree(partial, ignore_errors=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("MODELS_BUCKET", ""))
    parser.add_argument("--target", default=os.environ.get("POSE_MODELS_MOUNT", "/models"))
    parser.add_argument("--prefixes",
                        default=os.environ.get("TEE_MODEL_PREFIXES",
                                               ",".join(DEFAULT_PREFIXES)),
                        help="comma-separated object prefixes (trailing slash) "
                             "or object names")
    parser.add_argument("--analysis-lock", default="",
                        help="analysis.lock naming the analysis bundle to "
                             "fetch, verify and unpack to <target>/analysis/; "
                             "empty = no bundle")
    parser.add_argument("--ready-file", default="",
                        help="touched once the store is complete (the producer "
                             "waits for it: POSE_MODELS_READY_FILE)")
    args = parser.parse_args(argv)
    if not args.bucket:
        print("tee-models: no bucket configured", flush=True)
        return 2
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    lock = read_lock(Path(args.analysis_lock)) if args.analysis_lock else None
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    ready = Path(args.ready_file) if args.ready_file else None
    if ready is not None:
        ready.unlink(missing_ok=True)
    started = time.monotonic()

    import multiprocessing

    from google.cloud import storage

    # transfer_manager forks workers; spawn keeps them clear of any lock
    # the credential exchange may hold.
    multiprocessing.set_start_method("spawn", force=True)
    # The WIF credential carries no project; the entrypoint exports the VM's
    # (only the quota project for these reads, so any real project would do).
    client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    bucket = client.bucket(args.bucket)
    jobs = plan(bucket, prefixes)
    total = sum(size for _, size in jobs)
    print(f"tee-models: {len(jobs)} objects, {total / 1e9:.2f} GB from "
          f"gs://{args.bucket} -> {target}", flush=True)
    fetch(bucket, jobs, target)
    if lock is not None:
        where = fetch_analysis(bucket, lock, target)
        print(f"tee-models: analysis {lock['version']} "
              f"(sha256 {lock['sha256'][:12]}...) -> {where}", flush=True)
    if ready is not None:
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.touch()
    elapsed = max(time.monotonic() - started, 1e-6)
    print(f"tee-models: done in {elapsed:.1f}s ({total / 1e6 / elapsed:.0f} MB/s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
