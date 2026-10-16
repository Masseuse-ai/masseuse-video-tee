"""The model store into the enclave's /models tmpfs, with the enclave's own
credential and no mount, pinned by models.lock - and the analysis bundle,
pinned by analysis.lock.

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
        --models-lock /app/models.lock --analysis-lock /app/analysis.lock \
        --ready-file /models/.complete

Copies exactly the objects models.lock names (the weights of the pose
model and the person detector, as the boot reads them), and refuses the
store unless every one is there with the lock's size and SHA-256; nothing
the lock does not name is copied. Fetches the analysis bundle
analysis.lock names, refuses it unless its SHA-256 is the lock's, and
unpacks it to <target>/analysis/; then touches --ready-file. The
producer's boot (live_pose.wait_for_model_store, POSE_MODELS_READY_FILE)
imports torch meanwhile and waits for that file before it loads weights,
so the two slowest things on the boot overlap. Exit code non-zero on any
failure so the entrypoint can retry; a stale marker is removed first, so a
retry's partial store never reads as complete.

Both locks are part of the image, so the attestation digest covers which
weights and which analysis bundle may run (analysis/protocol.md): whoever
can write the bucket can change what is in it, not what an attested image
will load. Changing the weights is a release.

Without --models-lock the prefixes the boot reads (hf-bf16/, detectors/)
are copied as listed, unverified: a development path the image never
takes. `--write-models-lock FILE` lists those prefixes and writes the lock
for what is there now.
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


def read_models_lock(path: Path) -> dict[str, tuple[int, str]]:
    """`<sha256> <size> <object name>` lines (`#` comments): the store,
    object by object. Names are relative and stay inside the target."""
    entries: dict[str, tuple[int, str]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"{path}:{number}: not '<sha256> <size> <object name>'")
        digest, size, name = parts[0].lower(), parts[1], parts[2]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{path}:{number}: sha256 is not 64 hex digits")
        if not size.isdigit():
            raise ValueError(f"{path}:{number}: size is not a byte count")
        segments = name.split("/")
        if (name.startswith("/") or name.endswith("/") or "\\" in name
                or any(s in ("", ".", "..") for s in segments)):
            raise ValueError(f"{path}:{number}: {name!r} is not an object name inside the store")
        if name in entries:
            raise ValueError(f"{path}:{number}: {name} is named twice")
        entries[name] = (int(size), digest)
    if not entries:
        raise ValueError(f"{path}: names no objects")
    return entries


def verify_store(root: Path, expected: dict[str, tuple[int, str]]) -> None:
    """Every object the lock names is under `root` with its size and
    SHA-256; raises ValueError naming every one that is not."""
    def problem(item) -> str | None:
        name, (size, digest) = item
        path = root / name
        if not path.is_file():
            return f"{name} is missing"
        actual_size = path.stat().st_size
        if actual_size != size:
            return f"{name} is {actual_size} bytes, models.lock pins {size}"
        actual = sha256_of(path)
        if actual != digest:
            return f"{name} is sha256 {actual}, models.lock pins {digest}"
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        problems = [p for p in pool.map(problem, sorted(expected.items())) if p]
    if problems:
        raise ValueError("refusing the model store: " + "; ".join(problems))


def write_models_lock(bucket, prefixes: list[str], path: Path) -> int:
    """List the prefixes as plan() does and write the lock for what is in
    the bucket now, hashing each object as it streams. Returns how many."""
    lines = []
    for name, size in sorted(plan(bucket, prefixes)):
        digest = hashlib.sha256()
        with bucket.blob(name).open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()} {size} {name}")
    header = ("# The model store tee_models.py copies into the enclave's /models,\n"
              "# object by object: <sha256> <size> <object name>. Part of the image,\n"
              "# so the attested digest covers which weights load; a store that\n"
              f"# differs is refused. Written from gs://{bucket.name} by\n"
              "#   python3 workload/producer/tee_models.py --bucket <bucket> --write-models-lock models.lock\n")
    path.write_text(header + "\n".join(lines) + "\n")
    return len(lines)


def fetch(bucket, jobs: list[tuple[str, int]], target: Path,
          expected: dict[str, tuple[int, str]] | None = None) -> None:
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
    # Checked where it landed, before any of it is in place.
    if expected is not None:
        try:
            verify_store(partial, expected)
        except ValueError:
            shutil.rmtree(partial, ignore_errors=True)
            raise
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
    parser.add_argument("--models-lock", default="",
                        help="models.lock naming every object of the store with "
                             "its size and SHA-256: exactly those are copied, and "
                             "the store is refused unless they all match; empty = "
                             "the prefixes, unverified")
    parser.add_argument("--write-models-lock", default="",
                        help="write models.lock for what the prefixes hold now "
                             "to this file, and copy nothing")
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
    expected = read_models_lock(Path(args.models_lock)) if args.models_lock else None
    lock = read_lock(Path(args.analysis_lock)) if args.analysis_lock else None
    target = Path(args.target)
    ready = Path(args.ready_file) if args.ready_file else None
    if not args.write_models_lock:
        target.mkdir(parents=True, exist_ok=True)
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
    if args.write_models_lock:
        count = write_models_lock(bucket, prefixes, Path(args.write_models_lock))
        print(f"tee-models: {count} objects from gs://{args.bucket} -> "
              f"{args.write_models_lock}", flush=True)
        return 0
    if expected is not None:
        jobs = [(name, size) for name, (size, _) in sorted(expected.items())]
    else:
        jobs = plan(bucket, prefixes)
    total = sum(size for _, size in jobs)
    print(f"tee-models: {len(jobs)} objects, {total / 1e9:.2f} GB from "
          f"gs://{args.bucket} -> {target}"
          f"{' (models.lock)' if expected is not None else ' (unverified)'}", flush=True)
    fetch(bucket, jobs, target, expected)
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
