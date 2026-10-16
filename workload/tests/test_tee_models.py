"""The mount-free model copy: which objects the enclave pulls into its
/models tmpfs, and that whole prefixes appear at once."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import tee_models  # noqa: E402


class FakeBlob:
    def __init__(self, bucket, name, size, payload=b""):
        self.bucket = bucket
        self.name = name
        self.size = size
        self.payload = payload or bucket.payloads.get(name) or name.encode()

    def download_to_filename(self, filename):
        if self.name not in self.bucket.objects:
            raise FileNotFoundError(f"gs://{self.bucket.name}/{self.name}")
        Path(filename).write_bytes(self.payload)

    def open(self, mode="rb"):
        return io.BytesIO(self.payload)


class FakeBucket:
    name = "models"

    def __init__(self, objects: dict[str, int], payloads: dict[str, bytes] | None = None):
        self.objects = objects
        self.payloads = payloads or {}

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, name, size) for name, size in sorted(self.objects.items())
                if name.startswith(prefix)]

    def get_blob(self, name):
        return FakeBlob(self, name, self.objects[name]) if name in self.objects else None

    def blob(self, name):
        return FakeBlob(self, name, self.objects.get(name, 0))


def fake_storage(monkeypatch, bucket=None):
    """google.cloud.storage as tee_models imports it: the transfer manager
    unused (every object is small here), a client whose bucket is `bucket`."""
    monkeypatch.setattr(tee_models, "CHUNK", 1024)
    fake_tm = SimpleNamespace(download_chunks_concurrently=lambda *a, **k: None)
    storage = SimpleNamespace(transfer_manager=fake_tm,
                              Client=lambda project=None: SimpleNamespace(bucket=lambda name: bucket))
    monkeypatch.setitem(sys.modules, "google.cloud.storage.transfer_manager", fake_tm)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(storage=storage))


def lock_line(name: str, payload: bytes) -> str:
    return f"{hashlib.sha256(payload).hexdigest()} {len(payload)} {name}"


def test_plan_takes_the_boot_prefixes_and_skips_what_the_boot_never_reads():
    bucket = FakeBucket({
        "hf-bf16/hub/models--x/snapshots/a/model.safetensors": 3_000_000_000,
        "hf-bf16/hub/models--x/snapshots/a/config.json": 900,
        "hf-bf16/hub/models--x/snapshots/a/sapiens2_1b_pose.safetensors": 3_000_000_000,
        "hf-bf16/hub/models--x/blobs/deadbeef": 3_000_000_000,
        "hf-bf16/hub/.locks/models--x/lock": 0,
        "hf-bf16/hub/": 0,
        "detectors/rtdetrv4/rtv4_hgnetv2_x_coco_ema.safetensors": 250_000_000,
        "artifacts/a.json": 4_000,
        "artifacts/other.json": 4_000,
        "runs/2026-09-01/poses.jsonl": 1,
        "testbed/mediamtx.yml": 1,
    })
    jobs = tee_models.plan(bucket, list(tee_models.DEFAULT_PREFIXES))
    assert [name for name, _ in jobs] == [
        "hf-bf16/hub/models--x/snapshots/a/config.json",
        "hf-bf16/hub/models--x/snapshots/a/model.safetensors",
        "detectors/rtdetrv4/rtv4_hgnetv2_x_coco_ema.safetensors",
    ]
    assert sum(size for _, size in jobs) == 3_000_000_000 + 900 + 250_000_000


def test_fetch_lands_whole_prefixes_in_place(tmp_path, monkeypatch):
    bucket = FakeBucket({"detectors/rtdetrv4/x.safetensors": 10,
                         "artifacts/a.json": 5})
    # Small objects only: the transfer manager is not exercised here.
    monkeypatch.setattr(tee_models, "CHUNK", 1024)
    jobs = tee_models.plan(bucket, ["detectors/", "artifacts/a.json"])
    target = tmp_path / "models"
    target.mkdir()
    (target / "detectors").mkdir()
    (target / "detectors" / "stale").write_text("old")
    fake_tm = SimpleNamespace(download_chunks_concurrently=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "google.cloud.storage.transfer_manager", fake_tm)
    monkeypatch.setitem(sys.modules, "google.cloud.storage",
                        SimpleNamespace(transfer_manager=fake_tm))
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(
        storage=sys.modules["google.cloud.storage"]))
    tee_models.fetch(bucket, jobs, target)
    assert (target / "detectors" / "rtdetrv4" / "x.safetensors").read_bytes() == \
        b"detectors/rtdetrv4/x.safetensors"
    assert (target / "artifacts" / "a.json").read_bytes() == b"artifacts/a.json"
    assert not (target / "detectors" / "stale").exists()
    assert not (target / ".partial").exists()


# -- models.lock -----------------------------------------------------------------

WEIGHTS = "hf-bf16/hub/models--x/snapshots/a/model.safetensors"
DETECTOR = "detectors/rtdetrv4/x.safetensors"


def test_models_lock_names_each_object_once_inside_the_store(tmp_path):
    lock = tmp_path / "models.lock"
    lock.write_text("# the store\n\n" + "\n".join([
        lock_line(WEIGHTS, b"weights"), lock_line(DETECTOR, b"detector")]) + "\n")
    assert tee_models.read_models_lock(lock) == {
        WEIGHTS: (7, hashlib.sha256(b"weights").hexdigest()),
        DETECTOR: (8, hashlib.sha256(b"detector").hexdigest()),
    }
    good = lock_line(DETECTOR, b"detector")
    for bad, reason in [
        ("", "names no objects"),
        ("abc 1 detectors/x", "sha256 is not 64 hex digits"),
        (f"{'a' * 64} 1k detectors/x", "size is not a byte count"),
        (f"{'a' * 64} 1", "not '<sha256> <size> <object name>'"),
        (f"{'a' * 64} 1 ../etc/passwd", "is not an object name inside the store"),
        (f"{'a' * 64} 1 /models/x", "is not an object name inside the store"),
        (f"{'a' * 64} 1 detectors//x", "is not an object name inside the store"),
        (f"{'a' * 64} 1 detectors/", "is not an object name inside the store"),
        (f"{good}\n{good}", "is named twice"),
    ]:
        lock.write_text(bad + "\n")
        with pytest.raises(ValueError, match=reason.replace("(", r"\(").replace(")", r"\)")):
            tee_models.read_models_lock(lock)


def test_a_store_the_lock_does_not_pin_is_refused_before_any_of_it_is_in_place(tmp_path, monkeypatch):
    payloads = {WEIGHTS: b"weights", DETECTOR: b"detector"}
    bucket = FakeBucket({name: len(p) for name, p in payloads.items()}, payloads)
    fake_storage(monkeypatch, bucket)
    target = tmp_path / "models"
    (target / "detectors").mkdir(parents=True)
    (target / "detectors" / "previous").write_text("the last good store")
    jobs = [(WEIGHTS, 7), (DETECTOR, 8)]
    expected = {WEIGHTS: (7, hashlib.sha256(b"weights").hexdigest()),
                DETECTOR: (8, hashlib.sha256(b"detector").hexdigest())}
    tee_models.fetch(bucket, jobs, target, expected)
    assert (target / WEIGHTS).read_bytes() == b"weights"
    assert (target / DETECTOR).read_bytes() == b"detector"

    # Someone with write access to the bucket swaps the detector: same
    # name and size, other bytes.
    (target / "detectors" / "previous").write_text("the last good store")
    bucket.payloads[DETECTOR] = b"detectoR"
    with pytest.raises(ValueError, match=rf"{DETECTOR} is sha256 [0-9a-f]{{64}}, models.lock pins {expected[DETECTOR][1]}"):
        tee_models.fetch(bucket, jobs, target, expected)
    assert (target / "detectors" / "previous").exists(), "nothing of the refused store was put in place"
    assert not (target / ".partial").exists()
    # Another size says so before any hashing; a missing object fails the copy.
    bucket.payloads[DETECTOR] = b"detector, longer"
    with pytest.raises(ValueError, match=f"{DETECTOR} is 16 bytes, models.lock pins 8"):
        tee_models.fetch(bucket, jobs, target, expected)
    del bucket.objects[DETECTOR]
    with pytest.raises(FileNotFoundError):
        tee_models.fetch(bucket, jobs, target, expected)


def test_with_the_lock_exactly_its_objects_are_copied_whatever_else_the_bucket_holds(tmp_path, monkeypatch):
    payloads = {WEIGHTS: b"weights", DETECTOR: b"detector",
                "detectors/rtdetrv4/a-newer-model.safetensors": b"not in this release",
                "hf-bf16/hub/models--x/snapshots/a/extra.py": b"import os"}
    bucket = FakeBucket({name: len(p) for name, p in payloads.items()}, payloads)
    fake_storage(monkeypatch, bucket)
    lock = tmp_path / "models.lock"
    lock.write_text(lock_line(WEIGHTS, b"weights") + "\n" + lock_line(DETECTOR, b"detector") + "\n")
    target = tmp_path / "models"
    ready = target / ".complete"
    assert tee_models.main(["--bucket", "models", "--target", str(target),
                            "--models-lock", str(lock), "--ready-file", str(ready)]) == 0
    landed = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    assert landed == [".complete", DETECTOR, WEIGHTS]


def test_the_lock_is_written_from_what_the_prefixes_hold(tmp_path, monkeypatch):
    payloads = {WEIGHTS: b"weights", DETECTOR: b"detector",
                "hf-bf16/hub/models--x/blobs/deadbeef": b"a blob the boot never opens",
                "artifacts/a.json": b"elsewhere"}
    bucket = FakeBucket({name: len(p) for name, p in payloads.items()}, payloads)
    fake_storage(monkeypatch, bucket)
    lock = tmp_path / "models.lock"
    assert tee_models.main(["--bucket", "models", "--target", str(tmp_path / "unused"),
                            "--write-models-lock", str(lock)]) == 0
    assert tee_models.read_models_lock(lock) == {
        WEIGHTS: (7, hashlib.sha256(b"weights").hexdigest()),
        DETECTOR: (8, hashlib.sha256(b"detector").hexdigest()),
    }
    assert not (tmp_path / "unused").exists(), "writing the lock copies nothing"
