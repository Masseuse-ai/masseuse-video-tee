"""The mount-free model copy: which objects the enclave pulls into its
/models tmpfs, and that whole prefixes appear at once."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import tee_models  # noqa: E402


class FakeBlob:
    def __init__(self, bucket, name, size, payload=b""):
        self.bucket = bucket
        self.name = name
        self.size = size
        self.payload = payload or name.encode()

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.payload)


class FakeBucket:
    name = "models"

    def __init__(self, objects: dict[str, int]):
        self.objects = objects

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, name, size) for name, size in sorted(self.objects.items())
                if name.startswith(prefix)]

    def get_blob(self, name):
        return FakeBlob(self, name, self.objects[name]) if name in self.objects else None

    def blob(self, name):
        return FakeBlob(self, name, self.objects.get(name, 0))


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
