"""analysis.lock parsing and the bundle's verify-then-unpack path in
tee_models.py: a bundle whose digest is not the lock's never lands."""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import tee_models  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_repository_lock_parses_and_names_a_bundle():
    lock = tee_models.read_lock(REPO / "analysis.lock")
    assert lock["object"].startswith("analysis/analysis-")
    assert lock["object"].endswith(".tar.zst")
    assert lock["version"] in lock["object"]
    assert len(lock["sha256"]) == 64


def test_lock_rejects_missing_or_malformed_fields(tmp_path):
    bad = tmp_path / "analysis.lock"
    bad.write_text("version=1\nobject=analysis/x.tar.zst\n")
    with pytest.raises(ValueError):
        tee_models.read_lock(bad)
    bad.write_text("version=1\nobject=analysis/x.tar.zst\nsha256=abc\n")
    with pytest.raises(ValueError):
        tee_models.read_lock(bad)
    good = tmp_path / "ok.lock"
    good.write_text("# comment\nversion=1\nobject=analysis/x.tar.zst\n"
                    f"sha256={'A' * 64}\n")
    assert tee_models.read_lock(good)["sha256"] == "a" * 64


def _bundle(tmp_path: Path, main_text: str = "print('hi')\n") -> Path:
    root = tmp_path / "stage" / "analysis"
    (root / "artifacts").mkdir(parents=True)
    (root / "main.py").write_text(main_text)
    (root / "artifacts" / "r.json").write_text("{}")
    tar_path = tmp_path / "b.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(root, arcname="analysis")
    zst_path = tmp_path / "b.tar.zst"
    subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(zst_path)],
                   check=True)
    return zst_path


class _Blob:
    def __init__(self, path: Path):
        self.path = path

    def download_to_filename(self, name: str) -> None:
        shutil.copy(self.path, name)


class _Bucket:
    name = "models"

    def __init__(self, objects: dict[str, Path]):
        self.objects = objects

    def get_blob(self, name: str):
        return _Blob(self.objects[name]) if name in self.objects else None


@pytest.mark.skipif(shutil.which("zstd") is None or shutil.which("tar") is None,
                    reason="needs zstd and tar")
def test_matching_digest_unpacks_world_readable(tmp_path):
    bundle = _bundle(tmp_path)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    lock = {"version": "v1", "object": "analysis/analysis-v1.tar.zst",
            "sha256": digest}
    target = tmp_path / "models"
    target.mkdir()
    where = tee_models.fetch_analysis(
        _Bucket({lock["object"]: bundle}), lock, target)
    assert where == target / "analysis"
    assert (where / "main.py").read_text() == "print('hi')\n"
    assert (where / "VERSION").read_text().strip() == "v1"
    assert (where / "artifacts" / "r.json").is_file()
    assert oct((where / "main.py").stat().st_mode & 0o777) == "0o644"
    assert oct(where.stat().st_mode & 0o777) == "0o755"
    assert not (target / ".partial-analysis").exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="needs zstd")
def test_wrong_digest_is_refused_and_nothing_lands(tmp_path):
    bundle = _bundle(tmp_path)
    lock = {"version": "v1", "object": "analysis/analysis-v1.tar.zst",
            "sha256": "0" * 64}
    target = tmp_path / "models"
    target.mkdir()
    with pytest.raises(ValueError, match="refusing"):
        tee_models.fetch_analysis(_Bucket({lock["object"]: bundle}), lock, target)
    assert not (target / "analysis").exists()
    assert not list((target / ".partial-analysis").glob("*")) \
        if (target / ".partial-analysis").exists() else True


def test_missing_object_is_reported(tmp_path):
    lock = {"version": "v1", "object": "analysis/none.tar.zst", "sha256": "0" * 64}
    target = tmp_path / "models"
    target.mkdir()
    with pytest.raises(FileNotFoundError):
        tee_models.fetch_analysis(_Bucket({}), lock, target)


def test_default_prefixes_carry_no_object_but_weights():
    # The bundle is named by the lock, never by a prefix; the weights are.
    assert all(prefix.endswith("/") for prefix in tee_models.DEFAULT_PREFIXES)
    buffer = io.StringIO()
    buffer.write(",".join(tee_models.DEFAULT_PREFIXES))
    assert "analysis" not in buffer.getvalue()
