"""The model-store prefetch: what leaves the mount, and what never does.

These are the cold-boot mechanics the cloud probes lean on: the skip rules
that keep 12GB of duplicate bytes off the wire, the atomic .partial rename
that makes a crashed copy invisible, the env repointing that swings loads
onto local disk, and the exactly-once guard that lets serve(), /warmup and
the first session all call it without duplicating work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import live_pose  # noqa: E402


def build_mount(root: Path) -> dict[str, Path]:
    """The bucket layout as the FUSE mount presents it, miniaturized."""
    sapiens = root / "hf/hub/models--facebook--sapiens2-pose-1b"
    snapshot = sapiens / "snapshots/f5fed8b"
    detector = root / "detectors/rtdetrv4"
    for directory in (sapiens / "blobs", snapshot, sapiens / "refs",
                      sapiens / ".locks", detector):
        directory.mkdir(parents=True, exist_ok=True)
    files = {
        "blob": sapiens / "blobs/2dab7014",
        "model": snapshot / "model.safetensors",
        "twin": snapshot / "sapiens2_1b_pose.safetensors",
        "config": snapshot / "config.json",
        "ref": sapiens / "refs/main",
        "lock": sapiens / ".locks/x.lock",
        "ckpt": detector / "rtv4_ema.safetensors",
    }
    for name, path in files.items():
        path.write_bytes(name.encode() * 4)
    return files


def prefetch_env(monkeypatch, mount: Path, target: Path) -> None:
    monkeypatch.setenv("POSE_PREFETCH", "1")
    monkeypatch.setenv("POSE_MODELS_MOUNT", str(mount))
    monkeypatch.setenv("POSE_PREFETCH_TARGET", str(target))
    monkeypatch.setenv("HF_HOME", str(mount / "hf"))
    monkeypatch.setenv(
        "POSE_DETECTOR_CHECKPOINT",
        str(mount / "detectors/rtdetrv4/rtv4_ema.safetensors"))
    monkeypatch.setenv("POSE_PREFETCH_TRANSPORT", "fuse")
    monkeypatch.delenv("POSE_PREFETCH_BUCKET", raising=False)
    monkeypatch.setattr(live_pose, "_PREFETCH_DONE", False)


def test_prefetch_copies_the_needed_files_and_repoints_the_envs(
        tmp_path, monkeypatch):
    mount, target = tmp_path / "models", tmp_path / "local"
    build_mount(mount)
    prefetch_env(monkeypatch, mount, target)

    live_pose.prefetch_model_store()

    snapshot = target / "hf/hub/models--facebook--sapiens2-pose-1b"
    assert (snapshot / "snapshots/f5fed8b/model.safetensors").is_file()
    assert (snapshot / "snapshots/f5fed8b/config.json").is_file()
    assert (snapshot / "refs/main").is_file()
    assert (target / "detectors/rtdetrv4/rtv4_ema.safetensors").is_file()
    import os
    assert os.environ["HF_HOME"] == str(target / "hf")
    assert os.environ["POSE_DETECTOR_CHECKPOINT"] == str(
        target / "detectors/rtdetrv4/rtv4_ema.safetensors")
    assert not list(target.glob("*.partial"))


def test_prefetch_never_moves_the_dead_bytes(tmp_path, monkeypatch):
    """blobs, the upload twin and lock files stay put."""
    mount, target = tmp_path / "models", tmp_path / "local"
    build_mount(mount)
    prefetch_env(monkeypatch, mount, target)

    live_pose.prefetch_model_store()

    copied = {path.name for path in (target / "hf").rglob("*")
              if path.is_file()}
    assert "2dab7014" not in copied
    assert "sapiens2_1b_pose.safetensors" not in copied
    assert "x.lock" not in copied


def test_prefetch_runs_exactly_once_and_retries_after_failure(
        tmp_path, monkeypatch):
    mount, target = tmp_path / "models", tmp_path / "local"
    files = build_mount(mount)
    prefetch_env(monkeypatch, mount, target)

    calls = {"n": 0}
    real = live_pose._fetch_all

    def flaky(jobs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("mid-copy crash")
        return real(jobs)

    monkeypatch.setattr(live_pose, "_fetch_all", flaky)

    try:
        live_pose.prefetch_model_store()
    except OSError:
        pass
    # The crash leaves no half-visible tree, and the guard stays open.
    assert not (target / "hf").exists()

    live_pose.prefetch_model_store()
    assert (target / "hf/hub/models--facebook--sapiens2-pose-1b/"
            "snapshots/f5fed8b/model.safetensors").read_bytes() \
        == files["model"].read_bytes()

    live_pose.prefetch_model_store()  # third call: cached no-op
    assert calls["n"] == 2
