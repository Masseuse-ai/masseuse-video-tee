"""The enclave boot's two slow halves run side by side: the producer starts
before the weights land and waits on the marker tee_models.py touches last
(live_pose.wait_for_model_store, POSE_MODELS_READY_FILE); the GPU boot is
started by the server itself (start_warmup, --prewarm, implied by --tee)
rather than by the trainer's first /warmup, and /warmup joins it."""

from __future__ import annotations

import json
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import live_pose  # noqa: E402
import producer  # noqa: E402
import tee_models  # noqa: E402
from telemetry import Telemetry  # noqa: E402
from test_tee_mode import request, server_args  # noqa: E402
from test_tee_models import FakeBucket  # noqa: E402


def test_wait_is_a_no_op_without_the_marker_variable(monkeypatch):
    monkeypatch.delenv("POSE_MODELS_READY_FILE", raising=False)
    slept: list[float] = []
    live_pose.wait_for_model_store(sleep=slept.append)
    assert slept == []


def test_wait_returns_once_the_marker_appears_and_reports_the_phase(tmp_path, monkeypatch):
    marker = tmp_path / "models" / ".complete"
    monkeypatch.setenv("POSE_MODELS_READY_FILE", str(marker))
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        if len(slept) == 3:
            marker.parent.mkdir(parents=True)
            marker.touch()

    telemetry = Telemetry()
    live_pose.wait_for_model_store(telemetry, sleep=sleep)
    assert len(slept) == 3
    assert "storeWait" in telemetry.snapshot()["bootMs"]


def test_wait_gives_up_after_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("POSE_MODELS_READY_FILE", str(tmp_path / "never"))
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(s: float) -> None:
        now[0] += 100.0

    with pytest.raises(TimeoutError, match="never"):
        live_pose.wait_for_model_store(timeout_s=250.0, sleep=sleep, clock=clock)


def test_tee_models_clears_a_stale_marker_first_and_touches_it_last(tmp_path, monkeypatch):
    bucket = FakeBucket({"detectors/rtdetrv4/x.safetensors": 10,
                         "artifacts/a.json": 5})
    target = tmp_path / "models"
    marker = target / ".complete"
    target.mkdir()
    marker.touch()
    monkeypatch.setattr(tee_models, "CHUNK", 1024)
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *a, **k: None)
    seen: list[str] = []
    original_fetch = tee_models.fetch

    def fetch(bucket, jobs, target):
        seen.append("stale marker present" if marker.exists() else "marker cleared")
        original_fetch(bucket, jobs, target)

    monkeypatch.setattr(tee_models, "fetch", fetch)
    fake_tm = SimpleNamespace(download_chunks_concurrently=lambda *a, **k: None)
    storage = SimpleNamespace(
        transfer_manager=fake_tm,
        Client=lambda project=None: SimpleNamespace(bucket=lambda name: bucket))
    monkeypatch.setitem(sys.modules, "google.cloud.storage.transfer_manager", fake_tm)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(storage=storage))
    code = tee_models.main(["--bucket", "models", "--target", str(target),
                            "--prefixes", "detectors/,artifacts/a.json",
                            "--ready-file", str(marker)])
    assert code == 0
    assert seen == ["marker cleared"]
    assert marker.exists()
    assert (target / "artifacts" / "a.json").exists()


@pytest.fixture
def fake_gpu_boot(monkeypatch):
    """acquire_gpu_pose that takes a moment and lands a pose, recording
    every call; the process-wide slot is left as it was found."""
    calls: list[float] = []
    gate = threading.Event()

    def acquire(args, telemetry):
        calls.append(time.monotonic())
        gate.wait(5.0)
        producer._GPU_POSE["pose"] = object()
        return producer._GPU_POSE["pose"]

    monkeypatch.setattr(producer, "acquire_gpu_pose", acquire)
    saved = producer._GPU_POSE["pose"]
    producer._GPU_POSE["pose"] = None
    try:
        yield SimpleNamespace(calls=calls, release=gate.set)
    finally:
        producer._GPU_POSE["pose"] = saved


def test_start_warmup_boots_once_and_warmup_joins_it(fake_gpu_boot):
    server = producer.build_server(server_args(tee=False), Telemetry(), tee=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert server.start_warmup() is True
        assert server.start_warmup() is False          # already booting
        code, _, body = request(port, "GET", "/warmup")
        assert code == 200
        assert json.loads(body)["status"] == "warming"
        fake_gpu_boot.release()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            code, _, body = request(port, "GET", "/warmup")
            if json.loads(body)["status"] == "ready":
                break
            time.sleep(0.02)
        assert json.loads(body)["status"] == "ready"
        assert server.start_warmup() is False          # booted: nothing to do
        assert len(fake_gpu_boot.calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_tee_implies_prewarm_and_the_flag_stands_alone():
    parser = producer.build_parser()
    base = ["--analysis-socket", ""]
    assert parser.parse_args([*base, "--prewarm"]).prewarm is True
    assert parser.parse_args([*base, "--serve"]).prewarm is False
    # --tee implies it in serve(); the flag itself stays False so the
    # implication lives in one place.
    assert parser.parse_args([*base, "--tee"]).prewarm is False
