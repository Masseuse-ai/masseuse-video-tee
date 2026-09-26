"""Online pose at the 6fps cadence: the Tracker, streamed.

`pose_track.Tracker` is already an online machine - identity anchored to the
last accepted box, scenery boxes discarded - and this wrapper adds only what
a stream needs on top: the scenery set is learned from the stream's own
opening seconds, on quarter-scale grays so the warm-up history costs
megabytes rather than a gigabyte. Until the scenery is learned the tracker
runs with an empty set, which errs toward keeping candidates - identity
locking still holds the person, and a phantom recurring box costs a few
warm-up frames, not an event.

The sideload variant replays a production capture's `poses.jsonl` keyed by
decode frame: locally there is no GPU, and equivalence against the replay
harness wants pose held fixed so the plumbing is the only thing under test.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# pose_rows, keypoints and pose_track live beside this file (workload/pixel).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pose_rows  # noqa: E402 - torch-free: selection geometry and the row shape
from keypoints import KEYPOINT_NAMES  # noqa: E402

# Scenery is learned from this much footage, at the pose cadence.
SCENERY_WARMUP_S = 20.0
SCENERY_SCALE = 4  # quarter-scale grays for the motion-energy history

PREFETCH_CHUNK = 64 * 1024 * 1024
PREFETCH_WORKERS = 16

# The prefetch runs exactly once per process, whichever of the serve()-start
# background thread, /warmup, or the first session boot reaches it first.
# Failure leaves the flag unset so the next caller retries from scratch;
# partially written trees are never visible (copies land in a .partial
# sibling and are renamed only when complete).
_PREFETCH_LOCK = threading.Lock()
_PREFETCH_DONE = False


def share_full_result(on_full, telemetry, frame_index: int, at_s: float,
                      keypoints, scores, box, box_score, people: int,
                      unresolved: bool) -> None:
    """Hand the whole inference result - every keypoint the model emitted,
    not the 21 the row keeps - to an `on_full` listener (the live overlay).

    The row is what the pipeline and the equivalence gate consume; this is
    a side channel that must never alter it or stall the pose thread, so a
    listener that raises is counted and otherwise ignored.
    """
    if on_full is None:
        return
    try:
        on_full(frame_index, at_s, keypoints, scores, box, box_score,
                people, unresolved)
    except Exception as error:  # noqa: BLE001 - a listener must not stall the pose
        if telemetry is not None:
            telemetry.count("poseListenerErrors")
        print(f"pose listener failed at {at_s:.1f}s: {error!r}", flush=True)


def row_keypoint_arrays(keypoints: dict) -> tuple[np.ndarray, np.ndarray]:
    """A captured row's named body points back in index order: what the
    sideload can offer the overlay in place of the model's 308."""
    count = max(KEYPOINT_NAMES) + 1
    points = np.zeros((count, 2), np.float32)
    scores = np.zeros(count, np.float32)
    names = {name: index for index, name in KEYPOINT_NAMES.items()}
    for name, (x, y, score) in keypoints.items():
        index = names.get(name)
        if index is None:
            continue
        points[index] = (x, y)
        scores[index] = score
    return points, scores


MODEL_STORE_WAIT_S = 600.0


def wait_for_model_store(telemetry=None, timeout_s: float = MODEL_STORE_WAIT_S,
                         sleep=time.sleep, clock=time.monotonic) -> None:
    """Block until the model store is complete, when the boot runs alongside
    the copy that fills it.

    The Confidential Space entrypoint starts the producer before
    tee_models.py has finished so the torch import overlaps the weights, and
    tee_models.py touches POSE_MODELS_READY_FILE last. No-op unless that
    variable is set (Cloud Run had the mount, the local loop has the files).
    The timeout is a backstop: the entrypoint gives up on the copy, and the
    boot with it, long before.
    """
    marker = os.environ.get("POSE_MODELS_READY_FILE", "")
    if not marker:
        return
    started = clock()
    path = Path(marker)
    while not path.exists():
        if clock() - started > timeout_s:
            raise TimeoutError(
                f"model store not complete after {timeout_s:.0f}s: {marker} missing")
        sleep(0.2)
    if telemetry:
        telemetry.boot_phase("storeWait", started)


def prefetch_model_store(telemetry=None) -> None:
    """Copy the model store from the FUSE mount to local disk, exactly once.

    transformers mmap-loads safetensors, and random page faults through
    gcsfuse measured 28MB/s - 206s for the 5.7GB model - where sequential
    object reads are what Cloud Storage is fast at. Gated on POSE_PREFETCH=1
    (the local compose loop has no mount and no metadata server).
    """
    global _PREFETCH_DONE
    if os.environ.get("POSE_PREFETCH") != "1":
        return
    with _PREFETCH_LOCK:
        if _PREFETCH_DONE:
            return
        _run_prefetch(telemetry)
        _PREFETCH_DONE = True


def _prefetch_roots() -> list[tuple[Path, Path, str]]:
    """(source, target, bucket prefix) for every mount tree the boot reads.

    Roots are derived from the envs that point into the mount - the hub
    cache via HF_HOME and the detector checkpoint's top-level directory -
    so a config change cannot silently strand a model on the slow path.
    """
    mount = Path(os.environ.get("POSE_MODELS_MOUNT", "/models"))
    local = Path(os.environ.get("POSE_PREFETCH_TARGET", "/tmp"))
    candidates = [Path(os.environ.get("HF_HOME", "/models/hf"))]
    checkpoint = os.environ.get("POSE_DETECTOR_CHECKPOINT", "")
    if checkpoint and Path(checkpoint).is_relative_to(mount):
        candidates.append(
            mount / Path(checkpoint).relative_to(mount).parts[0])
    roots: list[tuple[Path, Path, str]] = []
    for source in dict.fromkeys(candidates):
        if not source.is_dir() or not source.is_relative_to(mount):
            continue
        prefix = source.relative_to(mount).as_posix()
        roots.append((source, local / prefix, prefix))
    return roots


def _prefetch_skip(relative: Path) -> bool:
    """What the cloud boot provably never reads.

    - blobs/: GCS stores no symlinks, so the uploaded snapshots hold real
      files and the hub cache resolves through snapshots/ alone (verified
      by loading offline from a blobs-free replica).
    - .locks/: hub download bookkeeping, meaningless on a read-only copy.
    - sapiens2_1b_pose.safetensors: an upload-time twin of
      model.safetensors under a name transformers never looks up.
    """
    if "blobs" in relative.parts or ".locks" in relative.parts:
        return True
    if relative.name == "sapiens2_1b_pose.safetensors":
        return True
    return False


def _repoint(env_name: str, copied: dict[Path, Path]) -> None:
    value = os.environ.get(env_name, "")
    if not value:
        return
    path = Path(value)
    for source, target in copied.items():
        if path.is_relative_to(source):
            os.environ[env_name] = str(target / path.relative_to(source))
            return


def _run_prefetch(telemetry) -> None:
    started = time.monotonic()
    roots = _prefetch_roots()
    if not roots:
        return
    copied: dict[Path, Path] = {}
    pending: list[tuple[Path, Path, Path]] = []  # partial, target, source
    jobs: list[tuple[Path, Path, str, int]] = []  # source, dest, object, size
    skipped_bytes = 0
    for source, target, prefix in roots:
        if target.exists():
            copied[source] = target
            continue
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(source)
            size = path.stat().st_size
            if _prefetch_skip(relative):
                skipped_bytes += size
                continue
            dest = partial / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((path, dest, f"{prefix}/{relative.as_posix()}", size))
        pending.append((partial, target, source))
    mode = _fetch_all(jobs) if jobs else "cached"
    for partial, target, source in pending:
        os.replace(partial, target)
        copied[source] = target
    total = sum(size for _, _, _, size in jobs)
    elapsed = max(time.monotonic() - started, 1e-6)
    mbps = total / 1e6 / elapsed
    print(f"prefetch: mode={mode} files={len(jobs)} bytes={total} "
          f"skippedBytes={skipped_bytes} seconds={elapsed:.1f} "
          f"MBps={mbps:.1f}", flush=True)
    _repoint("HF_HOME", copied)
    _repoint("POSE_DETECTOR_CHECKPOINT", copied)
    if telemetry:
        telemetry.boot_phase("prefetch", started)
        telemetry.gauge("prefetchMBps", round(mbps, 1))


def _fetch_all(jobs: list[tuple[Path, Path, str, int]]) -> str:
    """The transport ladder, most parallel first, each rung said aloud.

    POSE_PREFETCH_TRANSPORT pins a rung for A/B probes: 'auto' (default),
    'api' (threaded ranged reads), or 'fuse' (parallel mount copies).
    """
    transport = os.environ.get("POSE_PREFETCH_TRANSPORT", "auto")
    bucket_name = os.environ.get("POSE_PREFETCH_BUCKET", "")
    if transport == "auto" and bucket_name:
        try:
            return _fetch_transfer_manager(jobs, bucket_name)
        except Exception as error:  # noqa: BLE001 - fall down the ladder
            print(f"prefetch: transfer-manager unavailable ({error!r}), "
                  f"falling back to threaded ranges", flush=True)
    if transport in ("auto", "api") and bucket_name:
        token = _metadata_token()
        if token is not None:
            _fetch_threaded_api(jobs, bucket_name, token)
            return "api"
    _fetch_fuse(jobs)
    return "fuse"


def _fetch_transfer_manager(jobs, bucket_name: str) -> str:
    """google-cloud-storage's sliced parallel download, chunk per process.

    Process workers keep TLS decryption and checksumming off the producer's
    GIL, and 'spawn' children cannot inherit locks held mid-import by the
    concurrent torch preload thread.
    """
    import multiprocessing

    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    multiprocessing.set_start_method("spawn", force=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for _, dest, name, size in jobs:
        if size <= PREFETCH_CHUNK:
            continue
        blob = bucket.get_blob(name)
        if blob is None:
            raise FileNotFoundError(f"gs://{bucket_name}/{name}")
        transfer_manager.download_chunks_concurrently(
            blob, str(dest), chunk_size=PREFETCH_CHUNK,
            max_workers=PREFETCH_WORKERS)
    small = [job for job in jobs if job[3] <= PREFETCH_CHUNK]
    if small:
        from concurrent.futures import ThreadPoolExecutor

        def fetch(job) -> None:
            _, dest, name, _ = job
            bucket.blob(name).download_to_filename(str(dest))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, small))
    return "transfer-manager"


def _metadata_token() -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/"
                "instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"}),
                timeout=5) as response:
            return json.loads(response.read())["access_token"]
    except Exception as error:  # noqa: BLE001 - fallback, said aloud
        print(f"prefetch: metadata token failed ({error!r}), "
              f"falling back to the mount", flush=True)
        return None


def _fetch_threaded_api(jobs, bucket_name: str, token: str) -> None:
    """Ranged reads over the JSON API in one thread pool - the pre-library
    transport, kept as the measured fallback."""
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    ranged: list[tuple[Path, Path, str, int, int]] = []
    for path, dest, name, size in jobs:
        if size <= PREFETCH_CHUNK:
            ranged.append((path, dest, name, -1, size))
            continue
        with open(dest, "wb") as handle:
            handle.truncate(size)
        ranged.extend((path, dest, name, offset, size)
                      for offset in range(0, size, PREFETCH_CHUNK))

    def fetch(job: tuple[Path, Path, str, int, int]) -> None:
        path, dest, name, offset, size = job
        if offset < 0:
            shutil.copyfile(path, dest)
            return
        try:
            quoted = urllib.parse.quote(name, safe="")
            end = min(size, offset + PREFETCH_CHUNK) - 1
            request = urllib.request.Request(
                f"https://storage.googleapis.com/download/storage/v1/b/"
                f"{bucket_name}/o/{quoted}?alt=media",
                headers={"Authorization": f"Bearer {token}",
                         "Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001 - per-range fallback
            print(f"prefetch: range via api failed ({error!r}), reading "
                  f"the mount for {path.name}@{offset}", flush=True)
            with open(path, "rb") as src:
                src.seek(offset)
                data = src.read(PREFETCH_CHUNK)
        with open(dest, "r+b") as handle:
            handle.seek(offset)
            handle.write(data)

    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        list(pool.map(fetch, ranged))


def _fetch_fuse(jobs) -> None:
    """Parallel whole-file copies through the mount: ~19MB/s per stream,
    scaling with concurrent files - the floor every rung can fall back to."""
    from concurrent.futures import ThreadPoolExecutor

    def fetch(job) -> None:
        path, dest, _, _ = job
        shutil.copyfile(path, dest)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))


class GpuPose:
    """RT-DETRv4-X + Sapiens2-1B in bf16, one frame at a time.

    Sapiens2 loads with bf16 weights and every forward runs under bf16
    autocast (the detector keeps fp32 master weights; autocast picks its
    kernels). fp32 weights plus autocast recast the 1B model on every
    forward, and on the RTX PRO 6000 that overhead was the difference
    between missing and making the 167ms 6fps budget, so there is no other
    dtype. The remaining knobs default to the production configuration; an
    unset revision boots exactly what service.yaml describes:

      POSE_PREFETCH=1   copy the model store (hub cache + detector
                        checkpoints) from the FUSE mount to local disk
                        before loading - see prefetch_model_store. With
                        POSE_PREFETCH_BUCKET set the bytes bypass the mount
                        entirely; POSE_PREFETCH_TRANSPORT pins a transport
                        rung for A/B probes. Off by default (the local
                        compose loop has no mount).
      POSE_DETECT_STRIDE  run the person detector on every Nth pose frame
                        and pose the cached box in between. The fixed camera
                        and braced prone body move the box slowly; detect at
                        33ms per frame (measured, Blackwell) is a fifth of
                        the 6fps budget spent re-finding a box that has not
                        moved. A lost person always re-detects on the next
                        frame. Default 3.
      POSE_FLIP_TTA=1   the flip-TTA pair instead of a single forward. With
                        TTA the Blackwell worker ran at ~93% of the 6fps
                        budget and queue bursts dropped ~11% of poses -
                        enough clustering to blank the readings the analysis
                        derives in the exact windows the consumer votes on.
                        Default 0: half
                        the inference buys the margin, and the keypoint
                        delta is bounded by the file-mode equivalence gate.
      POSE_CUDA_GRAPHS=0  run the forwards eagerly instead of as captured
                        CUDA graphs (gpu_graph). Debugging only: eager is
                        launch-bound on a Confidential Computing GPU.
      POSE_GRAPH_BENCH=1,2,4,8  after the parity check, capture the pose
                        forward over batches of that many crops and print
                        `poseBench` lines comparing each with the batch of
                        one (pose_bench). Debug slots only: the launch
                        policy admits it, nothing in production sets it,
                        and the graphs it captures are gone before the
                        slot serves.

    Each frame is uploaded once as a CHW uint8 tensor; the detector's
    resize, the pose crop, both forwards and their post-processing run on
    the device, and each model hands back one small tensor. At boot the
    captured graphs are checked against their eager twins on a synthetic
    frame (`check_parity`): a model whose graph disagrees runs eagerly.
    """

    # The graph's keypoints may differ from the eager forward's by this much
    # (image pixels, and score) before the graph is distrusted. bf16 kernels
    # replayed are the same kernels; anything beyond rounding is a fault.
    PARITY_PX = 0.5
    PARITY_SCORE = 0.01

    def __init__(self, device: str = "cuda", model: str | None = None,
                 flip: bool | None = None, telemetry=None):
        self.device = device
        self.model = model
        self.flip = (os.environ.get("POSE_FLIP_TTA", "0") == "1"
                     if flip is None else flip)
        self.telemetry = telemetry
        self.tracker = None
        # The live overlay's tap on the full result (see share_full_result);
        # a session sets it for its duration and clears it after.
        self.on_full = None
        self._warmup: list[tuple[list[list[float]], np.ndarray]] = []
        self._scenery_done = False
        self._detect_stride = max(
            1, int(os.environ.get("POSE_DETECT_STRIDE", "3") or "3"))
        self._detect_countdown = 0
        self._cached_detection: tuple | None = None

    def reset_session_state(self) -> None:
        """Forget the previous session's scene, keep the booted weights.

        The model is session-independent; scenery, the identity anchor and
        the detect cache are not. Clearing them is what lets one booted
        instance serve consecutive sessions without minutes of reload.
        """
        self._warmup.clear()
        self._scenery_done = False
        self._cached_detection = None
        self._detect_countdown = 0
        if self.tracker is not None:
            self.tracker.previous = None
            self.tracker.scenery = []

    def boot(self) -> None:
        prefetch_model_store(self.telemetry)
        started = time.monotonic()
        import pose_track
        import torch
        if self.telemetry:
            self.telemetry.boot_phase("import", started)
        # The weights may still be landing (the enclave copies them alongside
        # this boot); everything up to here needed none of them.
        wait_for_model_store(self.telemetry)
        started = time.monotonic()
        # bf16 weights for Sapiens2, bf16 autocast for both forwards: the
        # production numeric context, which the tracker applies itself so the
        # captured graphs and their eager twins run the same kernels.
        self.tracker = pose_track.Tracker(
            self.device, self.model or pose_track.POSE_MODEL, self.flip,
            dtype=torch.bfloat16, autocast_dtype=torch.bfloat16)
        if self.telemetry:
            self.telemetry.boot_phase("weights", started)
            backend = getattr(self.tracker, "detector_backend", None)
            timings = getattr(backend, "boot_seconds", None) or {}
            for name, seconds in timings.items():
                self.telemetry.boot_seconds(name, seconds)
        started = time.monotonic()
        # Warm up, capture both graphs (each falls back to eager on its own),
        # then the first inference through the captured path against the
        # eager twin on a synthetic frame.
        self.tracker.capture(self.telemetry)
        self.check_parity()
        self.bench_batches(os.environ.get("POSE_GRAPH_BENCH"))
        if self.telemetry:
            self.telemetry.boot_phase("firstInference", started)
        self._log_gpu()

    def bench_batches(self, spec: str | None) -> dict | None:
        """The boot-time batch bench when `POSE_GRAPH_BENCH` names batch
        sizes (see `pose_bench`); nothing otherwise. A measurement must not
        cost a boot, so a failure inside it is printed and counted
        (`poseBenchFailed`) and the slot goes on to serve."""
        import pose_bench

        batches = pose_bench.parse_batches(spec)
        if not batches:
            return None
        try:
            return pose_bench.run(
                self.tracker, self.parity_frame(self.device), batches,
                telemetry=self.telemetry)
        except Exception as error:  # noqa: BLE001 - a bench, not the service
            print(f"poseBench: failed: {error!r}", flush=True)
            if self.telemetry:
                self.telemetry.count("poseBenchFailed")
            return None

    @staticmethod
    def parity_frame(device: str, size: tuple[int, int] = (360, 640)):
        """A synthetic CHW uint8 frame with some structure - a colour
        gradient and a brighter block - so the detector's top queries and
        the pose heatmaps are not degenerate. What it shows does not matter;
        both paths see the identical tensor."""
        import torch

        height, width = size
        rows = torch.linspace(0, 255, height).view(height, 1).expand(height, width)
        cols = torch.linspace(0, 255, width).view(1, width).expand(height, width)
        frame = torch.stack([rows, cols, (rows + cols) / 2]).round().to(torch.uint8)
        frame[:, height // 4: 3 * height // 4, width // 3: 2 * width // 3] = 200
        return frame.to(device)

    def check_parity(self) -> dict[str, float]:
        """Each captured graph against its eager twin on one synthetic frame.

        Replaying the same kernels over the same input must reproduce the
        eager result to rounding; a graph that does not (a stale buffer, a
        mis-wired static input) is put back to eager for that model, counted
        as `graphParityFailed`, and the slot serves at the old speed rather
        than with wrong keypoints. The deltas are printed and reported as
        gauges so a debug slot's log and the readings show which path runs.
        """
        import gpu_graph
        import torch

        tracker = self.tracker
        detector = tracker.detector_backend
        frame = self.parity_frame(self.device)
        height, width = frame.shape[-2:]
        box = [width / 3, height / 4, width / 3, height / 2]
        report: dict[str, float] = {}

        def demote(name: str) -> None:
            if self.telemetry:
                self.telemetry.count("graphParityFailed")
            print(f"{name}: graph disagrees with eager, running eagerly",
                  flush=True)

        pixels = detector.pixels(frame)
        if detector.graph.graphed:
            graphed = detector.graph.replay(pixels).clone()
            eager = detector.forward(pixels)
            # Top queries are rows in score order; two with near-equal
            # scores may swap between runs, so the boxes (with the label as
            # a coordinate a mismatch cannot hide in) are matched nearest
            # to nearest and the scores compared sorted.
            scale = torch.tensor(
                [width, height, width, height, max(width, height)],
                dtype=torch.float32, device=graphed.device)
            geometry = lambda rows: rows[:, [0, 1, 2, 3, 5]] * scale  # noqa: E731
            # Largest coordinate difference between each row and its nearest
            # counterpart, either way round (300x300x5: nothing to a GPU).
            distance = (geometry(graphed)[:, None, :]
                        - geometry(eager)[None, :, :]).abs().amax(dim=-1)
            report["detectParityPx"] = float(max(
                distance.amin(dim=1).amax(), distance.amin(dim=0).amax()))
            report["detectParityScore"] = float((
                graphed[:, 4].sort(descending=True).values
                - eager[:, 4].sort(descending=True).values).abs().max())
            if (report["detectParityPx"] > self.PARITY_PX
                    or report["detectParityScore"] > self.PARITY_SCORE):
                detector.graph = gpu_graph.Eager(
                    detector.forward, detector.graph.static_inputs)
                demote("detector")
        else:
            detector.graph.replay(pixels)  # eager: the warmup the boot always did

        crop = tracker.pose_crop(frame, box)
        box_tensor = torch.tensor(box, dtype=torch.float32, device=self.device)
        if tracker.pose_graph.graphed:
            graphed = tracker.pose_graph.replay(crop, box_tensor).clone()
            eager = tracker.pose_forward(crop, box_tensor)
            report["poseParityPx"] = float(
                (graphed[:, :2] - eager[:, :2]).abs().max())
            report["poseParityScore"] = float(
                (graphed[:, 2] - eager[:, 2]).abs().max())
            if (report["poseParityPx"] > self.PARITY_PX
                    or report["poseParityScore"] > self.PARITY_SCORE):
                tracker.pose_graph = gpu_graph.Eager(
                    tracker.pose_forward, tracker.pose_graph.static_inputs)
                demote("pose")
        else:
            tracker.pose_graph.replay(crop, box_tensor)

        on = {"detectGraph": detector.graph.graphed,
              "poseGraph": tracker.pose_graph.graphed}
        print(" ".join(
            [f"{name}={'on' if value else 'off'}" for name, value in on.items()]
            + [f"{name}={value:.4f}" for name, value in report.items()]),
            flush=True)
        if self.telemetry:
            for name, value in on.items():
                self.telemetry.gauge(name, 1.0 if value else 0.0)
            for name, value in report.items():
                self.telemetry.gauge(name, round(value, 4))
        return report

    def _log_gpu(self) -> None:
        """The driver and device, measured rather than assumed - the deploy
        leans on the 580-series driver and this is where that is checked."""
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            print(f"gpu: {out.stdout.strip()}", flush=True)
        except (OSError, subprocess.TimeoutExpired):
            print("gpu: nvidia-smi unavailable", flush=True)

    def _learn_scenery(self, at_s: float) -> None:
        if self._scenery_done or at_s < SCENERY_WARMUP_S:
            return
        per_frame = [boxes for boxes, _ in self._warmup]
        grays = [gray for _, gray in self._warmup]

        def motion(box: list[float]) -> float:
            x0, y0, x1, y1 = (int(v / SCENERY_SCALE) for v in box)
            deltas = [
                float(np.abs(after[y0:y1, x0:x1].astype(np.int16)
                             - before[y0:y1, x0:x1].astype(np.int16)).mean())
                for before, after in zip(grays, grays[1:])
                if after[y0:y1, x0:x1].size
            ]
            return float(np.mean(deltas)) if deltas else 0.0

        self.tracker.scenery = pose_rows.find_scenery(per_frame, motion)
        self._scenery_done = True
        self._warmup.clear()

    def step(self, rgb: np.ndarray, frame_index: int, at_s: float) -> dict:
        tracker = self.tracker
        started = time.monotonic()
        # The frame's one trip to the device; everything below reads it there.
        frame = tracker.frame_tensor(rgb, self.device)
        if self.telemetry:
            self.telemetry.observe("frameUpload", time.monotonic() - started)
        if not self._scenery_done:
            # Scenery learning needs every person candidate, before identity
            # selection; the detector backend owns preprocessing and label
            # mapping, so the candidates come through it.
            candidates = tracker.detection_candidates(frame)
            boxes = [box for box, _ in candidates]
            small = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (rgb.shape[1] // SCENERY_SCALE,
                 rgb.shape[0] // SCENERY_SCALE))
            self._warmup.append((boxes, small))
            self._learn_scenery(at_s)

        cached = self._cached_detection
        if (cached is not None and cached[0] is not None
                and self._detect_countdown > 0):
            self._detect_countdown -= 1
            box, score, people, unresolved = cached
        else:
            started = time.monotonic()
            box, score, people, unresolved = tracker.detect(frame)
            if self.telemetry:
                # Cache hits are not observed; the stage reports what a real
                # detect costs, not an average diluted by reuse. The copy
                # back is inside, so this is the latency, not the enqueue.
                self.telemetry.observe("detect", time.monotonic() - started)
            self._cached_detection = (box, score, people, unresolved)
            self._detect_countdown = self._detect_stride - 1
        if box is None:
            share_full_result(self.on_full, self.telemetry, frame_index, at_s,
                              None, None, None, None, people, unresolved)
            return pose_rows.missing_row(frame_index, at_s)
        started = time.monotonic()
        # (K, 3) x, y, score in image pixels, already on the host: the row
        # builder and the overlay read plain numpy from here on.
        result = tracker.pose_keypoints(frame, box)
        if self.telemetry:
            self.telemetry.observe("poseInfer", time.monotonic() - started)
        keypoints, scores = result[:, :2], result[:, 2]
        share_full_result(self.on_full, self.telemetry, frame_index, at_s,
                          keypoints, scores, box, score, people, unresolved)
        return pose_rows.row_for(frame_index, at_s, keypoints, scores, box,
                                 score, people, unresolved)


class SideloadPose:
    """Poses replayed from a production capture's `poses.jsonl`.

    The capture's rows are keyed by the same decode-cadence `frame` the
    producer steps in (the pose worker wrote them), so the lookup is direct.
    When the stream loops past the end of the capture, the lookup wraps by
    the capture's decode span, so an endless mediamtx loop replays the same
    poses each lap. A row the worker dropped or lost replays as keypoint-less,
    exactly as it was on the day.
    """

    def __init__(self, capture: Path, telemetry=None):
        started = time.monotonic()
        path = Path(capture)
        if path.is_dir():
            path = path / "poses.jsonl"
        self.rows: dict[int, dict] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                self.rows[int(row["frame"])] = row
        frames = sorted(self.rows)
        self.stride = min(
            (b - a for a, b in zip(frames, frames[1:]) if b > a), default=1)
        self.span = (frames[-1] + self.stride) if frames else 1
        self.telemetry = telemetry
        # Same tap as GpuPose, fed the row's 21 body points: the local loop
        # renders the overlay without a GPU (hands and face stay empty).
        self.on_full = None
        if telemetry:
            telemetry.boot_phase("weights", started)

    def boot(self) -> None:
        return None

    def step(self, rgb: np.ndarray, frame_index: int, at_s: float) -> dict:
        key = frame_index % self.span
        row = self.rows.get(key)
        if row is None:
            # A stride the capture did not use: the nearest row that is
            # within half a pose interval stands in.
            nearest = min(self.rows, key=lambda f: abs(f - key), default=None)
            if nearest is not None and abs(nearest - key) * 2 <= self.stride:
                row = self.rows[nearest]
        if row is None or not row.get("keypoints"):
            share_full_result(self.on_full, self.telemetry, frame_index, at_s,
                              None, None, None, None,
                              (row or {}).get("people", 0),
                              (row or {}).get("identityUnresolved", False))
            return {"frame": frame_index, "atS": round(at_s, 4),
                    "keypoints": None}
        if self.on_full is not None:
            points, scores = row_keypoint_arrays(row["keypoints"])
            share_full_result(self.on_full, self.telemetry, frame_index, at_s,
                              points, scores, row.get("box"),
                              row.get("boxScore"), row.get("people", 1),
                              row.get("identityUnresolved", False))
        return {"frame": frame_index, "atS": round(at_s, 4),
                "keypoints": row["keypoints"]}
