"""Online pose at the producer's cadence: the Tracker, streamed.

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


# The session's primary stream's view; a second stream's is named by the
# session (the producer calls the phone's, beside a fixed camera, "face").
BODY_VIEW = "body"


class ViewState:
    """What the pose keeps between one stream's frames.

    Scenery is learnt from the first seconds of a stream, the identity
    anchor follows one person through it, and the detector's box is reused
    for a few frames: all of it is about one camera's picture. A session
    with two cameras poses both through the one booted model, so each view
    has its own of these and the model sees them one at a time.
    """

    def __init__(self, on_full=None):
        # The live overlay's tap on this view's full result
        # (share_full_result).
        self.on_full = on_full
        self.warmup: list[tuple[list[list[float]], np.ndarray]] = []
        self.scenery_done = False
        self.detect_countdown = 0
        self.cached_detection: tuple | None = None
        # The tracker's identity anchor and scenery while this view is not
        # the one on the tracker (GpuPose.step swaps them in and out).
        self.previous: list[float] | None = None
        self.scenery: list[list[float]] = []

    def reset(self) -> None:
        self.warmup.clear()
        self.scenery_done = False
        self.detect_countdown = 0
        self.cached_detection = None
        self.previous = None
        self.scenery = []


class GpuPose:
    """RT-DETRv4-X + Sapiens2-1B in bf16, one frame at a time.

    Sapiens2 loads with bf16 weights and every forward runs under bf16
    autocast (the detector keeps fp32 master weights; autocast picks its
    kernels). fp32 weights plus autocast recast the 1B model on every
    forward, and on the RTX PRO 6000 that overhead was the difference
    between missing and making the then 167ms 6fps budget, so there is no
    other dtype. The budget today is the two views' together: the body and
    the face view each at 9 fps (producer --pose-fps) share this one
    model through `_lock`, 18 steps a second, 55ms each, of which the
    Sapiens2-1B graph replay is 46ms on the slot's H100 (pose_bench,
    2026-09-10); `pose_load` measures whether a slot holds that. The
    remaining knobs default to the production configuration; an unset
    revision boots exactly what service.yaml describes:

      POSE_PREFETCH=1   copy the model store (hub cache + detector
                        checkpoints) from the FUSE mount to local disk
                        before loading - see prefetch_model_store. With
                        POSE_PREFETCH_BUCKET set the bytes bypass the mount
                        entirely; POSE_PREFETCH_TRANSPORT pins a transport
                        rung for A/B probes. Off by default (the local
                        compose loop has no mount).
      POSE_DETECT_STRIDE  run the person detector on every Nth pose frame
                        and pose the cached box in between. The fixed camera
                        and braced prone body move the box slowly; a detect
                        (33ms on Blackwell, 8ms on the H100) on every pose
                        frame is budget spent re-finding a box that has not
                        moved. A lost person always re-detects on the next
                        frame. Default 3: at 9 fps, three detects a second
                        per view.
      POSE_FLIP_TTA=1   the flip-TTA pair instead of a single forward. With
                        TTA the Blackwell worker ran at ~93% of the then
                        6fps budget and queue bursts dropped ~11% of poses -
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
      POSE_LOAD_BENCH=9,9,120  after the batch bench, run the two views'
                        pose load - body and face cadences, seconds - the
                        way a session submits it, and print one `poseLoad`
                        line with drops, step and wait times and the GPU
                        busy fraction, reported as gauges too (pose_load).
                        The same kind of knob: admitted, never set in
                        production, nothing of it survives the boot.

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
        self._detect_stride = max(
            1, int(os.environ.get("POSE_DETECT_STRIDE", "3") or "3"))
        # What a session keeps between frames, per view (ViewState). The
        # body view - the session's primary stream - is always there; a
        # second view (the phone's camera beside a fixed one) is made on
        # first use. The body view's identity anchor and scenery live on the
        # tracker itself, as they always did; another view's are swapped in
        # for its step and out again.
        self.views: dict[str, ViewState] = {BODY_VIEW: ViewState()}
        # One frame at a time through the models, whichever view it is
        # from: the captured graphs have one set of static inputs.
        self._lock = threading.Lock()

    # The body view's state, under the names the rest of the module and the
    # tests always used.
    @property
    def on_full(self):
        """The live overlay's tap on the full result (see
        share_full_result); a session sets it for its duration and clears
        it after."""
        return self.views[BODY_VIEW].on_full

    @on_full.setter
    def on_full(self, value) -> None:
        self.views[BODY_VIEW].on_full = value

    @property
    def _warmup(self):
        return self.views[BODY_VIEW].warmup

    @property
    def _scenery_done(self) -> bool:
        return self.views[BODY_VIEW].scenery_done

    @_scenery_done.setter
    def _scenery_done(self, value: bool) -> None:
        self.views[BODY_VIEW].scenery_done = bool(value)

    @property
    def _detect_countdown(self) -> int:
        return self.views[BODY_VIEW].detect_countdown

    @_detect_countdown.setter
    def _detect_countdown(self, value: int) -> None:
        self.views[BODY_VIEW].detect_countdown = int(value)

    @property
    def _cached_detection(self):
        return self.views[BODY_VIEW].cached_detection

    @_cached_detection.setter
    def _cached_detection(self, value) -> None:
        self.views[BODY_VIEW].cached_detection = value

    def view(self, name: str | None) -> ViewState:
        """The named view's state, made on first use; None is the body's."""
        name = name or BODY_VIEW
        state = self.views.get(name)
        if state is None:
            state = self.views[name] = ViewState()
        return state

    def reset_session_state(self) -> None:
        """Forget the previous session's scene, keep the booted weights.

        The model is session-independent; scenery, the identity anchor and
        the detect cache are not - for any view. Clearing them is what lets
        one booted instance serve consecutive sessions without minutes of
        reload. The overlay taps are kept: the session sets and clears
        those itself.
        """
        with self._lock:
            for state in self.views.values():
                state.reset()
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
        self.bench_load(os.environ.get("POSE_LOAD_BENCH"))
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

    def bench_load(self, spec: str | None) -> dict | None:
        """The boot-time load bench when `POSE_LOAD_BENCH` names a load
        (see `pose_load`); nothing otherwise. As with the batch bench a
        failure is printed and counted (`poseLoadFailed`), never a lost
        boot, and whatever the bench's detections left on the tracker is
        cleared before the slot serves."""
        import pose_load

        plan = pose_load.parse_spec(spec)
        if plan is None:
            return None
        try:
            return pose_load.run(self.load_stepper(), plan, lock=self._lock,
                                 telemetry=self.telemetry)
        except Exception as error:  # noqa: BLE001 - a bench, not the service
            print(f"poseLoad: failed: {error!r}", flush=True)
            if self.telemetry:
                self.telemetry.count("poseLoadFailed")
            return None
        finally:
            self.reset_session_state()

    def load_stepper(self, size: tuple[int, int] = (720, 1280)):
        """The load bench's step: a session's device work for one pose
        frame of `view`, its k-th. The frame is uploaded as the decoder's
        HWC uint8 array is (`Tracker.frame_tensor`), the detector runs on
        every POSE_DETECT_STRIDE-th step as `_step` runs it, and the pose
        crop and graph replay run over a fixed box (`pose_bench.slot_boxes`)
        with the copy back, since the synthetic frame shows nobody to
        detect. `size` is rows x columns: a 1280x720 stream's frame."""
        import pose_bench

        tracker = self.tracker
        rgb = self.load_frame(size)
        height, width = size
        box = pose_bench.slot_boxes(1, float(width), float(height))[0]
        stride = self._detect_stride
        device = self.device

        def step(view: str, k: int) -> None:
            frame = tracker.frame_tensor(rgb, device)
            if k % stride == 0:
                tracker.detect(frame)
            tracker.pose_keypoints(frame, box)

        return step

    @staticmethod
    def load_frame(size: tuple[int, int] = (720, 1280)) -> np.ndarray:
        """A synthetic HWC uint8 RGB frame on the host with some structure
        (the parity frame's gradient and block, at a stream's size), for
        the load bench to upload as a decoded frame is uploaded."""
        height, width = size
        rows = np.linspace(0, 255, height, dtype=np.float32)[:, None]
        cols = np.linspace(0, 255, width, dtype=np.float32)[None, :]
        rows = np.broadcast_to(rows, (height, width))
        cols = np.broadcast_to(cols, (height, width))
        frame = np.stack([rows, cols, (rows + cols) / 2], axis=-1)
        frame = np.round(frame).astype(np.uint8)
        frame[height // 4: 3 * height // 4, width // 3: 2 * width // 3] = 200
        return np.ascontiguousarray(frame)

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

    def _learn_scenery(self, state: ViewState, at_s: float) -> None:
        if state.scenery_done or at_s < SCENERY_WARMUP_S:
            return
        per_frame = [boxes for boxes, _ in state.warmup]
        grays = [gray for _, gray in state.warmup]

        def motion(box: list[float]) -> float:
            x0, y0, x1, y1 = (int(v / SCENERY_SCALE) for v in box)
            deltas = [
                float(np.abs(after[y0:y1, x0:x1].astype(np.int16)
                             - before[y0:y1, x0:x1].astype(np.int16)).mean())
                for before, after in zip(grays, grays[1:])
                if after[y0:y1, x0:x1].size
            ]
            return float(np.mean(deltas)) if deltas else 0.0

        # The view's scenery goes on the tracker: this runs with the view's
        # state swapped in (step), so the tracker's is this view's.
        self.tracker.scenery = pose_rows.find_scenery(per_frame, motion)
        state.scenery_done = True
        state.warmup.clear()

    def step(self, rgb: np.ndarray, frame_index: int, at_s: float,
             view: str | None = None) -> dict:
        """One frame of `view` (the body's by default) through detection
        and keypoints: the row for the assembler, the full result to the
        view's tap. Views take turns on the model."""
        name = view or BODY_VIEW
        state = self.view(name)
        with self._lock:
            if name == BODY_VIEW:
                return self._step(state, rgb, frame_index, at_s)
            # Another view's turn: its identity anchor and scenery on the
            # tracker for the step, the body's kept and put back after.
            tracker = self.tracker
            body_previous, body_scenery = tracker.previous, tracker.scenery
            tracker.previous, tracker.scenery = state.previous, state.scenery
            try:
                return self._step(state, rgb, frame_index, at_s)
            finally:
                state.previous, state.scenery = tracker.previous, tracker.scenery
                tracker.previous, tracker.scenery = body_previous, body_scenery

    def _step(self, state: ViewState, rgb: np.ndarray, frame_index: int,
              at_s: float) -> dict:
        tracker = self.tracker
        started = time.monotonic()
        # The frame's one trip to the device; everything below reads it there.
        frame = tracker.frame_tensor(rgb, self.device)
        if self.telemetry:
            self.telemetry.observe("frameUpload", time.monotonic() - started)
        if not state.scenery_done:
            # Scenery learning needs every person candidate, before identity
            # selection; the detector backend owns preprocessing and label
            # mapping, so the candidates come through it.
            candidates = tracker.detection_candidates(frame)
            boxes = [box for box, _ in candidates]
            small = cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                (rgb.shape[1] // SCENERY_SCALE,
                 rgb.shape[0] // SCENERY_SCALE))
            state.warmup.append((boxes, small))
            self._learn_scenery(state, at_s)

        cached = state.cached_detection
        if (cached is not None and cached[0] is not None
                and state.detect_countdown > 0):
            state.detect_countdown -= 1
            box, score, people, unresolved = cached
        else:
            started = time.monotonic()
            box, score, people, unresolved = tracker.detect(frame)
            if self.telemetry:
                # Cache hits are not observed; the stage reports what a real
                # detect costs, not an average diluted by reuse. The copy
                # back is inside, so this is the latency, not the enqueue.
                self.telemetry.observe("detect", time.monotonic() - started)
            state.cached_detection = (box, score, people, unresolved)
            state.detect_countdown = self._detect_stride - 1
        if box is None:
            share_full_result(state.on_full, self.telemetry, frame_index, at_s,
                              None, None, None, None, people, unresolved)
            return pose_rows.missing_row(frame_index, at_s)
        started = time.monotonic()
        # (K, 3) x, y, score in image pixels, already on the host: the row
        # builder and the overlay read plain numpy from here on.
        result = tracker.pose_keypoints(frame, box)
        if self.telemetry:
            self.telemetry.observe("poseInfer", time.monotonic() - started)
        keypoints, scores = result[:, :2], result[:, 2]
        share_full_result(state.on_full, self.telemetry, frame_index, at_s,
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
        # The capture's pose interval in frames: the smallest gap between
        # its rows. A cadence that divides the 30 fps grid has one gap (5
        # at 6 fps); 9 fps has a 3-4-3 pattern, so 3, and a lookup at any
        # picked slot resolves within half of it below.
        self.stride = min(
            (b - a for a, b in zip(frames, frames[1:]) if b > a), default=1)
        self.span = (frames[-1] + self.stride) if frames else 1
        self.telemetry = telemetry
        # Same taps as GpuPose, per view; the body's is fed the row's 21
        # body points, so the local loop renders the overlay without a GPU
        # (hands and face stay empty). A capture holds one stream's poses,
        # so another view replays as nobody there.
        self.views: dict[str, ViewState] = {BODY_VIEW: ViewState()}
        if telemetry:
            telemetry.boot_phase("weights", started)

    @property
    def on_full(self):
        return self.views[BODY_VIEW].on_full

    @on_full.setter
    def on_full(self, value) -> None:
        self.views[BODY_VIEW].on_full = value

    def view(self, name: str | None) -> ViewState:
        """The named view's state, made on first use; None is the body's."""
        name = name or BODY_VIEW
        state = self.views.get(name)
        if state is None:
            state = self.views[name] = ViewState()
        return state

    def boot(self) -> None:
        return None

    def step(self, rgb: np.ndarray, frame_index: int, at_s: float,
             view: str | None = None) -> dict:
        state = self.view(view)
        if (view or BODY_VIEW) != BODY_VIEW:
            share_full_result(state.on_full, self.telemetry, frame_index, at_s,
                              None, None, None, None, 0, False)
            return {"frame": frame_index, "atS": round(at_s, 4),
                    "keypoints": None}
        key = frame_index % self.span
        row = self.rows.get(key)
        if row is None:
            # A stride the capture did not use: the nearest row that is
            # within half a pose interval stands in.
            nearest = min(self.rows, key=lambda f: abs(f - key), default=None)
            if nearest is not None and abs(nearest - key) * 2 <= self.stride:
                row = self.rows[nearest]
        if row is None or not row.get("keypoints"):
            share_full_result(state.on_full, self.telemetry, frame_index, at_s,
                              None, None, None, None,
                              (row or {}).get("people", 0),
                              (row or {}).get("identityUnresolved", False))
            return {"frame": frame_index, "atS": round(at_s, 4),
                    "keypoints": None}
        if state.on_full is not None:
            points, scores = row_keypoint_arrays(row["keypoints"])
            share_full_result(state.on_full, self.telemetry, frame_index, at_s,
                              points, scores, row.get("box"),
                              row.get("boxScore"), row.get("people", 1),
                              row.get("identityUnresolved", False))
        return {"frame": frame_index, "atS": round(at_s, 4),
                "keypoints": row["keypoints"]}
