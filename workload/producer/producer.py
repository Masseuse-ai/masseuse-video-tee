"""The producer: a video stream in, keypoints and motion descriptors out.

This process is the only one in the enclave that holds a decoded frame.
One ffmpeg decode at 30fps yuv420p feeds everything: the Y plane is what
the regional motion descriptors are measured on, and every fifth frame is
converted to RGB for the pose worker (RT-DETRv4 person detection, then
Sapiens2-1B keypoints). Pose runs at 6fps in its own thread behind a
bounded queue; a stalled or slow pose drops frames - counted, clock still
advanced with keypoint-less rows, released to the assembler in slot order -
rather than ever blocking the decode. Rows come out of the online
interpolator at frame cadence, the descriptors out of the pair arithmetic
at both cadences (workload/pixel/motion.py), and both are handed over a
local socket to the analysis process (analysis/protocol.md), whose readings
come back to be posted to the trainer. Frames go no further than this file
and the overlay renderer, which draws the annotated view returned to the
same user's phone.

Two run shapes, one binary:

  local / testbed   --stream <url> [--duration N]: produce immediately,
                    capture JSONL to --sink-dir, exit at EOF or duration.
  serve             --serve: wait for GET /produce?stream=...&duration=...,
                    hold the response open as SSE for the session and stop
                    when the client closes or says POST /stop. `stream`
                    may be a file under the models mount (played at -re
                    pace, ending at EOF) and `run=<name>` uploads the
                    session's capture, raw pose rows included, to
                    gs://<capture-bucket>/runs/<name>/ when it ends - never
                    in --tee mode, where no capture leaves the enclave.

    python producer.py --stream rtsps://localhost:8322/cam \
        --pose sideload --track /path/to/capture --duration 150 \
        --analysis-socket /run/analysis.sock

(`--track` names a captured session - the directory holding its
`poses.jsonl` - which `--pose sideload` replays in place of the GPU. With
no `--analysis-socket` the pixel path runs alone and nothing is posted.)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_link import open_link  # noqa: E402
from camlink_gateway import (CamlinkGateway, GatewayError,  # noqa: E402
                             parse_expectation)
from external_source import ExternalSource, SourceError  # noqa: E402
from live_pose import GpuPose, SideloadPose, prefetch_model_store  # noqa: E402
from motion import DescriptorWorker, RowAssembler  # noqa: E402
from overlay import build_renderer  # noqa: E402
from relay_proxy import (RelayProxy, location_secret,  # noqa: E402
                         route as relay_route)
from sinks import Capture, Poster  # noqa: E402
from tee_mode import (CLIENT_NONCE, IdleExit, TeeMode,  # noqa: E402
                      bearer as bearer_token, cors_headers)
from telemetry import Telemetry  # noqa: E402

FPS = 30.0
# The geometry a live stream is decoded at: the probed aspect with its long
# edge here. The phones ask for 1280x720 (web/src/camera/useCamera.ts), so
# a phone that has finished ramping decodes at its own size; one still
# ramping (or throttled) is upscaled to it rather than pinning the session
# at whatever the first seconds carried.
STREAM_LONG_SIDE = 1280
POSE_QUEUE_DEPTH = 2
# A run name becomes a capture sub-directory and a GCS path segment.
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# The only files a /produce request may name: the bucket mount. A track or
# clip path from the query is normalised and must sit under this root; the
# CLI's own defaults are the operator's and are not checked.
MOUNT_ROOT = os.environ.get("POSE_MOUNT_ROOT", "/models")


def mounted_path(candidate: str) -> str | None:
    """`candidate` as a normalised path under MOUNT_ROOT, or None."""
    normalized = os.path.normpath(candidate)
    root = os.path.normpath(MOUNT_ROOT) + os.sep
    if normalized.startswith(root):
        return normalized
    return None
PENDING_CAP = 256
# Decided frames waiting for the descriptors thread; ~8 MB of 4K luma each.
DESCRIPTORS_QUEUE_DEPTH = 128
STALL_RESTART_S = 10.0
LOG_EVERY_S = 15.0
# Summary keys the producer owns; the analysis summary never overwrites them.
RESERVED_SUMMARY_KEYS = frozenset({
    "bootMs", "counters", "gauges", "stagesMs", "run", "captureDir",
    "overlay",
})


def pipe_size(probed: tuple[int, int],
              long_side: int = STREAM_LONG_SIDE) -> tuple[int, int]:
    """The fixed geometry a live stream is decoded at.

    The probed aspect with its long edge at `long_side`, both sides even
    (yuv420p), never smaller than what was probed: a 360x640 ramp-up frame
    becomes 720x1280, a 1080x1920 phone stays 1080x1920.
    """
    width, height = probed
    scale = max(1.0, long_side / max(width, height))

    def even(value: float) -> int:
        return max(2, int(round(value * scale / 2)) * 2)

    return even(width), even(height)


class Decoder:
    """ffmpeg rawvideo yuv420p frames off a stream, with reconnect.

    A live stream's geometry is pinned: the probe decides the aspect, the
    pipe carries `pipe_size` of it, and ffmpeg scales (letterboxing if the
    aspect turns) so every read is exactly one frame. Without the pin the
    reader trusted `ffprobe`'s size while the phone's encoder was still
    ramping up: the iPad of 2026-09-08 was probed at half its final
    resolution, ffmpeg's raw output followed its own first frame, and every
    720x1280 frame was sliced into four 360x640 "frames" - stripes and false
    colour on the overlay, 120 frames/s in the telemetry, the pose fed
    garbage. Files keep their native size; nothing changes theirs.

    `stopping` is honoured between reconnects as well as between frames: a
    session whose camera has left the relay sits in the reconnect loop, and
    a `/stop` that only the frame loop checked never landed there - the
    orphaned slot kept its GPU until the duration cap (2026-09-04, phone
    closed while the trainer was replaced). `lost_after_s` bounds that loop
    on its own: with no frames for that long the stream is treated as
    ended, the way a file ending is, and the session writes its summary.
    0 keeps reconnecting forever, the standing behaviour for a persistent
    RTSP source.
    """

    def __init__(self, url: str, telemetry: Telemetry,
                 realtime: bool = False,
                 stopping: threading.Event | None = None,
                 lost_after_s: float = 0.0, clock=time.monotonic,
                 sleep=time.sleep):
        self.url = url
        self.telemetry = telemetry
        self.stopping = stopping
        self.lost_after_s = float(lost_after_s or 0.0)
        self.clock = clock
        self.sleep = sleep
        # A file decodes as fast as ffmpeg can read it, which starves a
        # real GPU pose worker of wall time and craters the delivered pose
        # cadence in media time. -re paces a file like the camera it stands
        # in for; network streams pace themselves.
        network = url.startswith(("rtsp://", "rtsps://"))
        self.network = network
        self.realtime = realtime and not network
        # Whether something outside this process sets the frame cadence -
        # the camera, or -re standing in for it. An unpaced file is the
        # offline sideload, and there the decoder waits for the descriptors
        # rather than dropping them.
        self.paced = realtime or network
        self.proc: subprocess.Popen | None = None
        # What the pipe carries. A live stream's is `pipe_size` of the probe
        # and ffmpeg is told to deliver exactly it; a file's is its own.
        self.size: tuple[int, int] | None = None
        self.probed: tuple[int, int] | None = None

    def _probe(self) -> tuple[int, int]:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json",
             *self._input_flags(), self.url],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(out.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])

    def _input_flags(self) -> list[str]:
        if self.url.startswith(("rtsp://", "rtsps://")):
            return ["-rtsp_transport", "tcp"]
        return []

    def _filters(self) -> str:
        """The -vf chain: the cadence, and for a live stream the geometry.

        `scale` keeps the aspect and never exceeds the pipe size, `pad`
        centres the result in it, so a sender that changes resolution (or
        turns) still delivers frames of exactly `self.size` down the pipe.
        """
        chain = f"fps={FPS:g}"
        if self.network:
            width, height = self.size
            chain += (f",scale={width}:{height}:force_original_aspect_ratio="
                      f"decrease:force_divisible_by=2,"
                      f"pad={width}:{height}:-1:-1")
        return chain

    def _argv(self) -> list[str]:
        pacing = ["-re"] if self.realtime else []
        return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                *pacing, *self._input_flags(), "-i", self.url,
                "-vf", self._filters(), "-f", "rawvideo",
                "-pix_fmt", "yuv420p", "pipe:1"]

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(self._argv(), stdout=subprocess.PIPE)

    def frames(self):
        """Yields (index, at_s, yuv) forever; index survives reconnects."""
        if self.size is None:
            self.probed = self._probe()
            self.size = pipe_size(self.probed) if self.network else self.probed
            print(f"decoder: {self.url} probed {self.probed[0]}x{self.probed[1]}, "
                  f"decoding at {self.size[0]}x{self.size[1]}", flush=True)
        width, height = self.size
        frame_bytes = width * height * 3 // 2
        index = 0
        lost_since: float | None = None
        while True:
            if self.stopping is not None and self.stopping.is_set():
                return
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            buffer = self.proc.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                self.telemetry.count("reconnects")
                self.proc.stdout.close()
                self.proc.wait()
                self.proc = None
                if not self.url.startswith(("rtsp://", "rtsps://")):
                    return  # a file ended; only network streams reconnect
                now = self.clock()
                if lost_since is None:
                    lost_since = now
                elif self.lost_after_s and now - lost_since >= self.lost_after_s:
                    self.telemetry.count("inputLost")
                    print(f"decoder: no frames from {self.url} for "
                          f"{now - lost_since:.0f}s; treating the stream as "
                          "ended", flush=True)
                    return
                self.sleep(1.0)
                continue
            lost_since = None
            yuv = np.frombuffer(buffer, np.uint8).reshape(
                height * 3 // 2, width)
            yield index, index / FPS, yuv
            index += 1

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()


class PoseWorker(threading.Thread):
    """Pose at the 6fps cadence, never allowed to block the decode."""

    def __init__(self, pose, assembler: RowAssembler, lock: threading.Lock,
                 telemetry: Telemetry, on_row=None,
                 queue_depth: int | None = None, on_pose=None):
        super().__init__(daemon=True)
        self.pose = pose
        self.assembler = assembler
        self.lock = lock
        self.telemetry = telemetry
        # Every real pose row in slot order, with its flags and the frame
        # size, for the analysis process (the `pose` message of
        # analysis/protocol.md): the interpolated rows the assembler makes
        # would fake the sample density its consumers key on.
        self.on_pose = on_pose
        self.frame_size: tuple[int, int] | None = None
        # Sees every row this worker hands the assembler - posed, dropped or
        # errored - with the wall clock attached: the capture's poses.jsonl.
        self.on_row = on_row
        # Depth 2 shipped on the L4. On Blackwell the measured drops at
        # depth 2 were burst- and lock-wait-driven with the average cycle
        # under budget, so a deeper queue absorbs them; the price is
        # staleness, bounded at depth x the pose cadence.
        depth = (queue_depth if queue_depth is not None
                 else int(os.environ.get("POSE_QUEUE_DEPTH", "0") or "0"))
        self.queue: queue.Queue = queue.Queue(
            maxsize=depth if depth > 0 else POSE_QUEUE_DEPTH)
        self.stopping = threading.Event()
        # Rows reach the assembler in slot order, whatever order they finish
        # in. A drop is known the instant the queue refuses it, while the
        # slots queued before it are still being inferred; handing the
        # dropped row over at once used to move the assembler's clock past
        # slots that had not arrived, and the frames under those slots were
        # decided with their pose missing - poseless where the whole track
        # would have snapped or bridged - at a cost of about half a second
        # of descriptors per drop.
        self._order: deque[int] = deque()
        self._finished: dict[int, tuple[dict, dict]] = {}
        self._release_lock = threading.Lock()

    def submit(self, rgb: np.ndarray, index: int, at_s: float) -> None:
        with self._release_lock:
            self._order.append(index)
        try:
            self.queue.put_nowait((rgb, index, at_s))
        except queue.Full:
            # The clock must advance even when the pose is dropped, or the
            # assembler would wait forever for a decision that never comes -
            # but only once the slots ahead of it have reported.
            self.telemetry.count("poseDropped")
            row = {"frame": index, "atS": round(at_s, 4), "keypoints": None}
            self._finish(index, row, {"dropped": True})

    def _finish(self, index: int, row: dict, flags: dict) -> None:
        """Bank one slot's row and release every row that is now in order."""
        with self._release_lock:
            self._finished[index] = (row, flags)
            ready = []
            while self._order and self._order[0] in self._finished:
                ready.append(self._finished.pop(self._order.popleft()))
        for ready_row, ready_flags in ready:
            with self.lock:
                self.assembler.push_pose(ready_row)
            if self.on_pose is not None:
                try:
                    self.on_pose(ready_row, ready_flags, self.frame_size)
                except Exception:  # noqa: BLE001 - a sink must not stall the pose
                    self.telemetry.count("poseRowSinkErrors")
            self._record(ready_row, **ready_flags)

    def _record(self, row: dict, **flags) -> None:
        if self.on_row is None:
            return
        try:
            self.on_row({**row, "wallS": round(time.time(), 3), **flags})
        except Exception:  # noqa: BLE001 - a sink must not stall the pose
            self.telemetry.count("poseRowSinkErrors")

    def run(self) -> None:
        while not self.stopping.is_set():
            try:
                rgb, index, at_s = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            flags: dict = {}
            if self.frame_size is None and hasattr(rgb, "shape"):
                self.frame_size = (int(rgb.shape[1]), int(rgb.shape[0]))
            try:
                with self.telemetry.time_stage("pose"):
                    row = self.pose.step(rgb, index, at_s)
            except Exception as error:  # noqa: BLE001 - said aloud, survived
                # An uncaught step error would kill this thread and the rest
                # of the session would report only runaway poseDropped (a
                # full boot was lost to exactly that). Count it, advance the
                # clock keypointless, keep serving.
                self.telemetry.count("poseErrors")
                print(f"pose step failed at {at_s:.1f}s: {error!r}",
                      flush=True)
                row = {"frame": index, "atS": round(at_s, 4),
                       "keypoints": None}
                flags = {"error": repr(error)}
            self._finish(index, row, flags)


# The booted GPU pose is process-wide: boot moves gigabytes and takes
# minutes cold, while the model itself carries no session state (what does -
# scenery, the identity anchor, the detect cache - is reset on reuse). The
# lock is what lets a /produce arriving mid-/warmup join that boot instead
# of duplicating VRAM and the prefetch.
_GPU_POSE_LOCK = threading.Lock()
_GPU_POSE: dict = {"pose": None}


def acquire_gpu_pose(args, telemetry: Telemetry) -> GpuPose:
    with _GPU_POSE_LOCK:
        pose = _GPU_POSE["pose"]
        if pose is None:
            pose = GpuPose(device=args.device, telemetry=telemetry)
            pose.boot()
            _GPU_POSE["pose"] = pose
        else:
            pose.reset_session_state()
        return pose


def gpu_pose_state() -> str:
    """What a /produce arriving now will find: "ready" when the process-wide
    pose is booted and nobody is mid-boot, "booting" otherwise (a cold
    instance, or a /warmup still moving weights that the session will join)."""
    if _GPU_POSE["pose"] is not None and not _GPU_POSE_LOCK.locked():
        return "ready"
    return "booting"


class BootKeepalive:
    """Keeps a /produce SSE response alive while Session() waits for the boot.

    A cold instance takes tens of seconds to boot the pose model, and the
    first /produce to arrive into that boot blocks in Session() (on the GPU
    pose lock) with the 200 already sent and nothing else on the wire. The
    FemLed server's client and Cloud Run's front end both give up on a
    silent stream: the client's headers timeout expired mid-boot in the
    2026-09-03 session and the abandoned request kept the busy lock, so
    every retry met a 409. This writes a `hello` event the moment the
    headers are out (state "ready" or "booting", so the client knows which
    wait it is in) and an SSE comment every `interval_s` from a helper
    thread until stop(), which the handler calls once the session exists.
    A client that leaves during the boot is noticed (`failed`), logged, and
    then handled by the pump's own broken-pipe path.
    """

    def __init__(self, wfile, interval_s: float = 5.0):
        self.wfile = wfile
        self.interval_s = interval_s
        self.failed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def hello(self, state: str) -> None:
        self._write(f"data: {json.dumps({'kind': 'hello', 'state': state})}\n\n".encode())

    def start(self) -> "BootKeepalive":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            if not self._write(b": keepalive\n\n"):
                return

    def _write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.failed = True
            return False


def capture_dir(sink_dir: str, run_name: str) -> Path:
    """Where a session's capture lands.

    A named run gets its own directory: the capture files open in append
    mode, so back-to-back named sessions on one warm instance would
    otherwise interleave their rows in a single poses.jsonl. Unnamed
    sessions keep writing to the sink dir itself, as the local loop and the
    equivalence gate expect.
    """
    if run_name and not RUN_NAME.match(run_name):
        raise ValueError(f"run name {run_name!r} is not a path segment")
    root = Path(sink_dir)
    return root / run_name if run_name else root


class Session:
    """One stream, start to stop."""

    def __init__(self, args, telemetry: Telemetry):
        self.args = args
        self.telemetry = telemetry
        if args.pose == "sideload":
            self.pose = SideloadPose(Path(args.track), telemetry)
            started = time.monotonic()
            self.pose.boot()
        else:
            started = time.monotonic()
            self.pose = acquire_gpu_pose(args, telemetry)
        telemetry.boot_phase("poseReady", started)

        # Pose cadence and the interpolation bridge it needs: the bridge
        # scales with the pose interval (1.3x of it, at least 0.35 s).
        pose_fps = float(getattr(args, "pose_fps", 6.0) or 6.0)
        self.pose_fps = pose_fps
        self.pose_stride = max(1, int(round(FPS / pose_fps)))
        bridge_s = max(0.35, 1.3 * self.pose_stride / FPS)
        self.assembler = RowAssembler(FPS, bridge_s=bridge_s)
        self.assembler_lock = threading.Lock()
        self.descriptors = DescriptorWorker()
        self.run_name = getattr(args, "run", "") or ""
        self.capture = Capture(capture_dir(args.sink_dir, self.run_name))
        self.summary: dict | None = None
        self.poster = (Poster(args.post_url, telemetry)
                       if args.post_url else None)
        self.post_interval_s = float(
            getattr(args, "post_interval_s", 1.0) or 1.0)
        self.stopping = threading.Event()
        # The live annotated view (overlay.py): None unless a publish URL is
        # configured, in which case it runs on its own thread and taps the
        # decode, the full pose result and the analysis's HUD lines through
        # offer_*.
        record = (self.capture.directory / "overlay.mp4"
                  if getattr(args, "overlay_record", False) else None)
        self.overlay = build_renderer(args, telemetry, source_fps=FPS,
                                      record_path=record)
        # The analysis process (analysis/protocol.md). Keypoints and
        # descriptors go out; readings, records, HUD text and gauges come
        # back through _on_analysis. Opened last so a session that fails to
        # reach it has nothing else to tear down.
        self.analysis = open_link(
            getattr(args, "analysis_socket", "") or "",
            self._on_analysis, self._on_analysis_error,
            fps=FPS, poseFps=pose_fps, postIntervalS=self.post_interval_s,
            run=self.run_name or None)
        if self.analysis.connected:
            ready = self.analysis.ready or {}
            print(f"analysis: {ready.get('version', '?')} "
                  f"({ready.get('modelVersion', '?')}) on "
                  f"{args.analysis_socket}", flush=True)

    # -- what comes back from the analysis process ---------------------------

    def _on_analysis(self, message: dict) -> None:
        """One message from the analysis process, on the link's reader
        thread. Records go to the capture and the session's event stream,
        readings also to the trainer; HUD text to the overlay; gauges to the
        telemetry. Nothing here looks inside the bodies."""
        kind = message.get("kind")
        if kind == "post":
            body = message.get("body")
            if not isinstance(body, dict):
                return
            self.capture.post(body)
            # Also down the SSE: the capture dir dies with the instance and
            # the session driver's transcript is how readings leave it.
            self.telemetry.emit("post", body)
            if self.poster is not None:
                self.poster.post(body)
        elif kind == "onset":
            at_s = message.get("atS")
            if isinstance(at_s, (int, float)):
                self.capture.onset(float(at_s))
                self.telemetry.emit("onset", {"atS": round(float(at_s), 3)})
        elif kind == "event":
            from_s, to_s = message.get("fromS"), message.get("toS")
            if isinstance(from_s, (int, float)) and isinstance(to_s, (int, float)):
                self.capture.event(float(from_s), float(to_s))
        elif kind == "payload":
            payload = message.get("payload")
            if isinstance(payload, dict):
                self.capture.payload(payload)
                self.telemetry.emit("payload", payload)
                self.telemetry.count("payloads")
        elif kind == "hud":
            lines = message.get("lines")
            if self.overlay is not None and isinstance(lines, list):
                self.overlay.offer_context(
                    "hud", [str(line) for line in lines][:8])
        elif kind == "gauges":
            values = message.get("values")
            if isinstance(values, dict):
                for name, value in values.items():
                    if isinstance(value, (int, float)):
                        self.telemetry.gauge(str(name), float(value))
        elif kind == "log":
            print(f"analysis: {message.get('text', '')}", flush=True)

    def _on_analysis_error(self, text: str) -> None:
        self.telemetry.count("analysisErrors")
        print(text, flush=True)

    # -- what goes to it -----------------------------------------------------

    def _on_pose(self, row: dict, flags: dict,
                 frame_size: tuple[int, int] | None) -> None:
        """Every real pose row, from the pose worker."""
        self.analysis.send({
            "kind": "pose",
            "frame": row["frame"],
            "atS": row["atS"],
            "keypoints": row.get("keypoints"),
            "dropped": bool(flags.get("dropped")),
            "error": bool(flags.get("error")),
            "frameSize": list(frame_size) if frame_size else None,
        })

    def _gauges(self, at_s: float, wall_start: float,
                queued: int) -> None:
        wall = time.monotonic() - wall_start
        self.telemetry.gauge("streamS", at_s)
        self.telemetry.gauge("realtimeFactor",
                             at_s / wall if wall > 0 else 0.0)
        self.telemetry.gauge("e2eLagS", wall - at_s)
        self.telemetry.gauge("descriptorsQueued", queued)

    def _observe(self, row: dict, gray_row: np.ndarray) -> None:
        """One decided frame through the descriptors and over the socket.
        Runs on the descriptors thread; the last place this frame's pixels
        are read."""
        with self.telemetry.time_stage("descriptors"):
            fast, slow = self.descriptors.step(row, gray_row)
        self.analysis.send({
            "kind": "frame",
            "frame": row["frame"],
            "atS": row["atS"],
            "keypoints": row.get("keypoints"),
            "fast": fast.as_json() if fast is not None else None,
            "slow": slow.as_json() if slow is not None else None,
        })

    def _observe_loop(self, work: queue.Queue) -> None:
        while True:
            item = work.get()
            if item is None:
                return
            try:
                self._observe(*item)
            except Exception as error:  # noqa: BLE001 - said aloud, survived
                self.telemetry.count("descriptorErrors")
                print(f"descriptors failed at {item[0]['atS']}s: {error!r}",
                      flush=True)

    def run(self) -> dict:
        decoder = Decoder(
            self.args.stream, self.telemetry,
            realtime=self.args.pose == "gpu", stopping=self.stopping,
            lost_after_s=getattr(self.args, "input_lost_after", 0.0) or 0.0)
        worker = PoseWorker(self.pose, self.assembler, self.assembler_lock,
                            self.telemetry, on_row=self.capture.pose,
                            on_pose=self._on_pose)
        if self.overlay is not None:
            # The full result reaches the overlay before row_for narrows it;
            # the tap is per session on a process-wide GpuPose.
            self.pose.on_full = self.overlay.offer_pose
            self.overlay.start()
            print(f"overlay: publishing {self.overlay.publisher.url} "
                  f"{self.overlay.snapshot()['size']}@{self.overlay.fps:g} "
                  f"{self.overlay.publisher.encoder}, {self.overlay.delay_s:.1f}s "
                  f"behind decode", flush=True)
        worker.start()
        # Descriptors off the decode thread. Measuring a frame pair costs
        # ~25 ms of a 33 ms frame budget, and decisions arrive in bunches -
        # each pose row released frees the frames under it - so measuring
        # inline stalled the decoder for the bunch, the paced ffmpeg pipe
        # backed up, and the catch-up submitted pose frames faster than the
        # worker drains them. The queue absorbs the bunches; the decoder
        # never waits on Farneback.
        work: queue.Queue = queue.Queue(maxsize=DESCRIPTORS_QUEUE_DEPTH)
        observer = threading.Thread(target=self._observe_loop, args=(work,),
                                    daemon=True)
        observer.start()
        pending: dict[int, np.ndarray] = {}
        wall_start = time.monotonic()
        last_log = wall_start
        last_frame_wall = wall_start
        last_index = -1
        width = height = None
        try:
            for index, at_s, yuv in decoder.frames():
                if self.stopping.is_set():
                    break
                if self.args.duration and at_s >= self.args.duration:
                    break
                last_frame_wall = time.monotonic()
                last_index = index
                if width is None:
                    width = yuv.shape[1]
                    height = yuv.shape[0] * 2 // 3
                gray = yuv[:height]
                self.telemetry.count("framesIn")
                if self.overlay is not None:
                    self.overlay.offer_frame(index, at_s, yuv)  # O(1): a reference
                pending[index] = gray.copy()
                if len(pending) > PENDING_CAP:
                    pending.pop(min(pending))
                    self.telemetry.count("pendingEvicted")
                if index % self.pose_stride == 0:
                    with self.telemetry.time_stage("rgb"):
                        rgb = cv2.cvtColor(
                            yuv.reshape(-1, width), cv2.COLOR_YUV2RGB_I420)
                    worker.submit(rgb, index, at_s)
                with self.assembler_lock:
                    rows = self.assembler.flush()
                for row in rows:
                    gray_row = pending.pop(row["frame"], None)
                    if gray_row is None:
                        self.telemetry.count("grayMissing")
                        continue
                    if not decoder.paced:
                        work.put((row, gray_row))
                        continue
                    try:
                        work.put_nowait((row, gray_row))
                    except queue.Full:
                        # Counted, not waited for: a frame whose descriptors
                        # are lost breaks one pair, a stalled decode breaks
                        # the pose cadence for everything behind it.
                        self.telemetry.count("descriptorsDropped")
                self._gauges(at_s, wall_start, work.qsize())
                now = time.monotonic()
                if now - last_log >= LOG_EVERY_S:
                    print(self.telemetry.log_line(), flush=True)
                    last_log = now
                if now - last_frame_wall > STALL_RESTART_S:
                    self.telemetry.count("stalls")
        finally:
            worker.stopping.set()
            decoder.stop()
            if self.overlay is not None:
                # Before the capture closes: the recording, if any, must be
                # finalised on disk to ride the upload with the jsonl files.
                self.pose.on_full = None
                self.overlay.close()
            with self.assembler_lock:
                rows = self.assembler.drain(last_index)
            for row in rows:
                gray_row = pending.pop(row["frame"], None)
                if gray_row is None:
                    continue
                work.put((row, gray_row))
            work.put(None)
            observer.join()
            # Every frame is over the socket; the analysis flushes, answers
            # with its summary (its last readings may still arrive while it
            # does), and the link closes.
            last_at_s = (last_index / FPS) if last_index >= 0 else None
            analysis_summary = self.analysis.stop(last_at_s)
            if self.poster is not None:
                self.poster.close(timeout_s=self.poster.timeout_s + 1)
            summary = self.telemetry.snapshot()
            for key, value in analysis_summary.items():
                if key not in RESERVED_SUMMARY_KEYS:
                    summary[key] = value
            summary["run"] = self.run_name or None
            summary["captureDir"] = str(self.capture.directory)
            if self.overlay is not None:
                summary["overlay"] = self.overlay.snapshot()
            self.capture.summary(summary)
            self.capture.close()
            self.summary = summary
        return summary

    def upload_capture(self, bucket: str) -> dict | None:
        """Copy this run's capture to gs://bucket/runs/<run>/, if named.

        Unnamed sessions keep the old behaviour (files stay on the instance);
        the run name is what makes the destination unambiguous.
        """
        if not self.run_name or not bucket:
            return None
        result = self.capture.upload(bucket, f"runs/{self.run_name}",
                                     self.telemetry)
        print(f"capture upload: {json.dumps(result)}", flush=True)
        return result


# The CPU driver of Cloud Run's autoscaler kept recommending an instance
# for 88 s after a session went quiet and had let go by 148 s (2026-09-02,
# scaling/recommended_instances against a 0.3% CPU minute); the drain
# outlasts that with margin. The concurrency driver would hold for ten
# minutes and is disabled on the service instead.
TEARDOWN_DRAIN_S = 180.0


class Teardown:
    """/teardown: drain until the autoscaler has forgotten us, then exit.

    On Cloud Run a container exit is instance death, and a GPU instance
    that is dead stops billing - but only if the autoscaler agrees it
    should be dead. Its metrics-based drivers recommend one instance for as
    long as their lookbacks read utilisation, and Cloud Run answers an exit
    inside that window by starting a replacement that idles, billed, until
    the window closes: ten GPU-minutes per teardown while the concurrency
    driver (ten-minute lookback) was active, and a five-minute idle
    replacement even on the CPU driver alone. The concurrency driver is
    disabled in service.yaml; this class handles the CPU driver by exiting
    only once the instance has been quiet - no session, no boot, no
    prefetch - for `drain_s`.

    `mode=now` keeps the old immediate exit for the caller that wants a
    fresh instance right away and will boot the replacement itself.
    """

    def __init__(self, busy: threading.Lock, drain_s: float = TEARDOWN_DRAIN_S,
                 clock=time.monotonic, exit_impl=os._exit, sleep=time.sleep):
        self.busy = busy
        self.drain_s = float(drain_s)
        self.clock = clock
        self.exit_impl = exit_impl
        self.sleep = sleep
        self._lock = threading.Lock()
        self._working = 0
        self._quiet_since = clock()
        self._draining = False
        self._generation = 0

    # -- what counts as busy ------------------------------------------------

    def working(self):
        """Context manager for anything that burns CPU or holds a request:
        the model-store prefetch, a boot, a /produce session."""
        teardown = self

        class _Working:
            def __enter__(self):
                with teardown._lock:
                    teardown._working += 1
                return self

            def __exit__(self, *exc):
                with teardown._lock:
                    teardown._working -= 1
                    if teardown._working == 0:
                        teardown._quiet_since = teardown.clock()

        return _Working()

    def quiet_for_s(self) -> float:
        with self._lock:
            if self._working:
                return 0.0
            return max(0.0, self.clock() - self._quiet_since)

    def is_working(self) -> bool:
        """Something that counts as busy is in progress right now
        (tee_mode.IdleExit asks on every tick)."""
        with self._lock:
            return self._working > 0

    def exit_in_s(self) -> float:
        return max(0.0, self.drain_s - self.quiet_for_s())

    # -- the request ----------------------------------------------------------

    def request(self, mode: str = "drain") -> tuple[int, dict]:
        """Decide a /teardown: (status, body).

        409 while /produce holds the session lock - the pipeline is never
        killed out from under an open stream. `now` takes the lock to the
        grave so nothing starts between the response and the exit. `drain`
        releases it: a /warmup or /produce inside the drain cancels the
        teardown and gets the still-warm pipeline instead of a cold boot.
        """
        if not self.busy.acquire(blocking=False):
            return 409, {"error": "a session is running"}
        if mode == "now":
            with self._lock:
                self._draining = False
                self._generation += 1
            threading.Thread(target=self._exit_now, daemon=True).start()
            return 200, {"status": "terminating"}
        self.busy.release()
        with self._lock:
            self._draining = True
            self._generation += 1
            generation = self._generation
        exit_in = self.exit_in_s()
        threading.Thread(target=self._watch, args=(generation,),
                         daemon=True).start()
        return 200, {"status": "draining", "exitInS": round(exit_in, 1),
                     "drainS": self.drain_s}

    def cancel(self, reason: str) -> bool:
        with self._lock:
            if not self._draining:
                return False
            self._draining = False
            self._generation += 1
        print(f"teardown: drain cancelled by {reason}", flush=True)
        return True

    def snapshot(self) -> dict:
        with self._lock:
            draining = self._draining
        return {"draining": draining,
                "exitInS": round(self.exit_in_s(), 1) if draining else None,
                "quietForS": round(self.quiet_for_s(), 1),
                "drainS": self.drain_s}

    # -- the exits ------------------------------------------------------------

    def _exit_now(self) -> None:
        self.sleep(0.5)
        print("teardown: exiting now so the instance stops billing",
              flush=True)
        self.exit_impl(0)

    def _watch(self, generation: int) -> None:
        while True:
            with self._lock:
                if self._generation != generation or not self._draining:
                    return
            remaining = self.exit_in_s()
            if remaining <= 0:
                break
            self.sleep(min(remaining, 1.0))
        # A session that started in the last instant owns the instance.
        if not self.busy.acquire(blocking=False):
            self.cancel("a session")
            return
        print(f"teardown: exiting after {self.drain_s:.0f}s quiet so the "
              f"instance stops billing", flush=True)
        self.exit_impl(0)


# How long a departed client's session may take to wind down before the
# busy lock is released anyway. The per-frame stop check answers within a
# frame; the tail is the descriptors drain and the capture upload.
CLIENT_LEFT_JOIN_S = 30.0


def pump_session(session, runner: threading.Thread, wfile, subscription,
                 telemetry: Telemetry, *, upload_result=lambda: None,
                 wait_s: float = 2.0, join_timeout_s: float = CLIENT_LEFT_JOIN_S,
                 clock=time.monotonic, log=print) -> str:
    """Stream a /produce session's telemetry to its SSE client until it ends.

    The open request is the session: the caller that holds it open is the
    one consuming the readings, and when it goes away there is nobody to
    produce for. So a client that disconnects mid-session stops the session
    - `stopping` is honoured within a frame - and the runner is joined
    before this returns, so the busy lock the caller releases afterwards
    really does mean the GPU is idle. Before this, a departed client left
    the runner producing to its duration or EOF with the lock already
    released: a zombie billing the Blackwell, and a second /produce free to
    collide with it.

    Returns "ended" when the session ran to its own end (duration, EOF or a
    stop) and the summary event was written, "client_left" otherwise.
    """
    messages, event = subscription
    started = clock()
    try:
        while runner.is_alive():
            event.wait(timeout=wait_s)
            event.clear()
            while messages:
                line = messages.popleft()
                wfile.write(f"data: {line}\n\n".encode())
            wfile.write(b": keepalive\n\n")
            wfile.flush()
        telemetry.unsubscribe(subscription)
        while messages:
            line = messages.popleft()
            wfile.write(f"data: {line}\n\n".encode())
        summary = session.summary or {}
        final = {
            "kind": "summary",
            "run": session.run_name or None,
            "captureDir": summary.get("captureDir"),
            "upload": upload_result(),
            "counters": summary.get("counters"),
            "bootMs": summary.get("bootMs"),
        }
        wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        wfile.flush()
        return "ended"
    except (BrokenPipeError, ConnectionResetError):
        telemetry.unsubscribe(subscription)
        session.stopping.set()
        runner.join(timeout=join_timeout_s)
        ran_s = clock() - started
        if runner.is_alive():
            log(f"produce: client left after {ran_s:.0f}s; session still "
                f"winding down after {join_timeout_s:.0f}s, releasing anyway",
                flush=True)
        else:
            log(f"produce: client left after {ran_s:.0f}s; session stopped",
                flush=True)
        return "client_left"


def stop_session(holder: dict) -> tuple[int, dict]:
    """Decide a `POST /stop`: (status, body).

    The explicit end of a session, for the caller that cannot rely on its
    disconnect being seen: through Cloud Run's front end an aborted
    `/produce` client was observed not to reach the container for the
    whole remaining duration (stim-controller e2e, 2026-09-02), so the
    FemLed server says stop out loud and then waits for the lock to free
    before `/teardown`. Sets the running session's `stopping` flag; the
    pump writes the summary event and releases the lock as for any end.
    404 when nothing is running.
    """
    session = holder.get("session")
    if session is None:
        return 404, {"error": "no session is running"}
    session.stopping.set()
    return 200, {"status": "stopping", "run": session.run_name or None}


def serve(args, telemetry: Telemetry) -> int:
    """Cloud session mode: /healthz, /statz, /warmup, /teardown, /stop,
    /produce, and /overlay/* when the annotated view is configured.

    With --tee (a Confidential Space slot, tee_mode.py) the control routes
    take the trainer's OIDC token, the signalling routes take the phone's
    capability, and /attestation, /evidence-key, /lease, /ingest/source
    (an external camera, external_source.py) and /tunnel/expect,
    /tunnel/clear (the camera's home connector, camlink_gateway.py) join.
    """
    tee = None
    if getattr(args, "tee", False):
        tee = TeeMode.from_env()
        tee.start()
        print(f"tee: serving {tee.origin} for "
              f"{', '.join(tee.config.trainer_invoker_service_accounts)}",
              flush=True)
    server = build_server(args, telemetry, tee=tee)
    # The slot's own end (tee_mode.IdleExit): started here, not in
    # build_server, so a test's server never exits the test.
    idle_exit = getattr(server, "idle_exit", None)
    if idle_exit is not None:
        idle_exit.start()
        print(f"tee: idle exit after {idle_exit.idle_s:.0f}s without a lease or a "
              f"session, {idle_exit.boot_idle_s:.0f}s for a boot never leased",
              flush=True)
    # A slot booted for one session has nothing to wait for: the GPU boot
    # starts now rather than at the trainer's first /warmup poll, and the
    # session finds the pipeline hot. Not a touch of the idle clock - the
    # boot is the slot's own doing, not a sign of interest.
    if tee is not None or getattr(args, "prewarm", False):
        if server.start_warmup():
            print("prewarm: booting the GPU pose ahead of the first /warmup", flush=True)
    print(f"serving on :{server.server_address[1]}", flush=True)
    server.serve_forever()
    return 0


LEASE_BODY_CAP = 4096
EXTERNAL_VIEW_SIZE = "1280x720"


def external_view_args(session_args, external) -> bool:
    """Retune a /produce's annotated view when its stream is the external
    camera's path: a fixed camera behind the user is landscape and not a
    selfie, so the view goes out 16:9 and unmirrored, whatever the phone's
    own camera is published as (--overlay-size/--overlay-mirror in
    tee/entrypoint.sh). True when it did."""
    if external is None or getattr(session_args, "stream", "") != external.stream_url:
        return False
    session_args.overlay_mirror = False
    session_args.overlay_size = EXTERNAL_VIEW_SIZE
    return True


def build_server(args, telemetry: Telemetry,
                 tee: TeeMode | None = None,
                 external: ExternalSource | None = None) -> ThreadingHTTPServer:
    """The session server, bound and ready for serve_forever. `tee` (built
    by serve() from the environment, or handed in by a test) switches the
    handler into TEE mode; its attestation loop is the caller's to start.
    `external` is the slot's external camera (built here in TEE mode when
    the relay is configured; a test hands in one with its network calls
    replaced)."""
    busy = threading.Lock()
    # The session behind the busy lock, for /stop; set and cleared by the
    # /produce handler while it holds the lock.
    current: dict = {"session": None}
    warmup: dict = {"thread": None, "error": None}
    teardown = Teardown(
        busy, drain_s=getattr(args, "teardown_drain_s", TEARDOWN_DRAIN_S))
    # TEE only: the slot exits on its own when nobody holds it (the VM then
    # stops); the caller starts it (serve()).
    idle_exit = (IdleExit(teardown, tee.lease, idle_s=tee.config.idle_exit_s,
                          boot_idle_s=tee.config.boot_idle_s)
                 if tee is not None else None)

    # Instance boot overlaps the two slowest independent costs instead of
    # paying them serially inside the first /produce: the network-bound
    # model-store prefetch and the CPU-bound torch import. Both are
    # exactly-once and re-attempted by the first boot if they fail here.
    def preboot_prefetch() -> None:
        with teardown.working():
            try:
                prefetch_model_store(telemetry)
            except Exception as error:  # noqa: BLE001 - first boot retries
                print(f"preboot: prefetch failed ({error!r}); "
                      f"the first boot retries", flush=True)

    def preboot_import() -> None:
        with teardown.working():
            try:
                import pose_track  # noqa: F401 - heavy torch import, off-path
            except Exception as error:  # noqa: BLE001 - said aloud
                print(f"preboot: import failed ({error!r})", flush=True)

    threading.Thread(target=preboot_prefetch, daemon=True).start()
    threading.Thread(target=preboot_import, daemon=True).start()

    def start_warmup() -> bool:
        """Boot the process-wide GPU pose on a thread unless it is booted or
        booting: what /warmup does, and what serve() does at once on a slot
        that exists for one session (--prewarm). Never touches the /produce
        lock; a session arriving mid-boot joins it through acquire_gpu_pose's
        lock instead of waiting cold. True when a boot was started."""
        thread = warmup["thread"]
        if _GPU_POSE["pose"] is not None or (thread is not None and thread.is_alive()):
            return False
        warmup["error"] = None

        def warm() -> None:
            with teardown.working():
                try:
                    acquire_gpu_pose(args, telemetry)
                except Exception as error:  # noqa: BLE001
                    warmup["error"] = repr(error)
                    print(f"warmup failed: {error!r}", flush=True)

        thread = threading.Thread(target=warm, daemon=True)
        warmup["thread"] = thread
        thread.start()
        return True

    # WHEP signalling for the live overlay rides this port (relay_proxy.py);
    # nothing is served under /overlay unless a publish URL is configured.
    # The relay's loopback RTSP listener is wherever the overlay is
    # published to: that is where a relay path is read back from.
    publish = urlparse(getattr(args, "overlay_publish", "") or "")
    rtsp_base = (f"{publish.scheme}://{publish.netloc}"
                 if publish.scheme and publish.netloc else "rtsp://127.0.0.1:8554")
    relay = (RelayProxy(getattr(args, "overlay_relay_webrtc", "")
                        or "http://127.0.0.1:8889",
                        getattr(args, "overlay_relay_api", "")
                        or "http://127.0.0.1:9997",
                        public_origin=tee.origin if tee is not None else "",
                        rtsp_base=rtsp_base)
             if getattr(args, "overlay_publish", "") else None)
    # The external camera (external_source.py): TEE only, because the link
    # is a credential that must reach the enclave and nothing in front of
    # it. Outside TEE mode the route does not exist. Its gateway is the
    # home connector's (camlink_gateway.py, CAMLINK_GATEWAY_CONTROL).
    if external is None and tee is not None and relay is not None:
        external = ExternalSource(relay, own_ip=os.environ.get("TEE_PUBLIC_IP", ""),
                                  rtsp_base=rtsp_base,
                                  gateway=CamlinkGateway.from_env())
    if tee is None:
        external = None
    gateway = external.gateway if external is not None else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def _json(self, code: int, body: dict,
                  extra: list[tuple[str, str]] = ()) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        # -- TEE mode --------------------------------------------------------

        def _cors(self) -> list[tuple[str, str]]:
            """The phone's cross-origin headers, in TEE mode; nothing else."""
            if tee is None:
                return []
            return cors_headers(tee.config, self.headers.get("Origin"))

        def _not_found(self) -> None:
            """404 for a route that does not exist on this slot. A small
            request body is read first: answering and closing (HTTP/1.0)
            with unread bytes in the socket makes the kernel reset the
            connection, and the client may then never see the 404."""
            length = int(self.headers.get("Content-Length") or 0)
            if 0 < length <= LEASE_BODY_CAP:
                self.rfile.read(length)
            self.send_response(404)
            self.end_headers()

        def _control_gate(self) -> bool:
            """The trainer's OIDC token on a control route in TEE mode.
            True when the request may proceed; otherwise the 401 has been
            written. Outside TEE mode the front end (Cloud Run IAM) did
            this and the answer is always True."""
            if tee is None:
                return True
            ok, reason = tee.control_auth.check(self.headers)
            if ok:
                return True
            telemetry.count("teeControlRejects")
            self._json(401, {"error": "unauthorized", "reason": reason},
                       [("WWW-Authenticate", "Bearer")])
            return False

        def _tee_public(self, parsed) -> bool:
            """/attestation and /evidence-key: what the phone reads before
            it trusts this slot with its camera. Public, CORS for the page,
            and nothing in them is secret (the token is bound to this
            slot's audience and the key is the public half)."""
            if tee is None or parsed.path not in ("/attestation", "/evidence-key"):
                return False
            cors = self._cors()
            if self.command == "OPTIONS":
                self.send_response(204)
                for name, value in cors:
                    self.send_header(name, value)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True
            if self.command != "GET":
                self._json(405, {"error": "method not allowed"},
                           cors + [("Allow", "GET, OPTIONS")])
                return True
            if parsed.path == "/evidence-key":
                self._json(200, tee.evidence.describe(), cors)
                return True
            nonce = parse_qs(parsed.query).get("nonce", [None])[0]
            code, body = tee.attestation_body(nonce)
            self._json(code, body, cors + [("Cache-Control", "no-store")])
            return True

        def _lease(self) -> None:
            """POST /lease from the trainer: park the capability hash."""
            length = int(self.headers.get("Content-Length") or 0)
            if length > LEASE_BODY_CAP:
                self._json(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("not an object")
            except ValueError as error:
                self._json(400, {"error": f"invalid JSON: {error}"})
                return
            code, answer = tee.lease.grant(body)
            if code == 200:
                teardown.cancel("/lease")
                print(f"lease: session {answer['sessionId']} until "
                      f"{answer['expiresAt']}", flush=True)
                # A new phone must not inherit the last one's camera; the
                # same session re-leasing (a trainer restart) keeps it.
                if (external is not None and external.active
                        and external.session_id != answer["sessionId"]):
                    external.clear("lease for another session")
                # Nor the last one's home connector: an expectation posted
                # for another session is dropped with it.
                if (gateway is not None and gateway.session_id
                        and gateway.session_id != answer["sessionId"]):
                    gateway.clear_quietly("lease for another session")
            else:
                telemetry.count("teeLeaseRejects")
            self._json(code, answer)

        def _tunnel(self, path: str) -> None:
            """POST /tunnel/expect and /tunnel/clear from the trainer: which
            home connector the gateway (camlink_gateway.py) accepts for the
            leased session, and dropping it. The connector's key, the
            ticket hash and the expiry go to the gateway as they came;
            none of them is logged."""
            if path == "/tunnel/clear":
                try:
                    gateway.clear()
                except GatewayError:
                    telemetry.count("tunnelGatewayUnavailable")
                    self._json(502, {"error": "gateway unavailable"})
                    return
                telemetry.count("tunnelCleared")
                print("tunnel: cleared (the trainer asked)", flush=True)
                self._json(200, {"status": "cleared"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > LEASE_BODY_CAP:
                self._json(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("not an object")
            except ValueError as error:
                self._json(400, {"error": f"invalid JSON: {error}"})
                return
            try:
                session_id, expectation = parse_expectation(body, tee.clock())
            except ValueError as error:
                self._json(400, {"error": str(error)})
                return
            # The lease says whose slot this is; a connector may be expected
            # for that session only (after /stop there is no lease until the
            # trainer leases again, and the trainer retries then).
            if session_id != (tee.lease.snapshot()["sessionId"] or ""):
                self._json(409, {"error": "lease mismatch"})
                return
            try:
                gateway.expect(expectation)
            except GatewayError:
                telemetry.count("tunnelGatewayUnavailable")
                self._json(502, {"error": "gateway unavailable"})
                return
            gateway.session_id = session_id
            telemetry.count("tunnelExpected")
            print(f"tunnel: expecting a connector for session {session_id}",
                  flush=True)
            self._json(200, {"status": "expecting"})

        def _tee_signalling_gate(self, leg: str) -> str | None:
            """The phone's capability on a WHIP/WHEP/source request in TEE
            mode. The lease's session id when it opens; None once a 401 or
            400 has been written."""
            cors = self._cors()
            nonce = self.headers.get("X-Masseuse-Client-Nonce") or ""
            if nonce and not CLIENT_NONCE.match(nonce):
                self._json(400, {"error": "malformed client nonce"}, cors)
                return None
            ok, info = tee.lease.check(bearer_token(self.headers))
            if not ok:
                telemetry.count("teeCapabilityRejects")
                self._json(401, {"error": "unauthorized", "reason": info},
                           cors + [("WWW-Authenticate", "Bearer")])
                return None
            return info

        def _tee_answer(self, leg: str, code: int, headers, payload: bytes,
                        session_id: str) -> tuple[int, list, bytes] | None:
            """Dress a forwarded answer for the phone: the relay's own CORS
            replaced by ours, and on a 201 the signed DTLS fingerprint. An
            answer with no fingerprint is refused (and its relay session
            closed) rather than handed over unvouched."""
            kept = [(name, value) for name, value in headers
                    if not name.lower().startswith("access-control-")]
            kept.extend(self._cors())
            if self.command != "POST" or code != 201:
                return code, kept, payload
            location = next((value for name, value in kept
                             if name.lower() == "location"), "")
            secret = location_secret(location) or ""
            evidence = tee.evidence_for(
                role=leg, answer=payload,
                client_nonce=self.headers.get("X-Masseuse-Client-Nonce") or "",
                session_secret=secret, session_id=session_id)
            if evidence is None:
                telemetry.count("teeEvidenceMissingFingerprint")
                if secret:
                    relay.forward("DELETE", secret, {}, b"", leg=leg)
                return 502, self._cors() + [("Content-Type", "application/json")], \
                    json.dumps({"error": "answer carries no DTLS fingerprint"}).encode()
            telemetry.count("teeEvidenceIssued")
            kept.append(("X-Masseuse-Evidence", evidence))
            return code, kept, payload

        # -- the external camera ----------------------------------------------

        def _source(self) -> None:
            """/ingest/source: the camera the phone names instead of its own
            (external_source.py). PUT {url} connects - it blocks for the
            TLS probe and the relay's pull, twenty seconds at most - GET
            says which camera the session has, DELETE goes back to the
            phone's. Gated by the phone's capability like WHIP: the link
            is a credential, so it goes to the enclave and nowhere in
            front of it, and nothing about it (not the host) is logged.
            """
            cors = self._cors()
            if self.command == "OPTIONS":
                self.send_response(204)
                for name, value in cors:
                    self.send_header(name, value)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.command not in ("PUT", "GET", "DELETE"):
                self._json(405, {"error": "method not allowed"},
                           cors + [("Allow", "OPTIONS, PUT, GET, DELETE")])
                return
            session_id = self._tee_signalling_gate("source")
            if session_id is None:
                return
            if self.command == "GET":
                self._json(200, external.status(),
                           cors + [("Cache-Control", "no-store")])
                return
            if self.command == "DELETE":
                had = external.clear("the phone asked")
                if had:
                    telemetry.count("externalSourceRemoved")
                body = dict(external.status())
                body["status"] = "removed" if had else "none"
                self._json(200, body, cors)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > LEASE_BODY_CAP:
                self._json(413, {"status": "failed", "reason": "bad-url",
                                 "error": "body too large"}, cors)
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("not an object")
            except ValueError as error:
                self._json(400, {"status": "failed", "reason": "bad-url",
                                 "error": f"invalid JSON: {error}"}, cors)
                return
            try:
                answer = external.connect(body.get("url"), session_id)
            except SourceError as error:
                telemetry.count("externalSourceFailed")
                print(f"external camera: not connected ({error.reason}) for "
                      f"session {session_id}", flush=True)
                self._json(error.status, error.body(), cors)
                return
            except Exception as error:  # noqa: BLE001 - never the link itself
                telemetry.count("externalSourceFailed")
                print(f"external camera: failed ({type(error).__name__}) for "
                      f"session {session_id}", flush=True)
                self._json(500, {"status": "failed", "reason": "internal",
                                 "error": "the slot could not attach the camera"},
                           cors)
                return
            telemetry.count("externalSourceConnected")
            self._json(200, answer, cors)

        # -- the relay routes ------------------------------------------------

        def _overlay(self) -> bool:
            """Serve /overlay/* and /ingest/* if that is what this is; True
            when handled.

            Short and lock-free: these never take the /produce lock or a
            teardown hold - a subscriber's signalling must not keep an
            idle instance alive, and must not wait behind a session.

            The overlay leg needs a publishing session; the ingest leg (a
            phone's camera arriving over WHIP at the relay's `cam` path)
            deliberately does not: the trainer that owns this slot
            publishes the camera first and starts the session that reads
            it once /ingest/status shows the track. /ingest/source (the
            external camera) is the same shape with the relay pulling
            instead of the phone pushing; it exists in TEE mode only.
            """
            parsed = urlparse(self.path)
            matched = relay_route(parsed.path)
            if matched is None:
                return False
            if relay is None:
                self._json(404, {"error": "overlay is not configured"})
                return True
            kind, secret = matched
            if kind == "source":
                if external is None:
                    self._json(404, {"error": "an external camera needs a "
                                              "Confidential Space slot"})
                else:
                    self._source()
                return True
            if kind in ("status", "ingest-status"):
                if self.command != "GET":
                    self.send_response(405)
                    self.send_header("Allow", "GET")
                    self.end_headers()
                    return True
                if not self._control_gate():
                    return True
                code, body = (relay.status(current["session"])
                              if kind == "status" else relay.ingest_status(external))
                self._json(code, body)
                return True
            leg = "whip" if kind == "whip" else "whep"
            session_id = ""
            if tee is not None:
                # The phone's preflight is ours to answer: the relay never
                # sees a request the capability has not opened.
                if self.command == "OPTIONS":
                    self.send_response(204)
                    for name, value in self._cors():
                        self.send_header(name, value)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return True
                session_id = self._tee_signalling_gate(leg)
                if session_id is None:
                    return True
            if (leg == "whep" and current["session"] is None
                    and self.command != "OPTIONS"):
                self._json(503, {"error": "no session is running",
                                 "whep": "/overlay/whep"}, self._cors())
                return True
            if tee is not None and self.command == "POST" and secret is None:
                # One publisher and one subscriber per lease. Only the
                # lease's holder gets this far, so whatever holds the leg is
                # its own earlier session (a reload, a network change) and
                # is kicked rather than left to lock it out until the
                # relay's timeout.
                try:
                    kicked = relay.evict(leg)
                except Exception as error:  # noqa: BLE001 - said aloud
                    print(f"tee: evicting the {leg} leg failed: {error!r}",
                          flush=True)
                    kicked = 0
                if kicked:
                    telemetry.count("teeLegEvicted", kicked)
            length = int(self.headers.get("Content-Length") or 0)
            if length > relay.body_cap:
                self._json(413, {"error": "body too large"}, self._cors())
                return True
            body = self.rfile.read(length) if length else b""
            code, headers, payload = relay.forward(
                self.command, secret, self.headers, body,
                client=self.client_address[0], leg=leg)
            if tee is not None:
                code, headers, payload = self._tee_answer(
                    leg, code, headers, payload, session_id)
            self.send_response(code)
            for name, value in headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)
            return True

        def do_OPTIONS(self):
            if self._tee_public(urlparse(self.path)):
                return
            if not self._overlay():
                self.send_response(404)
                self.end_headers()

        def do_PATCH(self):
            if not self._overlay():
                self.send_response(404)
                self.end_headers()

        def do_DELETE(self):
            if not self._overlay():
                self.send_response(404)
                self.end_headers()

        def do_PUT(self):
            # WHEP: GET, HEAD and PUT on the endpoint are 405, not 501.
            if not self._overlay():
                self.send_response(404)
                self.end_headers()

        def do_HEAD(self):
            if not self._overlay():
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self._overlay():
                return
            parsed = urlparse(self.path)
            if self._tee_public(parsed):
                return
            if parsed.path == "/lease":
                if tee is None:
                    self._not_found()
                    return
                if self._control_gate():
                    self._lease()
                return
            if parsed.path in ("/tunnel/expect", "/tunnel/clear"):
                # The home connector's gateway exists on a TEE slot only,
                # like the external camera it serves.
                if tee is None or gateway is None:
                    self._not_found()
                    return
                if self._control_gate():
                    self._tunnel(parsed.path)
                return
            if parsed.path == "/stop":
                if not self._control_gate():
                    return
                code, body = stop_session(current)
                if code == 200:
                    print(f"stop: {body}", flush=True)
                    if tee is not None:
                        tee.lease.clear()
                payload = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                return
            if parsed.path != "/teardown":
                self.send_response(404)
                self.end_headers()
                return
            if not self._control_gate():
                return
            mode = parse_qs(parsed.query).get("mode", ["drain"])[0]
            if mode not in ("drain", "now"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"mode must be drain or now\n")
                return
            code, body = teardown.request(mode)
            if code == 200:
                print(f"teardown: {body}", flush=True)
                if tee is not None:
                    tee.lease.clear()
                if external is not None:
                    external.clear("teardown")
                if gateway is not None:
                    # The slot is ending: the connector is dropped so it is
                    # free for the next one (a no-op if the camera's clear
                    # above already did it).
                    gateway.clear_quietly("teardown")
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

        def do_GET(self):
            if self._overlay():
                return
            parsed = urlparse(self.path)
            if self._tee_public(parsed):
                return
            if parsed.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
                return
            if parsed.path in ("/statz", "/warmup", "/produce"):
                if not self._control_gate():
                    return
            if parsed.path == "/statz":
                snapshot = telemetry.snapshot()
                snapshot["teardown"] = teardown.snapshot()
                session = current["session"]
                overlay = getattr(session, "overlay", None) if session else None
                snapshot["overlay"] = (overlay.snapshot()
                                       if overlay is not None else None)
                if tee is not None:
                    snapshot["tee"] = tee.snapshot()
                    snapshot["tee"]["idleExit"] = idle_exit.snapshot()
                    snapshot["tee"]["externalCamera"] = (
                        external.status() if external is not None else None)
                body = json.dumps(snapshot).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/warmup":
                # The wake-up call: load weights into VRAM before the
                # session needs them (start_warmup above; on a slot that
                # booted for this session it is already under way).
                teardown.cancel("/warmup")
                if idle_exit is not None:
                    idle_exit.touch()
                start_warmup()
                status = ("ready" if _GPU_POSE["pose"] is not None
                          else "failed" if warmup["error"] else "warming")
                body = json.dumps({
                    "status": status,
                    "error": warmup["error"],
                    "bootMs": telemetry.snapshot()["bootMs"],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path != "/produce":
                self.send_response(404)
                self.end_headers()
                return
            if not busy.acquire(blocking=False):
                self.send_response(409)
                self.end_headers()
                self.wfile.write(b"a session is already running\n")
                return
            teardown.cancel("/produce")
            working = teardown.working()
            working.__enter__()
            try:
                params = parse_qs(parsed.query)
                session_args = argparse.Namespace(**vars(args))
                session_args.stream = params.get(
                    "stream", [args.stream])[0]
                session_args.duration = float(
                    params.get("duration", [args.duration or 0])[0])
                # The live-vs-sideload comparison runs both modes against
                # the same loop on the same instance.
                session_args.pose = params.get("pose", [args.pose])[0]
                requested_track = params.get("track")
                if requested_track is None:
                    session_args.track = args.track
                else:
                    mounted = mounted_path(requested_track[0])
                    if mounted is None:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(
                            f"track must be a capture under {MOUNT_ROOT}\n"
                            .encode())
                        return
                    session_args.track = mounted
                session_args.pose_fps = float(params.get(
                    "pose_fps", [getattr(args, "pose_fps", 6.0)])[0])
                # A named run lands its capture in gs://<bucket>/runs/<run>/
                # when the session ends; the name is a path segment there.
                session_args.run = params.get("run", [""])[0]
                if session_args.run and not RUN_NAME.match(session_args.run):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(
                        b"run must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}\n")
                    return
                # overlay_record=1 tees the annotated view to overlay.mp4 in
                # the capture (and so into the run's bucket prefix): the
                # offline check that the drawn skeleton sits on the body.
                session_args.overlay_record = params.get(
                    "overlay_record",
                    ["1" if getattr(args, "overlay_record", False) else "0"]
                )[0].lower() in ("1", "true", "yes")
                if tee is not None:
                    # A TEE run leaves no production-capture evidence by
                    # design: no annotated recording, whatever was asked.
                    session_args.overlay_record = False
                external_view_args(session_args, external)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                # Something on the wire at once, and a comment every few
                # seconds while Session() waits on a cold boot: a silent
                # stream is what the client and the front end give up on.
                keepalive = BootKeepalive(self.wfile)
                keepalive.hello(
                    "ready" if session_args.pose == "sideload"
                    else gpu_pose_state())
                keepalive.start()
                subscription = telemetry.subscribe()
                try:
                    session = Session(session_args, telemetry)
                except Exception as error:  # noqa: BLE001 - said aloud
                    keepalive.stop()
                    # The 200 and headers are already gone; a boot crash must
                    # arrive as an event, not as a silently empty stream.
                    self.wfile.write(
                        f"data: {json.dumps({'kind': 'error', 'error': repr(error)})}\n\n".encode())
                    self.wfile.flush()
                    telemetry.unsubscribe(subscription)
                    raise
                keepalive.stop()
                if keepalive.failed:
                    # The pump's first write meets the same broken pipe and
                    # winds the session down the usual way; say why here.
                    print("produce: client left during the boot", flush=True)
                current["session"] = session

                upload: dict = {"result": None}

                def run_then_upload() -> None:
                    # The upload rides the session thread, not the request:
                    # a client that dropped mid-session still gets its rows
                    # into the bucket when the session winds down.
                    try:
                        session.run()
                    finally:
                        upload["result"] = session.upload_capture(
                            getattr(args, "capture_bucket", ""))

                runner = threading.Thread(target=run_then_upload, daemon=True)
                runner.start()
                # Holds the request open for the session and, if the client
                # leaves first, stops the session and waits for it - so the
                # lock released below never frees a GPU that is still busy.
                pump_session(session, runner, self.wfile, subscription,
                             telemetry,
                             upload_result=lambda: upload["result"])
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                current["session"] = None
                working.__exit__(None, None, None)
                busy.release()

    # TEE: loopback only. The container shares the VM's network namespace,
    # so a wildcard bind would put the plain-HTTP control routes on the
    # VM's address; Caddy on :443 is the only way in (the firewall says the
    # same, this is the belt to its braces).
    server = ThreadingHTTPServer(("127.0.0.1" if tee is not None else "0.0.0.0", args.port), Handler)
    # For tests and /statz: the live session, whatever thread holds it.
    server.current = current  # type: ignore[attr-defined]
    # TEE: the watchdog serve() starts (None outside TEE mode).
    server.idle_exit = idle_exit  # type: ignore[attr-defined]
    # TEE: the external camera and its home connector's gateway, for tests
    # and /statz.
    server.external = external  # type: ignore[attr-defined]
    server.gateway = gateway  # type: ignore[attr-defined]
    # The GPU boot, for serve() to start ahead of the first /warmup.
    server.start_warmup = start_warmup  # type: ignore[attr-defined]
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="")
    parser.add_argument("--pose", choices=("gpu", "sideload"),
                        default="gpu")
    parser.add_argument("--pose-fps", type=float, default=6.0,
                        help="pose cadence; the bridge widens with it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", default="",
                        help="for --pose sideload: a fetched production "
                             "capture (its directory, or its poses.jsonl) "
                             "whose rows stand in for the GPU")
    parser.add_argument("--analysis-socket",
                        default=os.environ.get("ANALYSIS_SOCKET", ""),
                        help="Unix socket of the analysis process "
                             "(analysis/protocol.md); empty = run the pixel "
                             "path alone, post nothing")
    parser.add_argument("--sink-dir", default="/tmp/producer-capture")
    parser.add_argument("--run", default="",
                        help="name this session's capture; it lands in "
                             "<sink-dir>/<run> and, with --capture-bucket, "
                             "in gs://<bucket>/runs/<run>/ when the session "
                             "ends. /produce takes the same as run=")
    parser.add_argument("--capture-bucket",
                        default=os.environ.get("POSE_PREFETCH_BUCKET", ""),
                        help="GCS bucket named runs upload to; defaults to "
                             "the model-store bucket")
    parser.add_argument("--post-url", default="")
    parser.add_argument("--post-interval-s", type=float, default=1.0,
                        help="reading cadence the analysis is asked for")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--input-lost-after", type=float, default=0.0,
                        help="end the session once the stream has delivered "
                             "no frames for this many seconds (a camera that "
                             "left the relay and did not come back); 0 = "
                             "keep reconnecting for the whole duration")
    # The live annotated view (overlay.py, relay_proxy.py). Off unless a
    # publish URL is given; then every session paints the full Sapiens2
    # result on the frame it was detected on and publishes it to the relay,
    # and /overlay/whep on this port signals WebRTC subscribers to it.
    parser.add_argument("--overlay-publish", default="",
                        help="RTSP URL the annotated view is published to "
                             "(the relay sidecar's overlay path, e.g. "
                             "rtsp://127.0.0.1:8554/overlay); empty = off")
    parser.add_argument("--overlay-size", default="1280x720",
                        help="annotated view size, WxH, even")
    parser.add_argument("--overlay-fps", type=float, default=15.0,
                        help="annotated view cadence; frames are duplicated "
                             "or skipped to hold it")
    parser.add_argument("--overlay-delay-s", type=float, default=1.0,
                        help="how far behind the decode the view runs, so "
                             "each frame is drawn with the pose detected on "
                             "it (pose latency + one pose interval)")
    parser.add_argument("--overlay-mirror", action="store_true",
                        help="publish the annotated view as a selfie: picture "
                             "and skeleton flipped left-to-right, lettering "
                             "still readable, so a phone showing its own "
                             "camera mirrored can switch to it seamlessly")
    parser.add_argument("--overlay-encoder", choices=("x264", "nvenc", "auto"),
                        default="x264",
                        help="H.264 encoder: x264 ultrafast (default, needs "
                             "nothing from the driver), nvenc, or auto "
                             "(nvenc if a probe succeeds)")
    parser.add_argument("--overlay-bitrate", default="3M")
    parser.add_argument("--overlay-record", action="store_true",
                        help="also write the annotated view to overlay.mp4 "
                             "in the capture directory (uploaded with the "
                             "run). /produce takes overlay_record=1")
    parser.add_argument("--overlay-relay-webrtc",
                        default="http://127.0.0.1:8889",
                        help="the relay's WHEP endpoint, loopback")
    parser.add_argument("--overlay-relay-api",
                        default="http://127.0.0.1:9997",
                        help="the relay's control API, loopback")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--tee", action="store_true",
                        help="Confidential Space slot (tee_mode.py): the "
                             "trainer's OIDC token gates the control routes, "
                             "the phone's capability gates WHIP/WHEP, every "
                             "answer carries attestation-bound evidence, and "
                             "no capture leaves the enclave. Needs "
                             "TEE_PUBLIC_HOST and "
                             "TRAINER_INVOKER_SERVICE_ACCOUNT; implies --serve "
                             "and --prewarm")
    parser.add_argument("--prewarm", action="store_true",
                        help="boot the GPU pose as soon as the server is up "
                             "instead of at the first /warmup (an instance "
                             "that exists for one session)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--teardown-drain-s", type=float,
                        default=TEARDOWN_DRAIN_S,
                        help="how long the instance must have been quiet "
                             "before a /teardown exits: long enough for the "
                             "autoscaler's CPU driver to stop recommending "
                             "an instance (measured gone by 148 s), so the "
                             "exit is not answered with a replacement "
                             "instance. /teardown?mode=now skips it")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    telemetry = Telemetry()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if args.tee:
        # The enclave's invariants, enforced here as well as in the image's
        # baked arguments: nothing a session records may leave it.
        if args.capture_bucket:
            raise SystemExit("--tee forbids --capture-bucket: a TEE run "
                             "uploads no capture")
        if args.overlay_record:
            raise SystemExit("--tee forbids --overlay-record")
        args.serve = True
    if args.serve:
        return serve(args, telemetry)
    if not args.stream:
        raise SystemExit("--stream is required outside --serve")
    if args.run and not RUN_NAME.match(args.run):
        raise SystemExit("--run must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    session = Session(args, telemetry)
    summary = session.run()
    upload = session.upload_capture(args.capture_bucket)
    print(json.dumps({
        "bootMs": summary["bootMs"],
        "counters": summary["counters"],
        "gauges": summary["gauges"],
        "analysis": summary.get("analysis"),
        "run": summary["run"],
        "captureDir": summary["captureDir"],
        "upload": upload,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
