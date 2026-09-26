"""The live stream: the annotated view sent on to one destination the user
names, from their phone, for one session.

By default the annotated view goes back to the device that sent the
video and nowhere else (README). This module is the one exception, and
it is the user's to open: the page hands the enclave a live-streaming
destination (`PUT /ingest/egress`, capability bearer, over the slot's own
TLS, after the phone has verified the attestation), the way it hands it
a camera link (external_source.py). The trainer and Cloudflare never see
the destination; the trainer learns from /ingest/status that a stream is
on and to which host, and may stop it (`POST /egress/stop`, its own
identity), never start one or name where it goes. Nothing about the
destination beyond its host is written to a log or handed to anyone: the
stream key rides in the URL's path and stays inside the process.

What the enclave checks before it connects:

  scheme      rtmps:// only. A plain rtmp:// destination would carry the
              view across the internet in the clear.
  address     the host must resolve to a public address (the same rule as
              a camera link: RFC 1918, loopback, link-local, multicast and
              the VM's own address are refused with a reason the page can
              show). A stream goes to a service on the internet, never
              into the enclave's own network.

Then one ffmpeg: the view read back from the relay's loopback RTSP path
(`overlay`, the same H.264 the phone's WHEP leg carries), copied without
re-encoding, muxed to FLV and pushed to the destination; with `audio`
the microphone's Opus from the camera path as AAC beside it, silence
otherwise; with `hud` the HUD card (hud_card.py) composited over the
view at the bottom-left through a second, raw RGBA input the card feeds
twice a second, which means one re-encode (libx264 ultrafast, or NVENC
where the overlay's own publisher uses it). A process that exits is
respawned with backoff while the stream is wanted (the production
restarts on a camera change; the destination sees the stream drop and
come back). The stream lives for the lease: cleared on the phone's
DELETE, the trainer's stop, /teardown, and a lease for another session.
Not on /stop, which is how the trainer restarts a production.
"""

from __future__ import annotations

import ipaddress
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit

from external_source import _refused_address

URL_CAP = 2048
DEFAULT_RTMPS_PORT = 443
ALLOWED_SCHEMES = ("rtmps",)
BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)
HUD_FPS = 2.0
STOP_WAIT_S = 3.0


class EgressError(Exception):
    """A refusal or a failure, with a reason code the page shows the user
    and the HTTP status the handler answers with."""

    def __init__(self, reason: str, message: str, status: int = 400):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status = status

    def body(self) -> dict:
        return {"status": "failed", "reason": self.reason, "error": self.message}


def parse_destination(url) -> tuple[str, str, int]:
    """(the destination, stripped; its host; its port) for anything shaped
    like a live-streaming service's rtmps:// address, or an EgressError
    saying what is wrong with it. Nothing here touches the network."""
    if not isinstance(url, str) or not url.strip():
        raise EgressError("bad-url", "paste the service's rtmps:// address, stream key included")
    url = url.strip()
    if len(url) > URL_CAP:
        raise EgressError("bad-url", f"the address is longer than {URL_CAP} characters")
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise EgressError("bad-url", "the address contains spaces or control characters")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as error:
        raise EgressError("bad-url", f"that is not a valid address ({error})") from None
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise EgressError("not-rtmps", "only rtmps:// addresses are accepted: a plain "
                          "rtmp:// address would send the view across the internet "
                          "unencrypted")
    if not host:
        raise EgressError("bad-url", "the address names no host")
    if not parts.path or parts.path == "/":
        raise EgressError("bad-url", "the address needs the service's application path and your stream key")
    return url, host, int(port or DEFAULT_RTMPS_PORT)


def resolve_public(host: str, port: int, own_ip: str = "",
                   resolver=socket.getaddrinfo) -> str:
    """The address the destination resolves to, or an EgressError: every
    resolved address must be a public one, and not this VM's."""
    try:
        found = resolver(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise EgressError("unreachable", f"{host} does not resolve", 502) from None
    addresses = []
    for entry in found:
        candidate = entry[4][0] if len(entry) >= 5 else None
        if not candidate:
            continue
        try:
            addresses.append(ipaddress.ip_address(candidate.split("%", 1)[0]))
        except ValueError:
            continue
    if not addresses:
        raise EgressError("unreachable", f"{host} does not resolve", 502)
    own = None
    if own_ip:
        try:
            own = ipaddress.ip_address(own_ip)
        except ValueError:
            own = None
    for ip in addresses:
        if _refused_address(ip) or (own is not None and ip == own):
            raise EgressError("private-address",
                              "the stream must go to a service on the internet: "
                              f"{host} resolves to an address inside a private "
                              "or reserved range")
    return str(addresses[0])


class EgressPublisher:
    """One ffmpeg from the relay to the destination, supervised: spawned
    when started, respawned with backoff while wanted, stopped on demand.
    With a `hud` card the card's picture is written to the process twice
    a second from a thread of this publisher's."""

    def __init__(self, destination: str, view_url: str, *, audio_url: str | None = None,
                 hud=None, encoder: str = "x264", telemetry=None,
                 popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep,
                 log=print):
        self.destination = destination
        self.view_url = view_url
        self.audio_url = audio_url
        self.hud = hud
        self.encoder = encoder
        self.telemetry = telemetry
        self.popen = popen
        self.clock = clock
        self.sleep = sleep
        self.log = log
        self.proc = None
        self.spawns = 0
        self.failures = 0
        self.last_error: str | None = None
        self.started_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- the command --------------------------------------------------------

    def argv(self) -> list[str]:
        argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp", "-i", self.view_url]
        audio_index = None
        hud_index = None
        if self.audio_url:
            argv += ["-rtsp_transport", "tcp", "-i", self.audio_url]
            audio_index = 1
        if self.hud is not None:
            width, height = self.hud.size
            hud_index = 2 if audio_index is not None else 1
            argv += ["-f", "rawvideo", "-pixel_format", "rgba",
                     "-video_size", f"{width}x{height}",
                     "-framerate", f"{HUD_FPS:g}", "-i", "pipe:0"]
        if hud_index is None:
            argv += ["-map", "0:v", "-c:v", "copy"]
        else:
            argv += ["-filter_complex",
                     f"[0:v][{hud_index}:v]overlay=x=16:y=main_h-overlay_h-16:format=auto[v]",
                     "-map", "[v]"]
            if self.encoder == "nvenc":
                argv += ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-zerolatency", "1", "-rc", "cbr"]
            else:
                argv += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
            argv += ["-profile:v", "main", "-pix_fmt", "yuv420p", "-g", "60", "-bf", "0",
                     "-b:v", "4500k", "-maxrate", "4500k", "-bufsize", "9000k"]
        if audio_index is not None:
            argv += ["-map", f"{audio_index}:a", "-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
        else:
            argv += ["-an"]
        argv += ["-f", "flv", self.destination]
        return argv

    # -- the process --------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.started_at = self.clock()
        self._thread = threading.Thread(target=self._supervise, name="egress", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = STOP_WAIT_S) -> None:
        self._stop.set()
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                proc.terminate()
                proc.wait(timeout=timeout_s)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def _spawn(self) -> bool:
        self.spawns += 1
        try:
            proc = self.popen(self.argv(),
                              stdin=subprocess.PIPE if self.hud is not None else subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as error:
            self._failed(f"spawn failed: {error!r}")
            return False
        with self._lock:
            self.proc = proc
        if self.telemetry is not None:
            self.telemetry.count("egressSpawns")
        return True

    def _failed(self, why: str) -> None:
        self.failures += 1
        self.last_error = why[:200]
        if self.telemetry is not None:
            self.telemetry.count("egressRestarts")
        # The destination's host would be in ffmpeg's own words; ours name
        # neither host nor key.
        self.log(f"egress: {why}", flush=True)

    def _feed_hud(self, proc) -> None:
        """The card's picture into the process, twice a second, until the
        process goes or the stream is stopped."""
        period = 1.0 / HUD_FPS
        while not self._stop.is_set() and proc.poll() is None:
            try:
                proc.stdin.write(self.hud.render())
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return
            self.sleep(period)

    def _supervise(self) -> None:
        while not self._stop.is_set():
            if not self._spawn():
                self._wait(self.failures)
                continue
            proc = self.proc
            if proc is None:
                break
            feeder = None
            if self.hud is not None:
                feeder = threading.Thread(target=self._feed_hud, args=(proc,), name="egress-hud", daemon=True)
                feeder.start()
            stderr = b""
            try:
                _, stderr = proc.communicate()
            except (OSError, ValueError):
                pass
            if feeder is not None:
                feeder.join(timeout=2.0)
            if self._stop.is_set():
                break
            code = proc.returncode
            tail = (stderr or b"")[-200:].decode("utf-8", "replace").strip()
            # ffmpeg's words may carry the destination: only the exit code
            # and a bounded, host-free tail are kept.
            tail = " ".join(part for part in tail.split() if "://" not in part)
            self._failed(f"ffmpeg exited {code}{': ' + tail if tail else ''}")
            with self._lock:
                if self.proc is proc:
                    self.proc = None
            self._wait(self.failures)
        with self._lock:
            self.proc = None

    def _wait(self, failures: int) -> None:
        backoff = BACKOFF_S[min(failures, len(BACKOFF_S)) - 1]
        deadline = self.clock() + backoff
        while not self._stop.is_set() and self.clock() < deadline:
            self.sleep(min(0.25, max(0.0, deadline - self.clock())))

    def snapshot(self) -> dict:
        return {"alive": self.alive, "spawns": self.spawns, "failures": self.failures,
                "error": self.last_error}


class Egress:
    """The one live stream per lease and its handlers' model.

    `connect(body, session_id)` validates and starts (replacing a stream
    already on); `status()` is what /ingest/status carries (`active`,
    `host`, `since`, `audio`, `hud`, `error`, `restarts`; never the
    address); `clear(reason)` stops. `set_hud_state` takes the trainer's
    HUD card state (hud_card.py) for the stream.
    """

    def __init__(self, view_url: str, audio_url: str | None, *, own_ip: str = "",
                 hud_card=None, encoder: str = "x264", telemetry=None,
                 resolver=socket.getaddrinfo, popen=subprocess.Popen,
                 clock=time.time, monotonic=time.monotonic, log=print):
        self.view_url = view_url
        self.audio_url = audio_url
        self.own_ip = own_ip
        self.hud_card = hud_card
        self.encoder = encoder
        self.telemetry = telemetry
        self.resolver = resolver
        self.popen = popen
        self.clock = clock
        self.monotonic = monotonic
        self.log = log
        self._lock = threading.Lock()
        self.publisher: EgressPublisher | None = None
        self.host: str | None = None
        self.since: float | None = None
        self.audio = False
        self.hud = False
        self.session_id: str | None = None

    @property
    def active(self) -> bool:
        return self.publisher is not None

    def connect(self, body: dict, session_id: str) -> dict:
        url = body.get("url") if isinstance(body, dict) else None
        audio = bool(body.get("audio")) if isinstance(body, dict) else False
        hud = bool(body.get("hud", True)) if isinstance(body, dict) else True
        destination, host, port = parse_destination(url)
        resolve_public(host, port, own_ip=self.own_ip, resolver=self.resolver)
        with self._lock:
            self._stop_locked("replaced")
            publisher = EgressPublisher(
                destination, self.view_url,
                audio_url=self.audio_url if audio else None,
                hud=self.hud_card if hud and self.hud_card is not None else None,
                encoder=self.encoder, telemetry=self.telemetry, popen=self.popen,
                clock=self.monotonic, log=self.log)
            publisher.start()
            self.publisher = publisher
            self.host = host
            self.since = self.clock()
            self.audio = audio
            self.hud = hud and self.hud_card is not None
            self.session_id = session_id
        if self.telemetry is not None:
            self.telemetry.count("egressStarted")
        self.log(f"egress: on, to {host} (audio={'on' if audio else 'off'}, "
                 f"hud={'on' if self.hud else 'off'}) for session {session_id}", flush=True)
        return {"status": "connected", **self.status()}

    def _stop_locked(self, reason: str) -> bool:
        publisher, self.publisher = self.publisher, None
        if publisher is None:
            return False
        publisher.stop()
        self.log(f"egress: off ({reason})", flush=True)
        return True

    def clear(self, reason: str) -> bool:
        with self._lock:
            had = self._stop_locked(reason)
            self.host = None
            self.since = None
            self.audio = False
            self.hud = False
            self.session_id = None
        if had and self.telemetry is not None:
            self.telemetry.count("egressStopped")
        return had

    def set_hud_state(self, body: dict) -> dict:
        if self.hud_card is None:
            raise ValueError("no HUD card on this slot")
        return self.hud_card.set_state(body)

    def status(self) -> dict:
        with self._lock:
            publisher = self.publisher
            snapshot = publisher.snapshot() if publisher is not None else None
            return {
                "active": publisher is not None,
                "host": self.host,
                "since": self.since,
                "audio": self.audio,
                "hud": self.hud,
                "error": snapshot["error"] if snapshot else None,
                "restarts": snapshot["failures"] if snapshot else 0,
                "alive": snapshot["alive"] if snapshot else False,
                "card": self.hud_card.describe() if self.hud_card is not None else None,
            }
