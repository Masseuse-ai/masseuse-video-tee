"""The phone's picture to the person's own computer, for OBS to work on.

By default the phone's camera goes nowhere but the analysis and the view
sent back to the phone (README). This module is the one other place it
goes, and only at the person's asking, twice over: they ask on their
computer (`masseuse-camlink -share-phone on`, which the connector reports
to the service and which is the only thing that makes the connector accept
the picture), and their phone, seeing that, asks the enclave to send it
(`PUT /ingest/share`, capability bearer, over the slot's own TLS, after the
phone has verified the attestation). The trainer never starts it and never
sees the picture; it learns from /ingest/status that it is on.

Where it goes is the connector's own endpoint, and nowhere else: the
gateway's own-endpoint listener (127.0.0.1:7442, camlink_gateway.py) carries
every connection to the attached connector's `127.0.0.1:7443`, the RTSPS
server inside the connector process, through the tunnel the connector
dialled to this enclave after verifying it. So the picture leaves the
enclave only inside the connector's tunnel, to the connector the session
is bound to, on the computer the person is sitting at.

How: one ffmpeg copies the phone's video track off the relay's loopback
RTSP path (`cam`, the H.264 the phone sent, `-c copy`: no decoding, no
re-encoding, no GPU) and publishes it as an RTSP RECORD into the
connector's `phone` path. The TLS to the connector is not ffmpeg's: the
connector's certificate is pinned (its SHA-256, probed through the same
listener the way external_source.py pins a camera through the relay
listener) and ffmpeg's plain RTSP goes over a loopback bridge of this
process's (TlsBridge), which completes TLS to the connector, checks the
leaf it sees is the pinned one, byte for byte, and relays; a leaf that
differs is refused and nothing is sent. Loopback both sides of the bridge
is what the WHIP leg already is (the relay's own paths).

The process is supervised: respawned with backoff while the share is
wanted (the connector's tunnel re-dials after a network blip; the picture
resumes when it is back), stopped on demand. A connector that answers the
publish with a refusal (403: it was not asked, an older release) is
`connector-refused` and is not retried until the phone asks again. The
share lives for the lease: cleared on the phone's DELETE, /teardown and a
lease for another session; not on /stop, which is how the trainer restarts
a production.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
import subprocess
import threading
import time

OWN_HOST = "127.0.0.1"
OWN_PORT = 7442
CONNECTOR_TARGET = "127.0.0.1:7443"
PHONE_PATH = "phone"
PROBE_TIMEOUT_S = 6.0
BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)
STOP_WAIT_S = 3.0
RELAY_CHUNK = 64 * 1024
# What ffmpeg says when the connector will not take the publish; matched
# on the exit's stderr tail. 403: not asked for on the computer (or an
# older connector); 405: a connector with no phone path at all.
REFUSAL_MARKS = ("403", "405", "Forbidden", "Method Not Allowed")


class ShareError(Exception):
    """A refusal or a failure, with a reason code the page shows the user
    and the HTTP status the handler answers with."""

    def __init__(self, reason: str, message: str, status: int = 400):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status = status

    def body(self) -> dict:
        return {"status": "failed", "reason": self.reason, "error": self.message}


def probe_connector(host: str = OWN_HOST, port: int = OWN_PORT,
                    timeout_s: float = PROBE_TIMEOUT_S) -> str:
    """The SHA-256 (64 lowercase hex) of the leaf the connector presents
    through the gateway's own-endpoint listener, or a ShareError. No SNI:
    the connector's certificate names 127.0.0.1, and the gateway is not
    told anything about where the connection goes (it always goes to the
    connector's own endpoint)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # the leaf is pinned by its hash, not a chain
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as raw:
            with context.wrap_socket(raw, server_hostname=None) as tls:
                der = tls.getpeercert(binary_form=True)
    except ssl.SSLError:
        raise ShareError("connector-offline", "the connector did not complete TLS: "
                         "is masseuse-camlink running and paired with this session?",
                         502) from None
    except (TimeoutError, socket.timeout):
        raise ShareError("connector-offline", "the connector did not answer within "
                         f"{timeout_s:.0f}s", 502) from None
    except OSError as error:
        raise ShareError("connector-offline", "the connector could not be reached "
                         f"({type(error).__name__})", 502) from None
    if not der:
        raise ShareError("connector-offline", "the connector presented no certificate", 502)
    return hashlib.sha256(der).hexdigest()


class TlsBridge:
    """A loopback listener whose every accepted connection is carried to
    `target` inside TLS, provided the leaf presented there is `pin`
    (SHA-256 hex): what ffmpeg's plain RTSP publish goes through. Refused
    connections are closed without a byte relayed."""

    def __init__(self, pin: str, target: tuple[str, int] = (OWN_HOST, OWN_PORT), *,
                 connect_timeout_s: float = PROBE_TIMEOUT_S, log=print):
        self.pin = pin.lower()
        self.target = target
        self.connect_timeout_s = connect_timeout_s
        self.log = log
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self.accepted = 0
        self.refused = 0
        self.port: int | None = None

    def start(self) -> int:
        """Listen on a loopback port the system chooses; the port."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.5)
        self._listener = listener
        self.port = int(listener.getsockname()[1])
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name="share-bridge", daemon=True)
        self._thread.start()
        return self.port

    @property
    def url(self) -> str:
        return f"rtsp://127.0.0.1:{self.port}/{PHONE_PATH}"

    def stop(self) -> None:
        self._stop.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            conns, self._conns = set(self._conns), set()
        for conn in conns:
            try:
                conn.close()
            except OSError:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_WAIT_S)

    def _accept_loop(self) -> None:
        listener = self._listener
        while not self._stop.is_set() and listener is not None:
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), name="share-bridge-conn",
                             daemon=True).start()

    def _connect(self) -> ssl.SSLSocket:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # the leaf is pinned below, byte for byte
        raw = socket.create_connection(self.target, timeout=self.connect_timeout_s)
        tls = context.wrap_socket(raw, server_hostname=None)
        der = tls.getpeercert(binary_form=True) or b""
        seen = hashlib.sha256(der).hexdigest()
        if seen != self.pin:
            tls.close()
            raise ssl.SSLCertVerificationError("the connector's certificate is not the pinned one")
        tls.settimeout(None)
        return tls

    def _serve(self, conn: socket.socket) -> None:
        try:
            upstream = self._connect()
        except ssl.SSLCertVerificationError:
            self.refused += 1
            self.log("share: the connector's certificate changed; not sending", flush=True)
            conn.close()
            return
        except (OSError, ssl.SSLError):
            self.refused += 1
            conn.close()
            return
        self.accepted += 1
        with self._lock:
            self._conns.update((conn, upstream))
        conn.settimeout(None)
        done = threading.Event()
        threading.Thread(target=self._pump, args=(conn, upstream, done), daemon=True).start()
        threading.Thread(target=self._pump, args=(upstream, conn, done), daemon=True).start()
        done.wait()
        for sock in (conn, upstream):
            try:
                sock.close()
            except OSError:
                pass
        with self._lock:
            self._conns.discard(conn)
            self._conns.discard(upstream)

    @staticmethod
    def _pump(src, dst, done: threading.Event) -> None:
        try:
            while True:
                chunk = src.recv(RELAY_CHUNK)
                if not chunk:
                    break
                dst.sendall(chunk)
        except (OSError, ssl.SSLError, ValueError):
            pass
        finally:
            done.set()


class SharePublisher:
    """One ffmpeg from the relay's `cam` path into the connector's phone
    path through the bridge, supervised: spawned when started, respawned
    with backoff while wanted, stopped on demand, and stopped for good
    when the connector refuses the publish."""

    def __init__(self, cam_url: str, bridge_url: str, *, telemetry=None,
                 popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep, log=print):
        self.cam_url = cam_url
        self.bridge_url = bridge_url
        self.telemetry = telemetry
        self.popen = popen
        self.clock = clock
        self.sleep = sleep
        self.log = log
        self.proc = None
        self.spawns = 0
        self.failures = 0
        self.refused = False
        self.last_error: str | None = None
        self.started_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def argv(self) -> list[str]:
        return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp", "-i", self.cam_url,
                "-an", "-map", "0:v", "-c:v", "copy",
                "-f", "rtsp", "-rtsp_transport", "tcp", self.bridge_url]

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.started_at = self.clock()
        self._thread = threading.Thread(target=self._supervise, name="share", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = STOP_WAIT_S) -> None:
        self._stop.set()
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is not None:
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
            proc = self.popen(self.argv(), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError as error:
            self._failed(f"spawn failed: {error!r}")
            return False
        with self._lock:
            self.proc = proc
        if self.telemetry is not None:
            self.telemetry.count("shareSpawns")
        return True

    def _failed(self, why: str) -> None:
        self.failures += 1
        self.last_error = why[:200]
        if self.telemetry is not None:
            self.telemetry.count("shareRestarts")
        self.log(f"share: {why}", flush=True)

    def _supervise(self) -> None:
        while not self._stop.is_set():
            if not self._spawn():
                self._wait(self.failures)
                continue
            proc = self.proc
            if proc is None:
                break
            stderr = b""
            try:
                _, stderr = proc.communicate()
            except (OSError, ValueError):
                pass
            if self._stop.is_set():
                break
            code = proc.returncode
            tail = (stderr or b"")[-200:].decode("utf-8", "replace").strip()
            tail = " ".join(part for part in tail.split() if "://" not in part)
            with self._lock:
                if self.proc is proc:
                    self.proc = None
            if any(mark in tail for mark in REFUSAL_MARKS):
                self.refused = True
                self.last_error = "connector-refused"
                self.log("share: the connector refused the phone's picture (not asked for "
                         "on the computer, or an older release); not retrying", flush=True)
                if self.telemetry is not None:
                    self.telemetry.count("shareRefused")
                break
            self._failed(f"ffmpeg exited {code}{': ' + tail if tail else ''}")
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
                "refused": self.refused, "error": self.last_error}


class Share:
    """The one share per lease and its handlers' model.

    `connect(session_id)` checks the connector is attached, pins its leaf
    and starts the bridge and the publisher (replacing a share already
    on); `status()` is what /ingest/status carries (`active`, `since`,
    `error`, `restarts`, `alive`, `refused`); `clear(reason)` stops.
    """

    def __init__(self, cam_url: str, *, gateway=None, telemetry=None,
                 probe=probe_connector, bridge_factory=TlsBridge,
                 popen=subprocess.Popen, clock=time.time, monotonic=time.monotonic,
                 log=print):
        self.cam_url = cam_url
        self.gateway = gateway
        self.telemetry = telemetry
        self.probe = probe
        self.bridge_factory = bridge_factory
        self.popen = popen
        self.clock = clock
        self.monotonic = monotonic
        self.log = log
        self._lock = threading.Lock()
        self.publisher: SharePublisher | None = None
        self.bridge = None
        self.since: float | None = None
        self.session_id: str | None = None
        self.pin: str | None = None

    @property
    def active(self) -> bool:
        return self.publisher is not None

    def connect(self, session_id: str) -> dict:
        gateway = self.gateway
        if gateway is None:
            raise ShareError("connector-offline", "this slot has no tunnel to a computer", 502)
        live = gateway.status()
        if live is None:
            raise ShareError("connector-offline", "the slot's tunnel gateway is not answering", 502)
        if not live.get("connected"):
            raise ShareError("connector-offline", "the connector is not attached: run "
                             "masseuse-camlink on your computer and pair it with this "
                             "session", 502)
        pin = self.probe()
        with self._lock:
            self._stop_locked("replaced")
            bridge = self.bridge_factory(pin, log=self.log)
            bridge.start()
            publisher = SharePublisher(self.cam_url, bridge.url, telemetry=self.telemetry,
                                       popen=self.popen, clock=self.monotonic, log=self.log)
            publisher.start()
            self.bridge = bridge
            self.publisher = publisher
            self.pin = pin
            self.since = self.clock()
            self.session_id = session_id
        if self.telemetry is not None:
            self.telemetry.count("shareStarted")
        self.log(f"share: on, the phone's picture to the connector for session {session_id}",
                 flush=True)
        return {"status": "connected", **self.status()}

    def _stop_locked(self, reason: str) -> bool:
        publisher, self.publisher = self.publisher, None
        bridge, self.bridge = self.bridge, None
        if publisher is not None:
            publisher.stop()
        if bridge is not None:
            bridge.stop()
        if publisher is None:
            return False
        self.log(f"share: off ({reason})", flush=True)
        return True

    def clear(self, reason: str) -> bool:
        with self._lock:
            had = self._stop_locked(reason)
            self.since = None
            self.session_id = None
            self.pin = None
        if had and self.telemetry is not None:
            self.telemetry.count("shareStopped")
        return had

    def status(self) -> dict:
        with self._lock:
            publisher = self.publisher
            snapshot = publisher.snapshot() if publisher is not None else None
            return {
                "active": publisher is not None,
                "since": self.since,
                "error": snapshot["error"] if snapshot else None,
                "restarts": snapshot["failures"] if snapshot else 0,
                "alive": snapshot["alive"] if snapshot else False,
                "refused": bool(snapshot["refused"]) if snapshot else False,
            }
