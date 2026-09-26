"""The phone's picture to the connector (share.py): the command, the gate
on the connector, the pinned bridge, the refusal, the supervision."""

from __future__ import annotations

import datetime
import hashlib
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import share  # noqa: E402
from share import Share, ShareError, SharePublisher, TlsBridge  # noqa: E402


class FakeProc:
    """A scripted ffmpeg: `outcome` is None (runs until terminated) or
    (returncode, stderr) for an exit on its own after `after_s`."""

    def __init__(self, argv, outcome=None, after_s: float = 0.0):
        self.argv = argv
        self.returncode = None
        self.outcome = outcome
        self.after_s = after_s
        self._done = threading.Event()
        if outcome is not None:
            threading.Timer(after_s, self._exit).start()

    def _exit(self):
        self.returncode = self.outcome[0]
        self._done.set()

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        self._done.wait(timeout)
        return b"", (self.outcome[1] if self.outcome else b"")

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self._done.set()

    def kill(self):
        self.returncode = -9
        self._done.set()


class FakeGatewayStatus:
    def __init__(self, info):
        self.info = info

    def status(self):
        return self.info


class FakeBridge:
    """A bridge that never listens: the URL alone."""

    instances: list = []

    def __init__(self, pin, log=print):
        self.pin = pin
        self.started = False
        self.stopped = False
        FakeBridge.instances.append(self)

    def start(self):
        self.started = True
        return 5555

    @property
    def url(self):
        return "rtsp://127.0.0.1:5555/phone"

    def stop(self):
        self.stopped = True


def wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_the_command_copies_the_phones_video_into_the_bridge_and_nothing_else():
    publisher = SharePublisher("rtsp://127.0.0.1:8554/cam", "rtsp://127.0.0.1:5555/phone")
    argv = publisher.argv()
    assert argv[0] == "ffmpeg" and argv[-1] == "rtsp://127.0.0.1:5555/phone"
    assert argv[argv.index("-i") + 1] == "rtsp://127.0.0.1:8554/cam"
    assert "-an" in argv and argv[argv.index("-c:v") + 1] == "copy"
    assert "-f" in argv and argv[argv.index("-f") + 1] == "rtsp"
    for forbidden in ("libx264", "h264_nvenc", "-c:a", "pipe:0", "rtmps"):
        assert forbidden not in argv


def test_the_share_needs_the_connector_attached_before_anything_is_probed():
    probes = []

    def probe():
        probes.append(1)
        return "ab" * 32

    no_gateway = Share("rtsp://127.0.0.1:8554/cam", gateway=None, probe=probe)
    with pytest.raises(ShareError) as refusal:
        no_gateway.connect("sess-1")
    assert refusal.value.reason == "connector-offline" and refusal.value.status == 502
    silent = Share("rtsp://127.0.0.1:8554/cam", gateway=FakeGatewayStatus(None), probe=probe)
    with pytest.raises(ShareError) as refusal:
        silent.connect("sess-1")
    assert refusal.value.reason == "connector-offline"
    detached = Share("rtsp://127.0.0.1:8554/cam",
                     gateway=FakeGatewayStatus({"connected": False}), probe=probe)
    with pytest.raises(ShareError) as refusal:
        detached.connect("sess-1")
    assert "not attached" in refusal.value.message
    assert probes == [] and not detached.active
    assert detached.status() == {"active": False, "since": None, "error": None,
                                 "restarts": 0, "alive": False, "refused": False}


def test_a_connected_share_pins_the_leaf_starts_the_bridge_and_the_publisher_and_clears():
    FakeBridge.instances.clear()
    procs = []

    def popen(argv, **kwargs):
        proc = FakeProc(argv)
        procs.append(proc)
        return proc

    log = []
    s = Share("rtsp://127.0.0.1:8554/cam", gateway=FakeGatewayStatus({"connected": True}),
              probe=lambda: "cd" * 32, bridge_factory=FakeBridge, popen=popen,
              clock=lambda: 1_700_000_000.0, log=lambda *a, **k: log.append(a[0]))
    answer = s.connect("sess-1")
    assert answer["status"] == "connected" and answer["active"] is True
    assert answer["since"] == 1_700_000_000.0
    assert s.pin == "cd" * 32 and s.session_id == "sess-1"
    bridge = FakeBridge.instances[-1]
    assert bridge.pin == "cd" * 32 and bridge.started
    wait_for(lambda: len(procs) == 1)
    assert procs[0].argv[-1] == bridge.url
    status = s.status()
    assert status["active"] and status["alive"] and status["restarts"] == 0
    # A second connect replaces: the first process and bridge are stopped.
    s.connect("sess-1")
    wait_for(lambda: len(procs) == 2)
    assert procs[0].returncode == -15 and bridge.stopped
    assert s.clear("the phone asked") is True
    assert procs[1].returncode == -15 and FakeBridge.instances[-1].stopped
    assert not s.active and s.pin is None and s.status()["active"] is False
    assert s.clear("again") is False
    assert any("share: on" in line for line in log) and any("share: off" in line for line in log)
    for line in log:
        assert "://" not in line


def test_a_refusal_by_the_connector_stops_the_publisher_for_good():
    procs = []

    def popen(argv, **kwargs):
        proc = FakeProc(argv, outcome=(1, b"rtsp://127.0.0.1:5555/phone: Server returned 403 Forbidden"))
        procs.append(proc)
        return proc

    publisher = SharePublisher("rtsp://127.0.0.1:8554/cam", "rtsp://127.0.0.1:5555/phone",
                               popen=popen, sleep=lambda s: None, log=lambda *a, **k: None)
    publisher.start()
    wait_for(lambda: publisher.refused)
    time.sleep(0.1)
    snapshot = publisher.snapshot()
    assert len(procs) == 1 and snapshot["refused"] and snapshot["error"] == "connector-refused"
    assert snapshot["alive"] is False
    publisher.stop()


def test_a_plain_exit_is_respawned_with_backoff_and_the_tail_kept_without_addresses():
    procs = []
    clock = [0.0]

    def popen(argv, **kwargs):
        proc = FakeProc(argv, outcome=(1, b"Connection to rtsp://127.0.0.1:5555/phone failed: Connection refused"))
        procs.append(proc)
        return proc

    def sleep(s):
        clock[0] += s

    publisher = SharePublisher("rtsp://127.0.0.1:8554/cam", "rtsp://127.0.0.1:5555/phone",
                               popen=popen, clock=lambda: clock[0], sleep=sleep,
                               log=lambda *a, **k: None)
    publisher.start()
    wait_for(lambda: len(procs) >= 3)
    publisher.stop()
    snapshot = publisher.snapshot()
    assert snapshot["failures"] >= 3 and not snapshot["refused"]
    assert "ffmpeg exited 1" in snapshot["error"] and "://" not in snapshot["error"]


# -- the bridge, against a real TLS server --------------------------------------------


def self_signed() -> tuple[bytes, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "masseuse-camlink camera")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    return (cert.public_bytes(serialization.Encoding.PEM),
            key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()))


class EchoTlsServer:
    """The connector's end, as the gateway's own listener presents it: TLS
    with the connector's certificate, echoing what it is sent."""

    def __init__(self, tmp_path):
        cert_pem, key_pem = self_signed()
        (tmp_path / "cert.pem").write_bytes(cert_pem)
        (tmp_path / "key.pem").write_bytes(key_pem)
        der = x509.load_pem_x509_certificate(cert_pem).public_bytes(serialization.Encoding.DER)
        self.fingerprint = hashlib.sha256(der).hexdigest()
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(str(tmp_path / "cert.pem"), str(tmp_path / "key.pem"))
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.sock.settimeout(0.5)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    def _echo(self, conn):
        try:
            with self.context.wrap_socket(conn, server_side=True) as tls:
                self.accepted += 1
                while True:
                    data = tls.recv(4096)
                    if not data:
                        return
                    tls.sendall(data)
        except (OSError, ssl.SSLError):
            return

    def stop(self):
        self._stop.set()
        self.sock.close()


def test_the_bridge_relays_only_to_the_pinned_certificate(tmp_path):
    server = EchoTlsServer(tmp_path)
    try:
        # The probe sees the connector's leaf, as the enclave would through
        # the gateway's own listener.
        assert share.probe_connector("127.0.0.1", server.port, timeout_s=3.0) == server.fingerprint
        bridge = TlsBridge(server.fingerprint.upper(), target=("127.0.0.1", server.port),
                           log=lambda *a, **k: None)
        port = bridge.start()
        assert bridge.url == f"rtsp://127.0.0.1:{port}/phone"
        with socket.create_connection(("127.0.0.1", port), timeout=5) as plain:
            plain.sendall(b"ANNOUNCE rtsp://127.0.0.1/phone RTSP/1.0\r\n\r\n")
            plain.settimeout(5)
            assert plain.recv(4096).startswith(b"ANNOUNCE")
        wait_for(lambda: bridge.accepted == 1)
        bridge.stop()

        # Another pin: the connection is closed without a byte relayed.
        wrong = TlsBridge("00" * 32, target=("127.0.0.1", server.port), log=lambda *a, **k: None)
        port = wrong.start()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as plain:
            plain.sendall(b"hello")
            plain.settimeout(5)
            try:
                assert plain.recv(4096) == b""
            except ConnectionResetError:
                pass  # closed on us before we read: the same refusal
        wait_for(lambda: wrong.refused == 1)
        assert wrong.accepted == 0
        wrong.stop()

        # Nothing at the target: refused too, and the probe says offline.
        server.stop()
        with pytest.raises(ShareError) as refusal:
            share.probe_connector("127.0.0.1", server.port, timeout_s=1.0)
        assert refusal.value.reason == "connector-offline"
    finally:
        server.stop()
