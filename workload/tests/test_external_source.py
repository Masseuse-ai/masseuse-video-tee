"""The external camera: what the enclave accepts, how it pins the camera's
certificate, and how it drives the relay.

A link is rtsps:// to a public address or it is refused with a reason the
page can show; the TLS probe returns the leaf's SHA-256 in the relay's
`sourceFingerprint` form and doubles as the reachability check; connect
adds the `ext` path with that fingerprint and TCP transport, waits for a
video track and reports it; a camera that never streams, or streams no
video, leaves no path behind; clear removes the path; and /ingest/status
follows whichever camera is the session's.

A camera at home (a private literal, a `.local` or bare name) goes through
the connector instead: refused as `tunnel-offline` unless the gateway has
one attached, otherwise the gateway is given the host and port, the probe
goes to the relay listener with no SNI, and the relay is pointed at
127.0.0.1:7441 with the link's credentials, path and query kept; a public
host is dialled exactly as before; the status says which way, and whether
the connector is there.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import external_source as es  # noqa: E402
from camlink_gateway import CamlinkGateway  # noqa: E402
from relay_proxy import RelayProxy  # noqa: E402
from test_camlink_gateway import FakeGateway  # noqa: E402
from test_relay_proxy import StubRelay  # noqa: E402

PUBLIC = "203.0.113.20"  # documentation range: refused by is_global, so
GLOBAL = "8.8.8.8"       # a plainly global address stands in for the camera
OFFLINE = {"connected": False, "sinceMs": None}


def resolver_for(*addresses: str):
    """A getaddrinfo that answers with fixed addresses, whatever the name."""
    def resolve(host, port, type=0):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, type, 6, "",
                 (a, port) if ":" not in a else (a, port, 0, 0)) for a in addresses]
    return resolve


def failing_resolver(host, port, type=0):
    raise socket.gaierror(8, "nodename nor servname provided")


# -- validate ----------------------------------------------------------------------------


def test_only_rtsps_links_with_a_host_pass_the_shape_check():
    ok = resolver_for(GLOBAL)
    assert es.validate("rtsps://cam.example/live", resolver=ok) == ("cam.example", 322, GLOBAL)
    # Pasted links arrive with whitespace around them; that is not a refusal.
    assert es.validate("  RTSPS://user:p%40ss@cam.example:7441/abc?enableSrtp \n",
                       resolver=ok) == ("cam.example", 7441, GLOBAL)
    assert es.validate("rtsps://[2606:4700:4700::1111]/x",
                       resolver=resolver_for("2606:4700:4700::1111")) == (
        "2606:4700:4700::1111", 322, "2606:4700:4700::1111")
    for url, reason in (("rtsp://cam.example/live", "not-rtsps"),
                        ("https://cam.example/live", "not-rtsps"),
                        ("rtsps:///live", "bad-url"),
                        ("rtsps://", "bad-url"),
                        ("", "bad-url"),
                        ("   ", "bad-url"),
                        ("rtsps://cam.example:99999/x", "bad-url"),
                        ("rtsps://cam.example/li ve", "bad-url"),
                        ("rtsps://cam.example/x\ny", "bad-url"),
                        ("rtsps://cam.example/" + "x" * 2100, "bad-url"),
                        (None, "bad-url"), (42, "bad-url")):
        with pytest.raises(es.SourceError) as excinfo:
            es.validate(url, resolver=ok)
        assert excinfo.value.reason == reason, url
        assert excinfo.value.status == 400


def test_private_local_and_own_addresses_are_refused_public_ones_pass():
    for address in ("10.0.0.5", "172.16.3.4", "192.168.1.20", "127.0.0.1", "0.0.0.0",
                    "169.254.169.254", "100.64.0.9", "224.0.0.1", "255.255.255.255",
                    "240.0.0.1", "::1", "fe80::1", "fd12::1", "::", "ff02::1",
                    "::ffff:10.0.0.5", "::ffff:192.168.0.1", "203.0.113.20"):
        with pytest.raises(es.SourceError) as excinfo:
            es.validate("rtsps://cam.example/x", resolver=resolver_for(address))
        assert excinfo.value.reason == "private-address", address
    # A name that resolves to both a public and a private address is refused whole.
    with pytest.raises(es.SourceError) as excinfo:
        es.validate("rtsps://cam.example/x", resolver=resolver_for(GLOBAL, "10.0.0.5"))
    assert excinfo.value.reason == "private-address"
    # The slot's own public IP is not a camera.
    with pytest.raises(es.SourceError) as excinfo:
        es.validate("rtsps://cam.example/x", own_ip="34.10.20.30",
                    resolver=resolver_for("34.10.20.30"))
    assert excinfo.value.reason == "private-address"
    assert es.validate("rtsps://cam.example/x", own_ip="34.10.20.30",
                       resolver=resolver_for("34.10.20.31"))[2] == "34.10.20.31"
    # A literal public address needs no name.
    assert es.validate(f"rtsps://{GLOBAL}:322/x", resolver=socket.getaddrinfo) == (
        GLOBAL, 322, GLOBAL)
    with pytest.raises(es.SourceError) as excinfo:
        es.validate("rtsps://nx.invalid/x", resolver=failing_resolver)
    assert excinfo.value.reason == "unreachable" and excinfo.value.status == 502
    assert "nx.invalid" in excinfo.value.message


def test_home_addresses_and_names_are_the_connectors_public_ones_are_not():
    for host in ("192.168.1.108", "10.0.0.5", "172.16.3.4", "172.31.255.254",
                 "127.0.0.1", "169.254.7.7", "::1", "fe80::1", "fe80::1%en0",
                 "fd12::1", "fc00::9", "::ffff:192.168.0.1", "::ffff:10.1.2.3",
                 "cam.local", "Cam.LOCAL", "cam.local.", "camera", "localhost",
                 "unifi-g4"):
        assert es.is_private_host(host), host
    for host in ("cam.example", "cam.example.com", "8.8.8.8", "203.0.113.20",
                 "100.64.0.9", "172.32.0.1", "192.169.0.1", "224.0.0.1", "0.0.0.0",
                 "2606:4700:4700::1111", "::ffff:8.8.8.8", "cam.local.example",
                 "", None):
        assert not es.is_private_host(host), host
    # The shape checks come first either way: a bad link is a bad link.
    for url, reason in (("rtsp://192.168.1.108/live", "not-rtsps"),
                        ("rtsps://192.168.1.108/li ve", "bad-url"),
                        ("rtsps://cam.local:99999/x", "bad-url")):
        with pytest.raises(es.SourceError) as excinfo:
            es.parse_link(url)
        assert excinfo.value.reason == reason, url
    assert es.parse_link(" rtsps://u:p@192.168.1.108/x?y ") == (
        "rtsps://u:p@192.168.1.108/x?y", "192.168.1.108", 322)
    assert es.parse_link("rtsps://[fe80::1]:7441/x")[1:] == ("fe80::1", 7441)


def test_the_tunnel_source_url_keeps_everything_but_the_camera_address():
    assert es.tunnel_source_url(
        "rtsps://192.168.1.108:7441/SGSV8hfdHpQXGyIz?enableSrtp") == (
        "rtsps://127.0.0.1:7441/SGSV8hfdHpQXGyIz?enableSrtp")
    assert es.tunnel_source_url("rtsps://user:pass@192.168.1.156:322/stream0") == (
        "rtsps://user:pass@127.0.0.1:7441/stream0")
    # Credentials go through as pasted, percent-encoding included; the
    # default port, a .local name, an IPv6 literal and a stray fragment all
    # end up at the relay listener.
    assert es.tunnel_source_url("  RTSPS://u:p%40ss@cam.local/back?a=1&b=2 ") == (
        "rtsps://u:p%40ss@127.0.0.1:7441/back?a=1&b=2")
    assert es.tunnel_source_url("rtsps://[fe80::1%25en0]:8554/x#f") == (
        "rtsps://127.0.0.1:7441/x#f")
    assert es.tunnel_source_url("rtsps://10.0.0.5") == "rtsps://127.0.0.1:7441"


def test_source_error_body_is_what_the_page_reads():
    error = es.SourceError("timeout", "the camera did not start", 502)
    assert error.body() == {"status": "failed", "reason": "timeout",
                            "error": "the camera did not start"}
    assert str(error) == "the camera did not start"


def test_video_codecs_are_recognised_by_the_relays_names():
    for codec in ("H264", "H265", "AV1", "VP9", "VP8", "M-JPEG", "MPEG-4 Video",
                  "MPEG-1/2 Video", "h264"):
        assert es.is_video(codec), codec
    for codec in ("Opus", "MPEG-4 Audio", "G711", "LPCM", "AC-3", "", None):
        assert not es.is_video(codec), codec


# -- the TLS probe -----------------------------------------------------------------------


def self_signed(tmp_path: Path, host: str = "cam.example") -> tuple[Path, bytes]:
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, host)])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(7)
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .sign(key, hashes.SHA256()))
    pem = tmp_path / "cam.pem"
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption())
                    + cert.public_bytes(serialization.Encoding.PEM))
    return pem, cert.public_bytes(serialization.Encoding.DER)


class OneShotServer:
    """Accepts one connection on loopback: a TLS handshake with `pem`, or
    a few bytes of plain text when there is none."""

    def __init__(self, pem: Path | None = None):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.pem = pem
        self.seen_sni: list = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        try:
            conn.settimeout(3)
            if self.pem is None:
                conn.sendall(b"RTSP/1.0 200 OK\r\n\r\n")
                return
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(self.pem))
            context.sni_callback = lambda sock, name, ctx: self.seen_sni.append(name)
            with context.wrap_socket(conn, server_side=True) as tls:
                tls.recv(1)
        except (OSError, ssl.SSLError):
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self.sock.close()


def test_the_probe_returns_the_leafs_sha256_in_the_relays_form(tmp_path):
    pytest.importorskip("cryptography")
    pem, der = self_signed(tmp_path)
    server = OneShotServer(pem)
    try:
        got = es.tls_fingerprint("cam.example", server.port, "127.0.0.1", timeout_s=3)
    finally:
        server.close()
    assert got == hashlib.sha256(der).hexdigest().upper()
    assert len(got) == 64 and got == got.upper() and ":" not in got
    # The URL's host went as the SNI although the connection went to the
    # resolved address; a literal address sends none.
    assert server.seen_sni == ["cam.example"]
    server = OneShotServer(pem)
    try:
        es.tls_fingerprint("127.0.0.1", server.port, "127.0.0.1", timeout_s=3)
    finally:
        server.close()
    assert server.seen_sni == [None]
    # Told to send none, it sends none whatever the host: the probe through
    # the gateway's relay listener, where the camera must not be named.
    server = OneShotServer(pem)
    try:
        got = es.tls_fingerprint("cam.example", server.port, "127.0.0.1", timeout_s=3,
                                 sni=False)
    finally:
        server.close()
    assert got == hashlib.sha256(der).hexdigest().upper()
    assert server.seen_sni == [None]


def test_the_probe_names_a_port_that_is_not_tls_and_one_that_is_closed():
    plain = OneShotServer(None)
    try:
        with pytest.raises(es.SourceError) as excinfo:
            es.tls_fingerprint("cam.example", plain.port, "127.0.0.1", timeout_s=3)
    finally:
        plain.close()
    assert excinfo.value.reason == "not-tls" and excinfo.value.status == 502
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    with pytest.raises(es.SourceError) as excinfo:
        es.tls_fingerprint("cam.example", port, "127.0.0.1", timeout_s=3)
    assert excinfo.value.reason == "unreachable" and excinfo.value.status == 502


# -- connect / clear against the relay ------------------------------------------------------


FINGERPRINT = "AB" * 32


def make_source(relay: StubRelay, **overrides) -> tuple[RelayProxy, es.ExternalSource]:
    proxy = RelayProxy(relay.base, relay.base)
    probes: list[tuple] = []

    def probe(host, port, ip, timeout_s, **options):
        probes.append((host, port, ip, timeout_s) + ((options,) if options else ()))
        return FINGERPRINT

    clock = {"now": 0.0}
    options = dict(resolver=resolver_for(GLOBAL), probe=probe, sleep=lambda s: None,
                   monotonic=lambda: clock.__setitem__("now", clock["now"] + 0.5) or clock["now"],
                   clock=lambda: 1_700_000_000, log=lambda *a, **k: None)
    options.update(overrides)
    source = es.ExternalSource(proxy, own_ip="34.1.2.3", **options)
    source.probes = probes  # type: ignore[attr-defined]
    return proxy, source


def test_connect_adds_the_pinned_path_waits_for_video_and_reports_it():
    relay = StubRelay()
    try:
        proxy, source = make_source(relay)
        relay.ext_ready_after = 3
        assert source.active is False
        assert source.status() == {"kind": "phone", "path": "cam", "ready": None,
                                   "tracks": [], "since": None, "mode": None,
                                   "tunnel": OFFLINE}
        answer = source.connect("rtsps://u:p@cam.example:7441/live?enableSrtp", "sess-1")
        assert answer == {"status": "connected", "kind": "external", "path": "ext",
                          "tracks": ["H264"], "since": 1_700_000_000, "mode": "direct"}
        assert source.probes == [("cam.example", 7441, GLOBAL, 5.0)]
        # The relay got the link, the pin, TCP transport, and an eager pull.
        assert relay.ext_config == {"source": "rtsps://u:p@cam.example:7441/live?enableSrtp",
                                    "sourceFingerprint": FINGERPRINT,
                                    "rtspTransport": "tcp", "sourceOnDemand": False}
        assert relay.ext_polls == 3
        assert source.active is True and source.session_id == "sess-1"
        assert source.mode == "direct"
        assert source.stream_url == "rtsp://127.0.0.1:8554/ext"
        assert source.status() == {"kind": "external", "path": "ext", "ready": True,
                                   "tracks": ["H264"], "since": 1_700_000_000,
                                   "mode": "direct", "tunnel": OFFLINE}
        # /ingest/status follows the external camera; the phone stays in view.
        code, body = proxy.ingest_status(source)
        assert code == 200
        assert body["ready"] is True and body["tracks"] == ["H264"]
        assert body["path"] == "ext" and body["streamUrl"] == "rtsp://127.0.0.1:8554/ext"
        assert body["source"] == {"kind": "external", "path": "ext", "ready": True,
                                  "tracks": ["H264"], "type": "rtspSource",
                                  "since": 1_700_000_000, "mode": "direct",
                                  "tunnel": OFFLINE}
        assert body["phone"] == {"ready": False, "tracks": []}
        # A second link replaces the first: the old path goes before the new
        # one is added, and nothing was ever requested with a fingerprint
        # other than the probe's.
        relay.ext_ready_after = 1
        source.connect("rtsps://other.example/x", "sess-1")
        assert relay.ext_deletes == 1
        assert relay.ext_config["source"] == "rtsps://other.example/x"
        # Clear: the path is deleted and the camera is the phone's again.
        assert source.clear("test") is True
        assert relay.ext_deletes == 2 and relay.ext_config is None
        assert source.active is False and source.clear() is False
        assert source.mode is None
        code, body = proxy.ingest_status(source)
        assert body["source"]["kind"] == "phone" and body["path"] == "cam"
        assert body["source"]["mode"] is None and body["source"]["tunnel"] == OFFLINE
        assert body["streamUrl"] == "rtsp://127.0.0.1:8554/cam"
    finally:
        relay.stop()


# -- a camera at home, through the connector -------------------------------------------


UNIFI = "rtsps://192.168.1.108:7441/SGSV8hfdHpQXGyIz?enableSrtp"
WYZE = "rtsps://user:pass@192.168.1.156:322/stream0"


def make_tunnel_source(relay: StubRelay, gateway: FakeGateway | None, **overrides):
    client = (CamlinkGateway(gateway.base, timeout_s=2.0, log=lambda *a, **k: None)
              if gateway is not None else None)
    return make_source(relay, gateway=client, **overrides)


def test_a_home_camera_is_refused_until_the_connector_is_there():
    relay, gateway = StubRelay(), FakeGateway(connected=False)
    try:
        proxy, source = make_tunnel_source(relay, gateway)
        with pytest.raises(es.SourceError) as excinfo:
            source.connect(UNIFI, "sess-1")
        assert excinfo.value.reason == "tunnel-offline"
        assert excinfo.value.status == 502
        assert "connector" in excinfo.value.message
        assert excinfo.value.body()["reason"] == "tunnel-offline"
        # Nothing else was touched: no target, no probe, no relay path, and
        # the camera's address went nowhere.
        assert gateway.paths() == ["/status"]
        assert source.probes == [] and relay.ext_config is None
        assert relay.ext_deletes == 0 and source.active is False
        # The status the phone and the trainer read says why.
        assert source.status()["tunnel"] == OFFLINE
        assert proxy.ingest_status(source)[1]["source"]["tunnel"] == OFFLINE
        # A connector that leaves between the status and the target reads
        # the same way (the gateway's 409).
        gateway.connected = True
        racing = FakeGateway(connected=False)
        try:
            source.gateway = CamlinkGateway(racing.base, timeout_s=2.0,
                                            log=lambda *a, **k: None)
            source.gateway.status = lambda: {"connected": True}  # type: ignore[method-assign]
            with pytest.raises(es.SourceError) as excinfo:
                source.connect(UNIFI, "sess-1")
            assert excinfo.value.reason == "tunnel-offline"
            assert "went away" in excinfo.value.message
            assert racing.paths("POST") == ["/target"] and source.probes == []
        finally:
            racing.stop()
        # A gateway that is not answering at all: offline too, and the
        # status says offline rather than failing.
        source.gateway = CamlinkGateway("http://127.0.0.1:1", timeout_s=1.0,
                                        log=lambda *a, **k: None)
        with pytest.raises(es.SourceError) as excinfo:
            source.connect(UNIFI, "sess-1")
        assert excinfo.value.reason == "tunnel-offline"
        assert source.status()["tunnel"] == OFFLINE
        # A slot with no gateway at all cannot reach a camera at home.
        source.gateway = None
        with pytest.raises(es.SourceError) as excinfo:
            source.connect(UNIFI, "sess-1")
        assert excinfo.value.reason == "tunnel-offline"
    finally:
        gateway.stop()
        relay.stop()


def test_a_home_camera_goes_through_the_connector_with_its_own_certificate_pinned():
    relay, gateway = StubRelay(), FakeGateway(connected=True, since_ms=1_699_999_000_000)
    try:
        proxy, source = make_tunnel_source(relay, gateway)
        relay.ext_ready_after = 2
        answer = source.connect(UNIFI, "sess-1")
        assert answer == {"status": "connected", "kind": "external", "path": "ext",
                          "tracks": ["H264"], "since": 1_700_000_000, "mode": "tunnel"}
        # The gateway was asked whether a connector is there, then given the
        # camera's address; the probe went to the relay listener with no SNI.
        assert gateway.paths() == ["/status", "/target"]
        assert gateway.target == {"host": "192.168.1.108", "port": 7441}
        assert source.probes == [("127.0.0.1", 7441, "127.0.0.1", 5.0, {"sni": False})]
        # The relay pulls through the listener, pinned to the leaf the probe
        # saw, with the link's path and query kept and the host gone.
        assert relay.ext_config == {
            "source": "rtsps://127.0.0.1:7441/SGSV8hfdHpQXGyIz?enableSrtp",
            "sourceFingerprint": FINGERPRINT, "rtspTransport": "tcp",
            "sourceOnDemand": False}
        assert source.active is True and source.mode == "tunnel"
        tunnel = {"connected": True, "sinceMs": 1_699_999_000_000}
        assert source.status() == {"kind": "external", "path": "ext", "ready": True,
                                   "tracks": ["H264"], "since": 1_700_000_000,
                                   "mode": "tunnel", "tunnel": tunnel}
        code, body = proxy.ingest_status(source)
        assert code == 200 and body["streamUrl"] == "rtsp://127.0.0.1:8554/ext"
        assert body["source"] == {"kind": "external", "path": "ext", "ready": True,
                                  "tracks": ["H264"], "type": "rtspSource",
                                  "since": 1_700_000_000, "mode": "tunnel",
                                  "tunnel": tunnel}
        # Credentials ride along; a link with no port means the rtsps default.
        relay.ext_ready_after = 1
        source.connect(WYZE, "sess-1")
        assert gateway.target == {"host": "192.168.1.156", "port": 322}
        assert relay.ext_config["source"] == "rtsps://user:pass@127.0.0.1:7441/stream0"
        source.connect("rtsps://viewer:s3cret@cam.local/live", "sess-1")
        assert gateway.target == {"host": "cam.local", "port": 322}
        assert relay.ext_config["source"] == "rtsps://viewer:s3cret@127.0.0.1:7441/live"
        assert all(p[:3] == ("127.0.0.1", 7441, "127.0.0.1") for p in source.probes)
        # A replaced link never dropped the connector; the camera going does.
        assert gateway.paths("POST") == ["/target"] * 3
        assert source.clear("the phone asked") is True
        assert gateway.paths("POST") == ["/target"] * 3 + ["/clear"]
        assert gateway.target is None and gateway.connected is False
        assert relay.ext_config is None
        assert source.status()["tunnel"] == OFFLINE
        # A public camera is dialled as it always was: resolved, probed with
        # its own name as the SNI, pulled at its own address, and the gateway
        # is not involved.
        gateway.requests.clear()
        source.connect("rtsps://u:p@cam.example:7441/live?enableSrtp", "sess-1")
        assert source.probes[-1] == ("cam.example", 7441, GLOBAL, 5.0)
        assert relay.ext_config["source"] == "rtsps://u:p@cam.example:7441/live?enableSrtp"
        assert source.mode == "direct" and gateway.paths("POST") == []
        # ...and clearing it leaves the connector alone.
        source.clear("test")
        assert gateway.paths("POST") == []
    finally:
        gateway.stop()
        relay.stop()


def test_a_probe_failure_through_the_connector_is_worded_for_it():
    relay, gateway = StubRelay(), FakeGateway(connected=True)
    try:
        def unreachable(host, port, ip, timeout_s, **options):
            raise es.SourceError("unreachable", f"{host}:{port} refused the connection", 502)

        _, source = make_tunnel_source(relay, gateway, probe=unreachable)
        with pytest.raises(es.SourceError) as excinfo:
            source.connect(UNIFI, "sess-1")
        assert excinfo.value.reason == "unreachable" and excinfo.value.status == 502
        assert "connector" in excinfo.value.message
        assert "127.0.0.1" not in excinfo.value.message
        assert "192.168" not in excinfo.value.message
        assert relay.ext_config is None and source.active is False

        def plain(host, port, ip, timeout_s, **options):
            raise es.SourceError("not-tls", f"{host}:{port} does not speak TLS", 502)

        _, source = make_tunnel_source(relay, gateway, probe=plain)
        with pytest.raises(es.SourceError) as excinfo:
            source.connect(WYZE, "sess-1")
        assert excinfo.value.reason == "not-tls" and "rtsps" in excinfo.value.message
    finally:
        gateway.stop()
        relay.stop()


def test_a_camera_that_never_streams_or_sends_no_video_leaves_no_path():
    relay = StubRelay()
    try:
        _, source = make_source(relay, connect_timeout_s=2.0)
        relay.ext_ready_after = 10 ** 6
        with pytest.raises(es.SourceError) as excinfo:
            source.connect("rtsps://cam.example/x", "sess-1")
        assert excinfo.value.reason == "timeout" and excinfo.value.status == 502
        assert relay.ext_config is None and relay.ext_deletes == 1
        assert source.active is False
        relay.ext_ready_after = 1
        relay.ext_tracks = ["Opus"]
        with pytest.raises(es.SourceError) as excinfo:
            source.connect("rtsps://cam.example/x", "sess-1")
        assert excinfo.value.reason == "no-video"
        assert "Opus" in excinfo.value.message
        assert relay.ext_config is None and relay.ext_deletes == 2
        # Validation and the probe run before the relay is touched.
        with pytest.raises(es.SourceError) as excinfo:
            source.connect("rtsp://cam.example/x", "sess-1")
        assert excinfo.value.reason == "not-rtsps"
        assert relay.ext_deletes == 2
    finally:
        relay.stop()


def test_a_relay_that_is_down_is_a_503_and_a_refused_add_says_so():
    proxy = RelayProxy("http://127.0.0.1:1", "http://127.0.0.1:1", timeout_s=0.5)
    source = es.ExternalSource(proxy, resolver=resolver_for(GLOBAL),
                               probe=lambda *a: FINGERPRINT, sleep=lambda s: None,
                               log=lambda *a, **k: None)
    with pytest.raises(es.SourceError) as excinfo:
        source.connect("rtsps://cam.example/x")
    assert excinfo.value.reason == "relay" and excinfo.value.status == 503
    assert source.clear() is False  # nothing to delete, and no exception
    relay = StubRelay()
    try:
        _, source = make_source(relay)
        relay.ext_config = {"source": "stale"}  # a path this process did not add
        relay.ext_ready_after = 1
        # The stale path is deleted first, so the add succeeds.
        assert source.connect("rtsps://cam.example/x")["status"] == "connected"
        assert relay.ext_deletes == 1
    finally:
        relay.stop()
