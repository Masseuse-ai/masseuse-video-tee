"""The home connector's gateway, as the producer drives it.

The client posts the trainer's expectation, the camera's target and a
clear to the gateway's loopback control API and reads its status back; a
gateway that is down is a GatewayError the callers turn into 502 or
`tunnel-offline`, never a hang; the status view the trainer polls is
cached for a couple of seconds and reads offline when the gateway cannot
be reached; and the trainer's /tunnel/expect body is checked for shape
before anything is forwarded.

`FakeGateway` here is masseuse-camlink-gateway's control API
(PROTOCOL.md section 5), enough for these tests and for test_external_source
and test_tee_mode, which build on it.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import camlink_gateway as cg  # noqa: E402

CONNECTOR_KEY = "k" * 43  # base64url of 32 bytes, unpadded
TICKET_HASH = "ab" * 32


class FakeGateway(ThreadingHTTPServer):
    """The gateway's control API on an ephemeral loopback port. `connected`
    is whether a connector is attached (what /status says and what /target
    needs); the expectation and the target it was given are remembered,
    every request is recorded, and /clear forgets all three."""

    def __init__(self, connected: bool = False, since_ms: int | None = 1_700_000_000_000):
        self.requests: list[dict] = []
        self.connected = connected
        self.since_ms = since_ms
        self.expectation: dict | None = None
        self.target: dict | None = None
        super().__init__(("127.0.0.1", 0), FakeGatewayHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def paths(self, method: str | None = None) -> list[str]:
        return [r["path"] for r in self.requests if method is None or r["method"] == method]

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class FakeGatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def _record(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        self.server.requests.append({"method": self.command, "path": self.path,
                                     "body": body})
        return body

    def _reply(self, code: int, body: bytes = b"") -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json" if body else "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_POST(self):
        body = self._record()
        if self.path == "/expect":
            self.server.expectation = body
            return self._reply(200, b"{}")
        if self.path == "/target":
            if not self.server.connected:
                return self._reply(409, b'{"error":"no connector attached"}')
            self.server.target = body
            return self._reply(200, b"{}")
        if self.path == "/clear":
            self.server.expectation = None
            self.server.target = None
            self.server.connected = False
            return self._reply(200, b"{}")
        self._reply(404, b'{"error":"not found"}')

    def do_GET(self):
        self._record()
        if self.path == "/healthz":
            return self._reply(200, b"ok")
        if self.path != "/status":
            return self._reply(404, b'{"error":"not found"}')
        expectation = self.server.expectation or {}
        self._reply(200, json.dumps({
            "expecting": bool(expectation),
            "connected": self.server.connected,
            "sinceMs": self.server.since_ms if self.server.connected else None,
            "connectorKeyPrefix": (expectation.get("connectorKey") or "")[:8] or None,
            "target": self.server.target is not None,
        }).encode())


DOWN = "http://127.0.0.1:1"


def test_expect_target_and_clear_reach_the_gateway_as_posted():
    gateway = FakeGateway()
    try:
        client = cg.CamlinkGateway(gateway.base, timeout_s=2.0, log=lambda *a, **k: None)
        expectation = {"connectorKey": CONNECTOR_KEY, "ticketHash": TICKET_HASH,
                       "expiresAt": 1_800_000_000}
        client.expect(expectation)
        assert gateway.expectation == expectation
        assert gateway.requests[-1] == {"method": "POST", "path": "/expect",
                                        "body": expectation}
        # No connector yet: the target is refused with the gateway's 409.
        with pytest.raises(cg.GatewayError) as excinfo:
            client.target("192.168.1.108", 7441)
        assert excinfo.value.status == 409 and not excinfo.value.unreachable
        assert gateway.target is None
        gateway.connected = True
        client.target("192.168.1.108", 7441)
        assert gateway.target == {"host": "192.168.1.108", "port": 7441}
        status = client.status()
        assert status["connected"] is True and status["target"] is True
        assert status["connectorKeyPrefix"] == "kkkkkkkk"
        client.clear()
        assert gateway.requests[-1] == {"method": "POST", "path": "/clear", "body": None}
        assert gateway.expectation is None and gateway.target is None
        assert client.status()["connected"] is False
    finally:
        gateway.stop()


def test_a_gateway_that_is_down_is_an_error_not_a_hang_and_reads_offline():
    client = cg.CamlinkGateway(DOWN, timeout_s=1.0, log=lambda *a, **k: None)
    for call in (lambda: client.expect({"connectorKey": CONNECTOR_KEY}),
                 lambda: client.target("192.168.1.108", 7441),
                 client.clear):
        with pytest.raises(cg.GatewayError) as excinfo:
            call()
        assert excinfo.value.unreachable and excinfo.value.status is None
    assert client.status() is None
    assert client.tunnel() == {"connected": False, "sinceMs": None}
    # The quiet form the hooks use never raises; it says what happened.
    said: list[str] = []
    quiet = cg.CamlinkGateway(DOWN, timeout_s=1.0, log=lambda m, **k: said.append(m))
    assert quiet.clear_quietly("teardown") is False
    assert "did not reach" in said[-1]
    # A gateway that answers with an error is an error carrying its code.
    def refusing(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)
    broken = cg.CamlinkGateway("http://gw", opener=refusing)
    with pytest.raises(cg.GatewayError) as excinfo:
        broken.expect({})
    assert excinfo.value.status == 500


def test_the_tunnel_view_is_cached_briefly_and_dropped_by_a_clear():
    gateway = FakeGateway(connected=True, since_ms=1_700_000_000_000)
    try:
        clock = {"now": 100.0}
        client = cg.CamlinkGateway(gateway.base, clock=lambda: clock["now"],
                                   log=lambda *a, **k: None)
        assert client.tunnel() == {"connected": True, "sinceMs": 1_700_000_000_000}
        assert gateway.paths("GET") == ["/status"]
        # Within two seconds the trainer's polls are answered from memory.
        gateway.connected = False
        clock["now"] += 1.9
        assert client.tunnel()["connected"] is True
        assert gateway.paths("GET") == ["/status"]
        clock["now"] += 0.2
        assert client.tunnel() == {"connected": False, "sinceMs": None}
        assert gateway.paths("GET") == ["/status", "/status"]
        # A clear forgets what was cached: the next view asks again.
        gateway.connected = True
        client.clear_quietly("test")
        assert client.tunnel()["connected"] is False  # the fake dropped the connector
        assert gateway.paths("GET") == ["/status"] * 3
        # Only what may have been set is cleared again: after a clear with
        # nothing posted since, the quiet form does not bother the gateway.
        assert client.clear_quietly("teardown") is False
        assert gateway.paths("POST") == ["/clear"]
        gateway.connected = True  # a connector attaches again
        client.target("192.168.1.108", 7441)
        assert client.clear_quietly("teardown") is True
        assert gateway.paths("POST") == ["/clear", "/target", "/clear"]
    finally:
        gateway.stop()
    assert cg.tunnel_view(None) == {"connected": False, "sinceMs": None}
    assert cg.tunnel_view({"connected": True, "sinceMs": 12.0}) == {
        "connected": True, "sinceMs": 12}
    assert cg.tunnel_view({"connected": True, "sinceMs": "soon"}) == {
        "connected": True, "sinceMs": None}
    assert cg.tunnel_view({"connected": False, "sinceMs": 5}) == {
        "connected": False, "sinceMs": None}


def test_the_expectation_is_checked_for_shape_before_it_is_forwarded():
    now = 1_700_000_000.0
    good = {"connectorKey": CONNECTOR_KEY, "ticketHash": TICKET_HASH.upper(),
            "expiresAt": 1_700_000_600, "sessionId": "sess-1"}
    session_id, forwarded = cg.parse_expectation(good, now)
    assert session_id == "sess-1"
    # Forwarded as the gateway wants it: the hash lowercased, no session id.
    assert forwarded == {"connectorKey": CONNECTOR_KEY, "ticketHash": TICKET_HASH,
                         "expiresAt": 1_700_000_600}
    assert cg.parse_expectation({**good, "expiresAt": 1_700_000_600.7}, now)[1][
        "expiresAt"] == 1_700_000_600
    for field, value in (("connectorKey", "k" * 42), ("connectorKey", "k" * 44),
                         ("connectorKey", "k" * 42 + "="), ("connectorKey", "k" * 42 + "+"),
                         ("connectorKey", None), ("connectorKey", 7),
                         ("ticketHash", "ab" * 31), ("ticketHash", "zz" * 32),
                         ("ticketHash", ""), ("expiresAt", now - 1),
                         ("expiresAt", now), ("expiresAt", now + 7 * 3600),
                         ("expiresAt", "1700000600"), ("expiresAt", True),
                         ("expiresAt", None), ("sessionId", ""),
                         ("sessionId", "bad/slash"), ("sessionId", 12),
                         ("sessionId", None)):
        with pytest.raises(ValueError) as excinfo:
            cg.parse_expectation({**good, field: value}, now)
        assert field in str(excinfo.value), (field, value)


def test_the_control_url_comes_from_the_environment_or_the_loopback_default():
    assert cg.CamlinkGateway.from_env({}).control == "http://127.0.0.1:8091"
    assert cg.CamlinkGateway.from_env(
        {"CAMLINK_GATEWAY_CONTROL": "http://127.0.0.1:9091/"}).control == "http://127.0.0.1:9091"
    assert cg.CamlinkGateway("").control == "http://127.0.0.1:8091"
    assert (cg.RELAY_HOST, cg.RELAY_PORT) == ("127.0.0.1", 7441)
