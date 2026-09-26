"""WHEP signalling reaches the relay through the producer's port, and only
that.

The relay's WebRTC endpoint binds loopback inside the instance; a subscriber
sees `/overlay/whep` on the producer. Locked down here against a stub relay:
POST/PATCH/DELETE/OPTIONS are forwarded with the WHEP headers and body, the
relay's status, SDP answer, Location, Link and ETag come back verbatim (the
Location keeps the same prefix, so the client follows it unchanged), other
methods and malformed secrets are refused, an oversize body is refused, a
relay that is down is a 502 rather than a hang; and /overlay/status says
503 with no session, 200 with one. Through the real handler: nothing under
/overlay exists unless the overlay is configured, and a subscriber before
the session gets a 503, not the relay's 404.

The ingest leg (/ingest/whip, /ingest/status -> the relay's `cam` path) is
the mirror image: forwarded the same way, its Location rewritten into
/ingest/whip/<secret> whatever shape the relay used (MediaMTX answers with
the bare secret), and served with no session at all, because the phone's
camera has to be at the relay before the session that reads it can start.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import producer  # noqa: E402
from relay_proxy import RelayProxy, own_location, route  # noqa: E402
from telemetry import Telemetry  # noqa: E402

WHIP_SECRET = "7d1e4c2a-aaaa-bbbb-cccc-ddddeeeeffff"


class StubRelay(ThreadingHTTPServer):
    """MediaMTX's WHEP/WHIP surface and paths API, just enough to be
    forwarded to. `cam_ready` is whether a phone is publishing to cam.

    The config API for the external camera (external_source.py): an added
    `ext` path is remembered in `ext_config`, reads as ready once it has
    been polled `ext_ready_after` times, and carries `ext_tracks`."""

    def __init__(self, path_ready: bool = True, cam_ready: bool = False):
        self.requests: list[dict] = []
        self.kicks: list[str] = []  # /v3/webrtcsessions/kick/<id>, in order
        self.path_ready = path_ready
        self.cam_ready = cam_ready
        self.ext_config: dict | None = None
        self.ext_ready_after = 1
        self.ext_polls = 0
        self.ext_tracks: list[str] = ["H264"]
        self.ext_deletes = 0
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def _record(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append({
            "method": self.command, "path": self.path,
            "headers": {k: v for k, v in self.headers.items()}, "body": body,
        })
        return body

    def _reply(self, code: int, headers: dict, body: bytes = b"") -> None:
        self.send_response(code)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._record()
        self._reply(204, {"Access-Control-Allow-Methods": "OPTIONS, GET, POST, PATCH, DELETE",
                          "Link": '<stun:stun.example:3478>; rel="ice-server"'})

    def do_POST(self):
        body = self._record()
        if self.path.startswith("/v3/webrtcsessions/kick/"):
            session_id = self.path.rsplit("/", 1)[1]
            self.server.kicks.append(session_id)
            if session_id == "cam-1":
                self.server.cam_ready = False
            return self._reply(200, {}, b"")
        if self.path == "/v3/config/paths/add/ext":
            if self.server.ext_config is not None:
                return self._reply(400, {"Content-Type": "application/json"},
                                   b'{"error":"path already exists"}')
            self.server.ext_config = json.loads(body or b"{}")
            self.server.ext_polls = 0
            return self._reply(200, {"Content-Type": "application/json"}, b"{}")
        if self.path == "/cam/whip":
            # MediaMTX's WHIP answer: the Location is the bare session
            # secret, a relative reference the proxy must own.
            if self.headers.get("Content-Type") != "application/sdp":
                return self._reply(400, {"Content-Type": "application/json"},
                                   b'{"error":"invalid Content-Type"}')
            self.server.cam_ready = True
            return self._reply(201, {
                "Content-Type": "application/sdp",
                "Location": WHIP_SECRET,
                "ETag": "*", "ID": "cam-1",
                "Accept-Patch": "application/trickle-ice-sdpfrag",
                "Link": '<turn:turn.example:3478>; rel="ice-server"',
            }, b"v=0\r\nwhip-answer-for:" + body)
        if self.path != "/overlay/whep":
            return self._reply(404, {}, b'{"error":"not found"}')
        if self.headers.get("Content-Type") != "application/sdp":
            return self._reply(400, {"Content-Type": "application/json"},
                               b'{"error":"invalid Content-Type"}')
        if not self.server.path_ready:
            return self._reply(404, {"Content-Type": "application/json"},
                               b'{"error":"path is not ready"}')
        self._reply(201, {
            "Content-Type": "application/sdp",
            "Location": "/overlay/whep/0f8b2c1e-1111-2222-3333-444455556666",
            "ETag": "*", "ID": "abc",
            "Link": '<stun:stun.example:3478>; rel="ice-server"',
            "Access-Control-Expose-Headers": "ETag, ID, Accept-Patch, Link, Location",
            "X-Internal": "must-not-leak",
        }, b"v=0\r\nanswer-for:" + body)

    def do_PATCH(self):
        self._record()
        self._reply(204, {})

    def do_DELETE(self):
        self._record()
        if self.path == "/v3/config/paths/delete/ext":
            if self.server.ext_config is None:
                return self._reply(404, {"Content-Type": "application/json"},
                                   b'{"error":"path not found"}')
            self.server.ext_config = None
            self.server.ext_deletes += 1
            return self._reply(200, {"Content-Type": "application/json"}, b"{}")
        if self.path.startswith("/cam/whip/"):
            self.server.cam_ready = False
        self._reply(200, {"Content-Type": "application/json"}, b'{"status":"ok"}')

    def do_GET(self):
        self._record()
        if self.path == "/v3/paths/get/ext":
            if self.server.ext_config is None:
                return self._reply(404, {"Content-Type": "application/json"},
                                   b'{"error":"path not found"}')
            self.server.ext_polls += 1
            if self.server.ext_polls < self.server.ext_ready_after:
                return self._reply(200, {"Content-Type": "application/json"}, json.dumps({
                    "name": "ext", "ready": False, "tracks": [], "bytesReceived": 0,
                    "source": None, "readers": [],
                }).encode())
            return self._reply(200, {"Content-Type": "application/json"}, json.dumps({
                "name": "ext", "ready": True, "tracks": list(self.server.ext_tracks),
                "bytesReceived": 1 << 20,
                "source": {"type": "rtspSource", "id": "ext-1"}, "readers": [],
            }).encode())
        if self.path == "/v3/paths/get/overlay":
            if not self.server.path_ready:
                return self._reply(404, {"Content-Type": "application/json"},
                                   b'{"error":"path not found"}')
            return self._reply(200, {"Content-Type": "application/json"}, json.dumps({
                "name": "overlay", "ready": True, "tracks": ["H264"],
                "bytesReceived": 4096, "source": {"type": "rtspSession", "id": "x"},
                "readers": [{"type": "webRTCSession", "id": "r1"}],
            }).encode())
        if self.path == "/v3/paths/get/cam":
            if not self.server.cam_ready:
                # A path in the config nobody has published to yet.
                return self._reply(200, {"Content-Type": "application/json"}, json.dumps({
                    "name": "cam", "ready": False, "tracks": [], "bytesReceived": 0,
                    "source": None, "readers": [],
                }).encode())
            return self._reply(200, {"Content-Type": "application/json"}, json.dumps({
                "name": "cam", "ready": True, "tracks": ["H264"],
                "bytesReceived": 65536, "source": {"type": "webRTCSession", "id": "cam-1"},
                "readers": [],
            }).encode())
        self._reply(404, {}, b"")


def test_route_recognises_the_seven_shapes_and_nothing_else():
    assert route("/overlay/status") == ("status", None)
    assert route("/overlay/whep") == ("whep", None)
    assert route("/overlay/whep/0f8b2c1e-1111-2222-3333-444455556666") == (
        "whep", "0f8b2c1e-1111-2222-3333-444455556666")
    assert route("/ingest/status") == ("ingest-status", None)
    assert route("/ingest/whip") == ("whip", None)
    assert route(f"/ingest/whip/{WHIP_SECRET}") == ("whip", WHIP_SECRET)
    assert route("/ingest/source") == ("source", None)
    assert route("/ingest/view") == ("view", None)
    for bad in ("/overlay", "/overlay/", "/overlay/whep/", "/overlay/whep/a/b",
                "/overlay/whep/../other", "/overlay/whep/with space",
                "/overlay/whip", "/other/whep", "/overlay/whep/" + "x" * 65,
                "/ingest", "/ingest/", "/ingest/whep", "/ingest/whip/",
                "/ingest/whip/a/b", "/cam/whip", "/ingest/whip/" + "x" * 65,
                "/ingest/source/", "/ingest/source/x", "/overlay/source",
                "/ingest/view/", "/ingest/view/x", "/overlay/view"):
        assert route(bad) is None, bad


def test_own_location_brings_every_relay_shape_into_the_proxy_namespace():
    # MediaMTX: the bare secret.
    assert own_location(WHIP_SECRET, "whip") == f"/ingest/whip/{WHIP_SECRET}"
    assert own_location("abc-123", "whep") == "/overlay/whep/abc-123"
    # A path in the relay's own namespace, absolute or not.
    assert own_location(f"/cam/whip/{WHIP_SECRET}", "whip") == f"/ingest/whip/{WHIP_SECRET}"
    assert own_location(f"cam/whip/{WHIP_SECRET}", "whip") == f"/ingest/whip/{WHIP_SECRET}"
    assert own_location("/overlay/whep/abc", "whep") == "/overlay/whep/abc"
    # A full URL.
    assert own_location(f"http://127.0.0.1:8889/cam/whip/{WHIP_SECRET}/", "whip") == (
        f"/ingest/whip/{WHIP_SECRET}")
    # Anything else is left alone rather than guessed at.
    assert own_location("/somewhere/else", "whip") == "/somewhere/else"
    assert own_location("", "whip") == ""


def test_whep_post_is_forwarded_and_the_answer_comes_back_verbatim():
    relay = StubRelay()
    try:
        proxy = RelayProxy(relay.base, relay.base)
        code, headers, body = proxy.forward(
            "POST", None,
            {"Content-Type": "application/sdp", "Authorization": "Bearer secret",
             "X-Custom": "nope"},
            b"v=0\r\noffer", client="203.0.113.9")
        assert code == 201
        assert body == b"v=0\r\nanswer-for:v=0\r\noffer"
        named = dict(headers)
        assert named["Content-Type"] == "application/sdp"
        assert named["Location"] == "/overlay/whep/0f8b2c1e-1111-2222-3333-444455556666"
        assert named["ETag"] == "*" and named["ID"] == "abc"
        assert "ice-server" in named["Link"]
        assert "X-Internal" not in named
        seen = relay.requests[-1]
        assert seen["path"] == "/overlay/whep" and seen["method"] == "POST"
        assert seen["headers"]["Content-Type"] == "application/sdp"
        assert seen["headers"]["X-Forwarded-For"] == "203.0.113.9"
        assert "Authorization" not in seen["headers"]  # IAM's, not the relay's
        assert "X-Custom" not in seen["headers"]
    finally:
        relay.stop()


def test_patch_delete_and_options_follow_the_session_location():
    relay = StubRelay()
    try:
        proxy = RelayProxy(relay.base, relay.base)
        secret = "0f8b2c1e-1111-2222-3333-444455556666"
        code, _, _ = proxy.forward(
            "PATCH", secret, {"Content-Type": "application/trickle-ice-sdpfrag"},
            b"a=candidate")
        assert code == 204
        assert relay.requests[-1]["path"] == f"/overlay/whep/{secret}"
        assert relay.requests[-1]["body"] == b"a=candidate"
        code, headers, body = proxy.forward("DELETE", secret, {}, b"")
        assert code == 200 and json.loads(body) == {"status": "ok"}
        code, headers, _ = proxy.forward(
            "OPTIONS", None, {"Access-Control-Request-Method": "POST"}, b"")
        assert code == 204
        assert dict(headers)["Access-Control-Allow-Methods"].startswith("OPTIONS")
        assert relay.requests[-1]["headers"]["Access-Control-Request-Method"] == "POST"
    finally:
        relay.stop()


def test_relay_errors_are_answers_and_a_relay_that_is_down_is_a_502():
    relay = StubRelay(path_ready=False)
    try:
        proxy = RelayProxy(relay.base, relay.base)
        code, _, body = proxy.forward(
            "POST", None, {"Content-Type": "application/sdp"}, b"v=0")
        assert code == 404 and b"path is not ready" in body
        code, _, _ = proxy.forward("POST", None, {"Content-Type": "text/plain"}, b"x")
        assert code == 400
    finally:
        relay.stop()
    down = RelayProxy("http://127.0.0.1:1", "http://127.0.0.1:1", timeout_s=1.0)
    code, headers, body = down.forward(
        "POST", None, {"Content-Type": "application/sdp"}, b"v=0")
    assert code == 502 and json.loads(body)["error"] == "relay unreachable"
    assert down.relay_status() is None


def test_methods_secrets_and_sizes_outside_whep_are_refused_before_the_relay():
    proxy = RelayProxy("http://127.0.0.1:1", "http://127.0.0.1:1", body_cap=16)
    assert proxy.forward("GET", None, {}, b"")[0] == 405
    assert proxy.forward("PUT", None, {}, b"")[0] == 405
    assert proxy.forward("PATCH", None, {}, b"")[0] == 405  # no secret
    assert proxy.forward("POST", "abc", {}, b"")[0] == 405  # with one
    assert proxy.forward("POST", None, {}, b"x" * 17)[0] == 413
    # The same gate guards the ingest leg; an unknown leg is a programming error.
    assert proxy.forward("GET", None, {}, b"", leg="whip")[0] == 405
    assert proxy.forward("POST", None, {}, b"x" * 17, leg="whip")[0] == 413
    try:
        proxy.forward("POST", None, {}, b"", leg="rtsp")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown leg must not reach the relay")


def test_whip_ingest_is_forwarded_to_cam_and_its_location_is_owned():
    relay = StubRelay()
    try:
        proxy = RelayProxy(relay.base, relay.base)
        code, body = proxy.ingest_status()
        assert code == 200
        assert body["ready"] is False and body["tracks"] == []
        assert body["path"] == "cam" and body["whip"] == "/ingest/whip"

        code, headers, body = proxy.forward(
            "POST", None, {"Content-Type": "application/sdp", "Authorization": "x"},
            b"v=0\r\ncamera-offer", client="198.51.100.7", leg="whip")
        assert code == 201
        assert body == b"v=0\r\nwhip-answer-for:v=0\r\ncamera-offer"
        named = dict(headers)
        assert named["Location"] == f"/ingest/whip/{WHIP_SECRET}"
        assert named["Accept-Patch"] == "application/trickle-ice-sdpfrag"
        assert "ice-server" in named["Link"]
        seen = relay.requests[-1]
        assert seen["path"] == "/cam/whip" and seen["method"] == "POST"
        assert seen["headers"]["X-Forwarded-For"] == "198.51.100.7"
        assert "Authorization" not in seen["headers"]

        code, body = proxy.ingest_status()
        assert code == 200 and body["ready"] is True and body["tracks"] == ["H264"]
        # Which camera this is: the phone's, straight off the relay.
        assert body["source"] == {"kind": "phone", "path": "cam", "ready": True,
                                  "tracks": ["H264"], "type": "webRTCSession",
                                  "since": None}
        assert body["phone"] == {"ready": True, "tracks": ["H264"]}
        assert body["streamUrl"] == "rtsp://127.0.0.1:8554/cam"

        code, _, _ = proxy.forward(
            "PATCH", WHIP_SECRET, {"Content-Type": "application/trickle-ice-sdpfrag"},
            b"a=candidate", leg="whip")
        assert code == 204
        assert relay.requests[-1]["path"] == f"/cam/whip/{WHIP_SECRET}"
        code, _, _ = proxy.forward("DELETE", WHIP_SECRET, {}, b"", leg="whip")
        assert code == 200
        assert proxy.ingest_status()[1]["ready"] is False
    finally:
        relay.stop()
    down = RelayProxy("http://127.0.0.1:1", "http://127.0.0.1:1", timeout_s=1.0)
    code, body = down.ingest_status()
    assert code == 502 and body["ready"] is False and body["error"] == "relay unreachable"


class FakeOverlay:
    def snapshot(self) -> dict:
        return {"lastState": "exact"}


def test_status_says_not_yet_without_a_session_and_reports_with_one():
    relay = StubRelay()
    try:
        proxy = RelayProxy(relay.base, relay.base)
        code, body = proxy.status(None)
        assert code == 503 and body["publishing"] is False
        assert body["relay"]["ready"] is True and body["relay"]["readers"] == 1
        assert body["whep"] == "/overlay/whep"
        session = argparse.Namespace(overlay=FakeOverlay(), run_name="session-1")
        code, body = proxy.status(session)
        assert code == 200
        assert body["session"] == "session-1" and body["publishing"] is True
        assert body["overlay"] == {"lastState": "exact"}
        assert body["relay"]["tracks"] == ["H264"]
        code, body = proxy.status(argparse.Namespace(overlay=None, run_name=""))
        assert code == 503 and body["error"] == "session has no overlay"
    finally:
        relay.stop()
    idle = RelayProxy("http://127.0.0.1:1", "http://127.0.0.1:1", timeout_s=1.0)
    code, body = idle.status(None)
    assert code == 502 and body["relay"] is None


# -- through the producer's handler ----------------------------------------------


def server_args(**overrides) -> argparse.Namespace:
    base = dict(stream="", pose="sideload", pose_fps=6.0, device="cpu",
                track="", analysis_socket="", sink_dir="/tmp/x", run="",
                capture_bucket="", post_url="", post_interval_s=1.0,
                duration=0.0, serve=True, port=0, teardown_drain_s=1.0,
                overlay_publish="", overlay_record=False,
                overlay_relay_webrtc="", overlay_relay_api="")
    base.update(overrides)
    return argparse.Namespace(**base)


def request(port: int, method: str, path: str, body: bytes = b"",
            headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response.status, dict(response.getheaders()), payload


def test_nothing_is_served_under_overlay_unless_it_is_configured():
    server = producer.build_server(server_args(), Telemetry())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert request(port, "GET", "/overlay/status")[0] == 404
        assert request(port, "POST", "/overlay/whep", b"v=0",
                       {"Content-Type": "application/sdp"})[0] == 404
        assert request(port, "GET", "/ingest/status")[0] == 404
        assert request(port, "POST", "/ingest/whip", b"v=0",
                       {"Content-Type": "application/sdp"})[0] == 404
        assert request(port, "GET", "/healthz")[0] == 200
        code, _, body = request(port, "GET", "/statz")
        assert code == 200 and json.loads(body)["overlay"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_the_handler_proxies_whep_when_a_session_is_up_and_refuses_before():
    relay = StubRelay()
    server = producer.build_server(
        server_args(overlay_publish="rtsp://127.0.0.1:8554/overlay",
                    overlay_relay_webrtc=relay.base, overlay_relay_api=relay.base),
        Telemetry())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        # Before the session: a clear 503, not the relay's 404.
        code, _, body = request(port, "POST", "/overlay/whep", b"v=0",
                                {"Content-Type": "application/sdp"})
        assert code == 503 and json.loads(body)["whep"] == "/overlay/whep"
        code, _, body = request(port, "GET", "/overlay/status")
        assert code == 503 and json.loads(body)["relay"]["ready"] is True
        # Preflight works regardless (the subscriber's backend may probe).
        assert request(port, "OPTIONS", "/overlay/whep", b"",
                       {"Access-Control-Request-Method": "POST"})[0] == 204
        # With a session: the exchange goes through and Location keeps the prefix.
        server.current["session"] = argparse.Namespace(
            overlay=FakeOverlay(), run_name="session-live")
        code, headers, body = request(port, "POST", "/overlay/whep", b"v=0\r\noffer",
                                      {"Content-Type": "application/sdp"})
        assert code == 201 and body.startswith(b"v=0\r\nanswer-for:")
        location = headers["Location"]
        assert location.startswith("/overlay/whep/")
        assert headers["Content-Type"] == "application/sdp"
        assert "X-Internal" not in headers
        code, _, _ = request(port, "PATCH", location, b"a=candidate",
                             {"Content-Type": "application/trickle-ice-sdpfrag"})
        assert code == 204
        assert request(port, "DELETE", location)[0] == 200
        assert request(port, "PUT", location)[0] == 405
        assert request(port, "POST", "/overlay/whep", b"x" * (64 * 1024 + 1),
                       {"Content-Type": "application/sdp"})[0] == 413
        code, _, body = request(port, "GET", "/overlay/status")
        assert code == 200 and json.loads(body)["session"] == "session-live"
        code, _, body = request(port, "GET", "/statz")
        assert json.loads(body)["overlay"] == {"lastState": "exact"}
        assert request(port, "PATCH", "/statz")[0] == 404
    finally:
        server.current["session"] = None
        server.shutdown()
        server.server_close()
        relay.stop()


def test_the_handler_takes_a_camera_over_whip_with_no_session_at_all():
    relay = StubRelay()
    server = producer.build_server(
        server_args(overlay_publish="rtsp://127.0.0.1:8554/overlay",
                    overlay_relay_webrtc=relay.base, overlay_relay_api=relay.base),
        Telemetry())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert server.current["session"] is None
        # Nobody publishing yet: a normal 200 saying so, for the poller.
        code, _, body = request(port, "GET", "/ingest/status")
        assert code == 200
        assert json.loads(body) == {
            "ready": False, "readers": 0, "tracks": [], "bytesReceived": 0,
            "path": "cam", "streamUrl": "rtsp://127.0.0.1:8554/cam",
            "source": {"kind": "phone", "path": "cam", "ready": False, "tracks": [],
                       "type": None, "since": None},
            "phone": {"ready": False, "tracks": []}, "whip": "/ingest/whip"}
        # The external camera is a Confidential Space feature: outside --tee
        # the route does not exist.
        assert request(port, "PUT", "/ingest/source", b'{"url":"rtsps://c.example/x"}',
                       {"Content-Type": "application/json"})[0] == 404
        assert request(port, "GET", "/ingest/source")[0] == 404
        # So is the home connector's gateway it goes through.
        assert request(port, "POST", "/tunnel/expect", b"{}",
                       {"Content-Type": "application/json"})[0] == 404
        assert request(port, "POST", "/tunnel/clear")[0] == 404
        assert request(port, "POST", "/ingest/status")[0] == 405
        # The phone's offer goes through without a session; the Location
        # it must follow is on this port.
        code, headers, body = request(port, "POST", "/ingest/whip", b"v=0\r\ncam",
                                      {"Content-Type": "application/sdp"})
        assert code == 201 and body == b"v=0\r\nwhip-answer-for:v=0\r\ncam"
        location = headers["Location"]
        assert location == f"/ingest/whip/{WHIP_SECRET}"
        assert headers["Accept-Patch"] == "application/trickle-ice-sdpfrag"
        code, _, body = request(port, "GET", "/ingest/status")
        assert code == 200 and json.loads(body)["ready"] is True
        assert json.loads(body)["tracks"] == ["H264"]
        # Trickle ICE and hang-up follow the same Location.
        assert request(port, "PATCH", location, b"a=candidate",
                       {"Content-Type": "application/trickle-ice-sdpfrag"})[0] == 204
        assert relay.requests[-1]["path"] == f"/cam/whip/{WHIP_SECRET}"
        assert request(port, "PUT", location)[0] == 405
        assert request(port, "DELETE", location)[0] == 200
        assert json.loads(request(port, "GET", "/ingest/status")[2])["ready"] is False
        # The overlay leg is still gated on a session.
        assert request(port, "POST", "/overlay/whep", b"v=0",
                       {"Content-Type": "application/sdp"})[0] == 503
    finally:
        server.shutdown()
        server.server_close()
        relay.stop()
