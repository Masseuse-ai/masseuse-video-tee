"""The producer's end of the analysis socket against a scripted analysis
process: hello/ready handshake, inbound dispatch, stop/summary, and the
null link when no socket is configured."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

import analysis_link  # noqa: E402


class FakeAnalysis:
    """Listens once, answers hello, echoes what it is told to."""

    def __init__(self, script):
        # macOS caps AF_UNIX paths at 104 bytes; keep it short.
        self.dir = tempfile.mkdtemp(prefix="al-", dir="/tmp")
        self.path = str(Path(self.dir) / "a.sock")
        self.received: list[dict] = []
        self.script = script
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.server.accept()
        with conn, conn.makefile("rb") as lines:
            for raw in lines:
                message = json.loads(raw)
                self.received.append(message)
                for reply in self.script(message):
                    conn.sendall((json.dumps(reply) + "\n").encode())
                if message.get("kind") == "stop":
                    break


def _script(message):
    kind = message.get("kind")
    if kind == "hello":
        return [{"kind": "ready", "protocol": analysis_link.PROTOCOL,
                 "version": "t-1", "modelVersion": "m/1"}]
    if kind == "frame":
        return [{"kind": "post", "body": {"atS": message["atS"], "n": 1}},
                {"kind": "hud", "lines": ["one", "two"]},
                {"kind": "gauges", "values": {"backlog": 2}},
                {"kind": "mystery", "x": 1}]
    if kind == "stop":
        return [{"kind": "summary", "summary": {"analysis": {"frames": 1},
                                                "counters": {"hijack": 1}}}]
    return []


def test_handshake_dispatch_and_summary():
    fake = FakeAnalysis(_script)
    got: list[dict] = []
    errors: list[str] = []
    link = analysis_link.open_link(fake.path, got.append, errors.append,
                                   fps=30.0, poseFps=9.0, postIntervalS=1.0,
                                   run=None)
    assert link.connected
    assert link.ready["version"] == "t-1"
    hello = fake.received[0]
    assert hello["kind"] == "hello" and hello["protocol"] == analysis_link.PROTOCOL
    assert hello["fps"] == 30.0 and hello["poseFps"] == 9.0 and hello["run"] is None

    link.send({"kind": "frame", "frame": 3, "atS": 0.1, "keypoints": None,
               "fast": None, "slow": None})
    deadline = time.monotonic() + 5
    while len(got) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    kinds = [m["kind"] for m in got]
    # ready and summary are consumed by the link; everything else, known or
    # not, reaches the session's handler in order.
    assert kinds == ["post", "hud", "gauges", "mystery"]
    summary = link.stop(0.1)
    assert summary == {"analysis": {"frames": 1}, "counters": {"hijack": 1}}
    assert fake.received[-1] == {"kind": "stop", "atS": 0.1}
    assert errors == []


def test_null_link_when_no_socket_is_configured():
    link = analysis_link.open_link("", lambda m: None)
    assert not link.connected
    link.send({"kind": "frame"})
    assert link.stop(1.0) == {}


def test_missing_socket_is_a_link_error():
    with pytest.raises(analysis_link.LinkError):
        analysis_link.AnalysisLink("/tmp/al-none/nope.sock", lambda m: None,
                                   connect_timeout_s=0.3)


def test_protocol_mismatch_is_refused():
    def script(message):
        if message.get("kind") == "hello":
            return [{"kind": "ready", "protocol": 99}]
        return []

    fake = FakeAnalysis(script)
    with pytest.raises(analysis_link.LinkError):
        analysis_link.open_link(fake.path, lambda m: None)


def test_peer_going_away_does_not_hang_stop():
    def script(message):
        if message.get("kind") == "hello":
            return [{"kind": "ready", "protocol": analysis_link.PROTOCOL}]
        return []

    fake = FakeAnalysis(script)
    link = analysis_link.open_link(fake.path, lambda m: None)
    fake.server.close()
    # The fake's serve loop ends on stop without answering.
    started = time.monotonic()
    assert link.stop(2.0, timeout_s=1.0) == {}
    assert time.monotonic() - started < 3.0
