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
    """Listens, answers hello, echoes what it is told to; `connections`
    accepted in turn (one by default), so a reconnect can be served."""

    def __init__(self, script, connections: int = 1):
        # macOS caps AF_UNIX paths at 104 bytes; keep it short.
        self.dir = tempfile.mkdtemp(prefix="al-", dir="/tmp")
        self.path = str(Path(self.dir) / "a.sock")
        self.received: list[dict] = []
        self.script = script
        self.connections = connections
        self.accepted = 0
        self.conn: socket.socket | None = None
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(2)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def drop(self) -> None:
        """The analysis's end goes away without a word (2026-09-16: its
        connection thread died on a message)."""
        conn = self.conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def _serve(self):
        for _ in range(self.connections):
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            self.accepted += 1
            self.conn = conn
            try:
                with conn, conn.makefile("rb") as lines:
                    for raw in lines:
                        message = json.loads(raw)
                        self.received.append(message)
                        for reply in self.script(message):
                            conn.sendall((json.dumps(reply) + "\n").encode())
                        if message.get("kind") == "stop":
                            return
            except OSError:
                continue


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


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_broken_link_is_said_once_drops_meanwhile_and_resumes_on_reconnect():
    def script(message):
        kind = message.get("kind")
        if kind == "hello":
            return [{"kind": "ready", "protocol": analysis_link.PROTOCOL,
                     "version": "t-2", "resumed": message.get("resume") is True}]
        if kind == "frame":
            return [{"kind": "post", "body": {"atS": message["atS"]}}]
        if kind == "stop":
            return [{"kind": "summary", "summary": {"analysis": {"frames": 9}}}]
        return []

    fake = FakeAnalysis(script, connections=2)
    got: list[dict] = []
    errors: list[str] = []
    link = analysis_link.open_link(fake.path, got.append, errors.append,
                                   fps=30.0, poseFps=9.0, postIntervalS=1.0,
                                   run=None, sessionId="s-1")
    assert link.ready["resumed"] is False
    assert fake.received[0]["sessionId"] == "s-1" and "resume" not in fake.received[0]
    link.send({"kind": "frame", "frame": 1, "atS": 0.1})
    assert _wait(lambda: len(got) == 1)

    # The analysis's end goes away. The next send fails once, the link is
    # broken and says so once; what follows is dropped, counted, and the
    # link stays open for the session.
    fake.drop()
    assert _wait(lambda: link.broken, 3.0), "the reader sees the end of the stream"
    for i in range(5):
        link.send({"kind": "frame", "frame": 2 + i, "atS": 0.2 + i / 10})
    assert link.broken
    assert link.dropped >= 4
    assert len(errors) <= 1, errors
    assert len(got) == 1, "nothing more came back"

    # Reconnect: a new connection, the hello's fields again with `resume`,
    # the analysis says it kept the session, and sends flow again.
    ready = link.reconnect()
    assert ready["resumed"] is True
    assert not link.broken and link.reconnects == 1
    hello2 = fake.received[-1]
    assert hello2["kind"] == "hello" and hello2["resume"] is True
    assert hello2["sessionId"] == "s-1" and hello2["fps"] == 30.0
    assert fake.accepted == 2
    link.send({"kind": "frame", "frame": 9, "atS": 0.9})
    assert _wait(lambda: len(got) == 2)
    assert got[-1]["body"]["atS"] == 0.9
    summary = link.stop(0.9)
    assert summary == {"analysis": {"frames": 9}}
    assert fake.received[-1]["kind"] == "stop"


def test_reconnect_to_nobody_leaves_the_link_broken_and_stop_does_not_hang():
    def script(message):
        if message.get("kind") == "hello":
            return [{"kind": "ready", "protocol": analysis_link.PROTOCOL}]
        return []

    fake = FakeAnalysis(script)
    link = analysis_link.open_link(fake.path, lambda m: None, sessionId="s-2")
    fake.drop()
    fake.server.close()
    Path(fake.path).unlink()
    assert _wait(lambda: link.broken, 3.0)
    with pytest.raises(analysis_link.LinkError):
        link.reconnect(connect_timeout_s=0.3)
    assert link.broken
    link.send({"kind": "frame"})
    assert link.dropped >= 1
    started = time.monotonic()
    assert link.stop(1.0, timeout_s=1.0) == {}
    assert time.monotonic() - started < 1.0, "a broken link asks nothing"
