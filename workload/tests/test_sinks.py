"""The Poster against a local capture server: body shape, auth header,
token reuse, and the never-stall failure contract."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))

from sinks import Poster  # noqa: E402
from telemetry import Telemetry  # noqa: E402


def capture_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length)),
            })
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_the_poster_sends_the_assembled_body_with_its_token():
    server, received = capture_server()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/pose-signals/readings"
    telemetry = Telemetry()
    minted = []

    def supplier():
        minted.append(1)
        return "token-abc"

    poster = Poster(url, telemetry, token_supplier=supplier)
    poster.post({"atS": 75.0, "ratePerMin": 42.0, "continuityPct": None,
                 "calibrationReady": True,
                 "recentOnsetsS": [70.1, 71.4],
                 "posture": {"hipElevationPx": 3.0,
                             "feetAboveHipsFrac": 0.0,
                             "hipMotionRatio": 1.1},
                 "modelVersion": "workload/v2"})
    # Sends are asynchronous and the newest body replaces an unsent one;
    # let the first land before offering the second.
    for _ in range(200):
        if telemetry.snapshot()["counters"].get("postsOk") == 1:
            break
        time.sleep(0.01)
    poster.post({"atS": 80.0, "ratePerMin": 44.0, "continuityPct": 61.2,
                 "calibrationReady": True, "recentOnsetsS": [],
                 "posture": None, "modelVersion": "workload/v2"})
    poster.close(timeout_s=5)
    server.shutdown()

    assert len(received) == 2
    first = received[0]
    assert first["authorization"] == "Bearer token-abc"
    assert first["body"]["recentOnsetsS"] == [70.1, 71.4]
    assert first["body"]["posture"]["hipMotionRatio"] == 1.1
    assert first["body"]["modelVersion"] == "workload/v2"
    assert received[1]["body"]["continuityPct"] == 61.2
    assert received[1]["body"]["posture"] is None
    assert len(minted) == 1, "the token is reused inside its lifetime"
    assert telemetry.snapshot()["counters"]["postsOk"] == 2


def test_a_dead_sink_is_counted_never_raised():
    telemetry = Telemetry()
    poster = Poster("http://127.0.0.1:9/api/pose-signals/readings", telemetry,
                    timeout_s=0.2, token_supplier=lambda: None)
    poster.post({"ratePerMin": 30.0, "calibrationReady": True})
    poster.close(timeout_s=5)
    counters = telemetry.snapshot()["counters"]
    assert counters["postsFailed"] == 1
    assert counters["postsUnauthenticated"] == 1


def test_the_poster_never_blocks_the_pipeline_and_keeps_the_newest():
    """A slow sink stalls the sender, not the caller; bodies queued behind
    an in-flight send collapse to the newest one."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(json.loads(self.rfile.read(length)))
            release.wait(5)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    telemetry = Telemetry()
    poster = Poster(
        f"http://127.0.0.1:{server.server_address[1]}/x", telemetry,
        timeout_s=10, token_supplier=lambda: "t")
    poster.post({"atS": 1.0})
    for _ in range(500):
        if received:
            break
        time.sleep(0.01)
    assert received, "the first body is in flight, held by the sink"
    started = time.monotonic()
    for at in (2.0, 3.0, 4.0):
        poster.post({"atS": at})
    assert time.monotonic() - started < 0.5, "post() returned at once"
    release.set()
    poster.close(timeout_s=10)
    server.shutdown()

    assert [body["atS"] for body in received] == [1.0, 4.0]
    counters = telemetry.snapshot()["counters"]
    assert counters["postsOk"] == 2
    assert counters["postsCoalesced"] == 2


def test_capture_records_posts_for_the_consumers_gates(tmp_path):
    from sinks import Capture

    capture = Capture(tmp_path)
    body = {"atS": 75.0, "recentOnsetsS": [70.1], "provisionalOnsetsS": [74.2],
            "posture": {"hipElevationPx": 12.5, "feetAboveHipsFrac": 0.0,
                        "hipMotionRatio": 4.76},
            "calibrationReady": True}
    capture.post(body)
    capture.close()
    rows = [json.loads(line)
            for line in (tmp_path / "posts.jsonl").read_text().splitlines()]
    assert rows == [body]


def test_capture_keeps_every_pose_row_dropped_ones_included(tmp_path):
    from sinks import Capture

    capture = Capture(tmp_path)
    posed = {"frame": 0, "atS": 0.0, "wallS": 1700000000.0,
             "box": [1.0, 2.0, 3.0, 4.0], "boxScore": 0.9, "people": 1,
             "keypoints": {"left_elbow": [10.0, 20.0, 0.95]}}
    dropped = {"frame": 5, "atS": 0.1667, "wallS": 1700000000.2,
               "keypoints": None, "dropped": True}
    capture.pose(posed)
    capture.pose(dropped)
    capture.close()
    rows = [json.loads(line)
            for line in (tmp_path / "poses.jsonl").read_text().splitlines()]
    assert rows == [posed, dropped]


class FakeBlob:
    def __init__(self, store, name, fail):
        self.store, self.name, self.fail = store, name, fail

    def upload_from_filename(self, path):
        if self.fail(self.name):
            raise OSError("refused")
        self.store[self.name] = Path(path).read_bytes()


class FakeBucket:
    def __init__(self, store, fail):
        self.store, self.fail = store, fail

    def blob(self, name):
        return FakeBlob(self.store, name, self.fail)


class FakeClient:
    def __init__(self, store, fail=lambda name: False):
        self.store, self.fail = store, fail

    def bucket(self, name):
        self.store.setdefault("__bucket__", name)
        return FakeBucket(self.store, self.fail)


def test_capture_upload_copies_each_file_under_the_run_prefix(tmp_path):
    from sinks import Capture

    telemetry = Telemetry()
    capture = Capture(tmp_path)
    capture.pose({"frame": 0, "atS": 0.0, "keypoints": None, "dropped": True})
    capture.post({"atS": 1.0})
    capture.summary({"counters": {}})
    capture.close()
    store: dict = {}

    result = capture.upload("my-bucket", "/runs/session-file-1/", telemetry,
                            client_factory=lambda: FakeClient(store))

    assert store["__bucket__"] == "my-bucket"
    names = sorted(k for k in store if k != "__bucket__")
    assert names == [
        "runs/session-file-1/events.jsonl", "runs/session-file-1/onsets.jsonl",
        "runs/session-file-1/payloads.jsonl", "runs/session-file-1/poses.jsonl",
        "runs/session-file-1/posts.jsonl", "runs/session-file-1/summary.json",
    ]
    assert json.loads(store["runs/session-file-1/poses.jsonl"])["dropped"]
    assert result["failed"] == [] and len(result["uploaded"]) == 6
    assert telemetry.snapshot()["counters"]["captureUploaded"] == 6


def test_capture_upload_failures_are_counted_never_raised(tmp_path):
    from sinks import Capture

    telemetry = Telemetry()
    capture = Capture(tmp_path)
    capture.close()
    store: dict = {}

    partial = capture.upload(
        "b", "runs/x", telemetry,
        client_factory=lambda: FakeClient(
            store, fail=lambda name: name.endswith("poses.jsonl")))
    assert partial["failed"] == ["poses.jsonl"]
    assert len(partial["uploaded"]) == 4

    def broken_client():
        raise RuntimeError("no credentials")

    dead = capture.upload("b", "runs/x", telemetry,
                          client_factory=broken_client)
    assert dead["uploaded"] == [] and len(dead["failed"]) == 5
    assert "no credentials" in dead["error"]
    counters = telemetry.snapshot()["counters"]
    assert counters["captureUploaded"] == 4
    assert counters["captureUploadFailed"] == 1 + 5
