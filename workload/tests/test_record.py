"""The session's record (producer/record.py): parts on the trainer's
30-second grid, the keypoint Parquet with all 308 points of a view, the
uploader's write-once objects, the lease that names the prefix, and the
session that routes every stream into it - with the flat capture and the
overlay's tap unchanged beside it."""

from __future__ import annotations

import argparse
import gzip
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "producer"))
sys.path.insert(0, str(WORKLOAD / "pixel"))

import producer  # noqa: E402
import record as record_module  # noqa: E402
import tee_mode  # noqa: E402
from live_pose import ViewState, share_full_result  # noqa: E402
from record import (KEYPOINT_COUNT, KeypointStream, PartStream, Record,  # noqa: E402
                    Uploader, parse_record, part_name, window_start_s)
from sinks import Capture  # noqa: E402
from telemetry import Telemetry  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ACCOUNT = "019966a0-0000-7000-8000-000000000001"
SESSION = "019966a0-0000-7000-8000-000000000002"
PREFIX = f"{ACCOUNT}/estim_sessions/{SESSION}/enclave"
BUCKET = "masseuse-ai-prod"
T0 = datetime(2026, 9, 15, 5, 12, 30, tzinfo=timezone.utc).timestamp()


class FakeGcs:
    """google-cloud-storage's surface as the uploader uses it: objects
    created once, an existing name a 412, a scripted failure per name."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, int | None]] = []
        self.fail: dict[str, int] = {}
        self.lock = threading.Lock()

    def bucket(self, name):
        gcs = self

        class Blob:
            def __init__(self, object_name):
                self.name = object_name

            def upload_from_filename(self, path, content_type=None,
                                     if_generation_match=None):
                with gcs.lock:
                    gcs.calls.append((self.name, if_generation_match))
                    left = gcs.fail.get(self.name, 0)
                    if left:
                        gcs.fail[self.name] = left - 1
                        raise OSError("transient")
                    if self.name in gcs.objects and if_generation_match == 0:
                        raise PreconditionFailed("412 precondition failed")
                    gcs.objects[self.name] = (Path(path).read_bytes(), content_type)

        class Bucket:
            def __init__(self, bucket_name):
                self.name = bucket_name

            def blob(self, object_name):
                return Blob(object_name)

        return Bucket(name)


class PreconditionFailed(Exception):
    code = 412


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def read_parquet(gcs: FakeGcs, name: str) -> dict:
    return pq.read_table(pa.BufferReader(gcs.objects[name][0])).to_pydict()


def rows_of(gcs: FakeGcs, name: str) -> list[dict]:
    data, content_type = gcs.objects[name]
    assert content_type == "application/gzip"
    return [json.loads(line) for line in gzip.decompress(data).decode().splitlines()]


def keypoints(seed: int):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1280, size=(KEYPOINT_COUNT, 2)).astype(np.float32)
    scores = rng.uniform(0, 1, size=KEYPOINT_COUNT).astype(np.float32)
    return xy, scores


# -- names and the lease's record --------------------------------------------------


def test_parts_are_named_by_utc_window_start_like_the_trainers():
    assert window_start_s(T0 + 17.9) == int(T0)
    assert window_start_s(T0 + 30.0) == int(T0) + 30
    assert part_name(int(T0)) == "part-20260915T051230Z"
    assert part_name(window_start_s(T0 + 59.999)) == "part-20260915T051300Z"


def test_the_lease_record_is_validated():
    assert parse_record(None) == (None, "")
    assert parse_record({"prefix": PREFIX}) == ({"prefix": PREFIX, "partSeconds": 30}, "")
    assert parse_record({"prefix": f"/{PREFIX}/", "partSeconds": "15"})[0]["partSeconds"] == 15
    assert parse_record("x")[0] is None
    assert parse_record({"prefix": "runs/x"})[0] is None
    assert parse_record({"prefix": f"{ACCOUNT}/estim_sessions/{SESSION}"})[0] is None
    assert parse_record({"prefix": PREFIX.upper()})[0] is None
    assert parse_record({"prefix": PREFIX, "partSeconds": 2})[0] is None
    assert parse_record({"prefix": PREFIX, "partSeconds": "x"})[0] is None


def test_the_lease_carries_the_record_only_with_a_capture_bucket():
    clock = Clock()
    body = {"sessionId": "s1", "capabilityHash": "ab" * 32,
            "record": {"prefix": PREFIX, "partSeconds": 30}}
    bare = tee_mode.Lease(clock=clock)
    code, answer = bare.grant(body)
    assert code == 400 and "capture bucket" in answer["error"]
    assert bare.active() is False
    code, answer = bare.grant({**body, "record": None})
    assert code == 200 and "record" not in answer and bare.record_for() is None

    lease = tee_mode.Lease(clock=clock, capture_bucket=BUCKET)
    code, answer = lease.grant({**body, "record": {"prefix": "nope"}})
    assert code == 400 and "record.prefix" in answer["error"]
    code, answer = lease.grant(body)
    assert code == 200
    assert answer["record"] == {"prefix": PREFIX, "partSeconds": 30, "bucket": BUCKET}
    assert lease.record_for() == {"prefix": PREFIX, "partSeconds": 30,
                                  "bucket": BUCKET, "sessionId": "s1"}
    assert lease.snapshot()["record"]["prefix"] == PREFIX
    lease.clear()
    assert lease.record_for() is None and lease.snapshot()["record"] is None


def test_tee_config_reads_and_checks_the_capture_bucket():
    env = {"TEE_PUBLIC_HOST": "slot-0.tee.example", "TRAINER_INVOKER_SERVICE_ACCOUNT": "sa@x"}
    assert tee_mode.TeeConfig.from_env(env).capture_bucket == ""
    with_bucket = tee_mode.TeeConfig.from_env({**env, "TEE_CAPTURE_BUCKET": BUCKET})
    assert with_bucket.capture_bucket == BUCKET
    with pytest.raises(SystemExit):
        tee_mode.TeeConfig.from_env({**env, "TEE_CAPTURE_BUCKET": "Not A Bucket"})
    tee = tee_mode.TeeMode(tee_mode.TeeConfig.from_env({**env, "TEE_CAPTURE_BUCKET": BUCKET}),
                           launcher=object(), clock=Clock())
    assert tee.lease.capture_bucket == BUCKET


# -- part streams -------------------------------------------------------------------


def test_a_jsonl_stream_rolls_on_the_grid_and_ticks_a_finished_window_out(tmp_path):
    clock = Clock()
    parts = []
    stream = PartStream("onsets", tmp_path, 30, clock=clock,
                        on_part=lambda s, path, window: parts.append((path, window)))
    assert stream.append({"atS": 1.0})
    clock.now = T0 + 29.0
    assert stream.append({"atS": 2.0, "wallS": T0 + 28.5})
    assert not parts, "the window is still open"
    clock.now = T0 + 30.5
    assert stream.append({"atS": 3.0})
    assert len(parts) == 1
    path, window = parts[0]
    assert window == int(T0) and path.name == "part-20260915T051230Z.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert [row["atS"] for row in rows] == [1.0, 2.0]
    assert rows[0]["wallS"] == T0 and rows[1]["wallS"] == T0 + 28.5
    # A row stamped before the roll it lost the race with rides along, counted.
    assert stream.append({"atS": 4.0, "wallS": T0 + 29.9})
    assert stream.snapshot()["late"] == 1
    # Nothing new for a while: the tick closes the window when it ends.
    stream.tick(T0 + 59.0)
    assert len(parts) == 1
    stream.tick(T0 + 60.0)
    assert len(parts) == 2 and parts[1][1] == int(T0) + 30
    second = [json.loads(line) for line in gzip.open(parts[1][0], "rt")]
    assert [row["atS"] for row in second] == [3.0, 4.0]
    # An empty window leaves nothing behind; close() hands the last part on.
    stream.tick(T0 + 120.0)
    clock.now = T0 + 125.0
    stream.append({"atS": 5.0})
    stream.close()
    assert len(parts) == 3 and parts[2][1] == int(T0) + 120
    assert stream.append({"atS": 6.0}) is False and stream.snapshot()["dropped"] == 1
    assert stream.snapshot()["rows"] == 5 and stream.snapshot()["parts"] == 3


def test_a_keypoint_stream_writes_all_308_points_and_the_gaps_as_parquet(tmp_path):
    clock = Clock()
    parts = []
    stream = KeypointStream("poses", "body", tmp_path, 30, metadata={"sessionId": "s1"},
                            clock=clock, on_part=lambda s, path, window: parts.append(path))
    xy0, sc0 = keypoints(0)
    assert stream.append_step(0, 0.0, xy0, sc0, (100.0, 50.0, 700.0, 650.0), 0.91, 1, False,
                              frame_size=(1280, 720))
    clock.now = T0 + 0.111
    # Nobody in the frame: the step said so with no keypoints.
    assert stream.append_step(3, 0.1, None, None, None, None, 0, False, frame_size=(1280, 720))
    clock.now = T0 + 0.222
    # A dropped step from the worker (the gap row), stamped by the worker.
    assert stream.append_step(7, 0.2333, None, None, None, None, None, False,
                              dropped=True, wall_s=T0 + 0.2)
    xy1, sc1 = keypoints(1)
    clock.now = T0 + 0.333
    assert stream.append_step(10, 0.3333, xy1, sc1, (110.0, 55.0, 710.0, 655.0), 0.93, 2, True,
                              frame_size=(1280, 720))
    # A model with another layout is kept keypoint-less and said in `error`.
    clock.now = T0 + 0.444
    assert stream.append_step(13, 0.4333, np.zeros((21, 2)), np.zeros(21), None, None, 1, False)
    stream.close()
    assert len(parts) == 1 and parts[0].name == "part-20260915T051230Z.parquet"
    table = pq.read_table(parts[0])
    assert table.num_rows == 5
    assert table.schema.metadata[b"sessionId"] == b"s1"
    assert table.schema.metadata[b"view"] == b"body"
    assert json.loads(table.schema.metadata[b"keypoints"])["count"] == KEYPOINT_COUNT
    cols = table.to_pydict()
    assert cols["frame"] == [0, 3, 7, 10, 13]
    assert cols["view"] == ["body"] * 5
    assert cols["wallS"] == [T0, round(T0 + 0.111, 3), round(T0 + 0.2, 3),
                             round(T0 + 0.333, 3), round(T0 + 0.444, 3)]
    assert cols["frameW"] == [1280, 1280, None, 1280, None]
    assert cols["boxX"][0] == 100.0 and cols["boxW"][0] == 600.0 and cols["boxH"][0] == 600.0
    assert cols["boxScore"][0] == pytest.approx(0.91) and cols["boxX"][1] is None
    assert cols["people"] == [1, 0, None, 2, 1]
    assert cols["identityUnresolved"] == [False, False, False, True, False]
    assert cols["dropped"] == [False, False, True, False, False]
    assert cols["error"][:4] == [None] * 4 and cols["error"][4].startswith("keypoints:")
    assert len(cols["x"][0]) == KEYPOINT_COUNT and len(cols["score"][3]) == KEYPOINT_COUNT
    np.testing.assert_allclose(cols["x"][0], xy0[:, 0])
    np.testing.assert_allclose(cols["y"][3], xy1[:, 1])
    np.testing.assert_allclose(cols["score"][3], sc1)
    assert cols["x"][1] is None and cols["y"][2] is None and cols["score"][4] is None
    # The float columns are byte-stream-split zstd: small on disk.
    meta = pq.read_metadata(parts[0])
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert stream.snapshot() == {"rows": 5, "parts": 1, "late": 0, "dropped": 0, "errors": 0}


# -- the uploader -------------------------------------------------------------------


def test_the_uploader_creates_each_object_once_and_retries_transient_failures(tmp_path):
    gcs = FakeGcs()
    telemetry = Telemetry()
    uploader = Uploader(BUCKET, PREFIX, telemetry, client_factory=lambda: gcs,
                        attempts=3, backoff_s=0.0, sleep=lambda s: None)
    uploader.start()
    one = tmp_path / "a.jsonl.gz"
    one.write_bytes(b"one")
    two = tmp_path / "b.jsonl.gz"
    two.write_bytes(b"two")
    gcs.fail["%s/onsets/b.jsonl.gz" % PREFIX] = 2
    three = tmp_path / "c.jsonl.gz"
    three.write_bytes(b"three")
    gcs.fail["%s/onsets/c.jsonl.gz" % PREFIX] = 5
    gcs.objects[f"{PREFIX}/onsets/d.jsonl.gz"] = (b"already", "application/gzip")
    four = tmp_path / "d.jsonl.gz"
    four.write_bytes(b"four")
    for path in (one, two, three, four):
        uploader.put(f"onsets/{path.name}", path, "application/gzip")
    assert uploader.close(timeout_s=5.0) is True
    assert gcs.objects[f"{PREFIX}/onsets/a.jsonl.gz"] == (b"one", "application/gzip")
    assert gcs.objects[f"{PREFIX}/onsets/b.jsonl.gz"][0] == b"two"
    assert f"{PREFIX}/onsets/c.jsonl.gz" not in gcs.objects
    assert gcs.objects[f"{PREFIX}/onsets/d.jsonl.gz"][0] == b"already", "never overwritten"
    assert all(match == 0 for _, match in gcs.calls), "every create is ifGenerationMatch=0"
    assert not one.exists() and not two.exists() and not four.exists()
    assert three.exists(), "a part that never left stays for the summary to name"
    snap = uploader.snapshot()
    assert snap["uploaded"] == 2 and snap["existed"] == 1
    assert snap["failed"] == ["onsets/c.jsonl.gz"] and snap["bytes"] == 6
    counters = telemetry.snapshot()["counters"]
    assert counters["recordPartsUploaded"] == 2 and counters["recordPartsExisted"] == 1
    assert counters["recordPartsFailed"] == 1


# -- the record ---------------------------------------------------------------------


def test_the_record_routes_every_stream_to_parts_under_the_prefix(tmp_path):
    gcs = FakeGcs()
    clock = Clock()
    telemetry = Telemetry()
    record = Record(tmp_path / "rec", BUCKET, PREFIX, "s1", part_s=30, telemetry=telemetry,
                    clock=clock, client_factory=lambda: gcs,
                    provenance={"imageVersion": "v0.6.0", "imageCommit": "abc", "slot": "tee-0"})
    record.hello(hello={"fps": 30, "views": ["body", "face"]},
                 ready={"version": "2026.09.15-1", "vocal": {"x": 1}})
    wait_for(lambda: f"{PREFIX}/hello.json" in gcs.objects)
    hello = json.loads(gcs.objects[f"{PREFIX}/hello.json"][0])
    assert hello["sessionId"] == "s1" and hello["bucket"] == BUCKET and hello["partSeconds"] == 30
    assert hello["producer"]["imageVersion"] == "v0.6.0"
    assert hello["keypoints"]["count"] == 308 and hello["keypoints"]["body"]["0"] == "nose"
    assert hello["keypoints"]["face"] == [63, 307] and hello["keypoints"]["leftHand"] == [21, 41]
    assert hello["streams"]["poses"] == {"format": "parquet", "view": "body"}
    assert hello["streams"]["vocal"] == {"format": "jsonl.gz"}
    assert hello["ready"]["vocal"] == {"x": 1} and hello["hello"]["views"] == ["body", "face"]

    # What the session routes: analysis-bound messages of the kept kinds,
    # the analysis's records, the vocal rows, both views' keypoints.
    record.outbound({"kind": "frame", "frame": 1, "atS": 0.033, "keypoints": {}, "fast": None})
    record.outbound({"kind": "audio", "atS": 0.1, "rms": -30.0})
    record.outbound({"kind": "segment", "id": 1, "error": "no-audio"})
    record.outbound({"kind": "pose", "frame": 1})  # not a kept kind
    record.append("onsets", {"atS": 1.5})
    record.append("events", {"startS": 1.5, "releaseS": 2.0})
    record.append("payloads", {"kind": "clench", "atS": 2.0})
    record.append("posts", {"atS": 2.0, "ratePerMin": 12.0})
    record.append("vocal", {"kind": "decision", "atS": 2.0, "verdict": "vocal"})
    assert record.append("poses", {"frame": 1}) is False, "keypoint streams take steps"
    assert record.append("nope", {}) is False
    xy, sc = keypoints(3)
    body = ViewState()
    body.frame_size = (1280, 720)
    listen = record.keypoint_listener("body", body)
    listen(0, 0.0, xy, sc, (1.0, 2.0, 3.0, 4.0), 0.5, 1, False)
    face = ViewState()
    face.frame_size = (720, 1280)
    record.keypoint_listener("face", face)(0, 0.0, None, None, None, None, 0, False)
    # A worker's gap rows; a plain missing row is not one (the listener had it).
    assert record.gap("body", {"frame": 3, "atS": 0.1, "keypoints": None, "dropped": True,
                               "wallS": T0 + 0.1})
    assert record.gap("face", {"frame": 3, "atS": 0.1, "keypoints": None, "error": True})
    assert record.gap("body", {"frame": 4, "atS": 0.133, "keypoints": None}) is False

    class Source:
        def snapshot(self):
            return {"counters": {"framesIn": 3}, "gauges": {"streamS": 0.1}, "stages": {}}

    record.telemetry_from(Source())
    wait_for(lambda: record.streams["telemetry"].snapshot()["rows"] >= 1, timeout_s=3.0)

    clock.now = T0 + 31.0
    result = record.close({"counters": {"framesIn": 3}})
    assert result["drained"] is True
    names = sorted(gcs.objects)
    part = "part-20260915T051230Z"
    for stream in ("frames", "audio", "segments", "onsets", "events", "payloads", "posts",
                   "vocal", "telemetry"):
        assert f"{PREFIX}/{stream}/{part}.jsonl.gz" in names, stream
    assert f"{PREFIX}/poses/{part}.parquet" in names and f"{PREFIX}/faces/{part}.parquet" in names
    assert f"{PREFIX}/summary.json" in names
    assert rows_of(gcs, f"{PREFIX}/frames/{part}.jsonl.gz") == [
        {"wallS": T0, "frame": 1, "atS": 0.033, "keypoints": {}, "fast": None}]
    assert rows_of(gcs, f"{PREFIX}/segments/{part}.jsonl.gz")[0]["error"] == "no-audio"
    assert rows_of(gcs, f"{PREFIX}/vocal/{part}.jsonl.gz")[0]["verdict"] == "vocal"
    assert rows_of(gcs, f"{PREFIX}/telemetry/{part}.jsonl.gz")[0]["counters"] == {"framesIn": 3}
    poses = read_parquet(gcs, f"{PREFIX}/poses/{part}.parquet")
    assert poses["frame"] == [0, 3] and poses["frameW"] == [1280, None]
    assert poses["dropped"] == [False, True] and len(poses["x"][0]) == 308 and poses["x"][1] is None
    assert gcs.objects[f"{PREFIX}/poses/{part}.parquet"][1] == "application/vnd.apache.parquet"
    faces = read_parquet(gcs, f"{PREFIX}/faces/{part}.parquet")
    assert faces["view"] == ["face", "face"] and faces["frameW"] == [720, None]
    assert faces["error"] == [None, "error"] and faces["people"] == [0, None]
    summary = json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])
    assert summary["counters"] == {"framesIn": 3}
    assert summary["record"]["streams"]["poses"]["rows"] == 2
    assert summary["record"]["streams"]["frames"]["rows"] == 1
    assert summary["record"]["endedWallS"] == T0 + 31.0
    assert summary["record"]["files"] == ["hello.json"]
    # Idempotent: the second close is the first's answer, nothing new lands.
    count = len(gcs.objects)
    assert record.close({"again": True})["drained"] is True and len(gcs.objects) == count
    assert record.append("onsets", {"atS": 9.0}) is False
    assert telemetry.snapshot()["counters"]["recordPartsUploaded"] == 13
    assert not any(p.is_file() for p in (tmp_path / "rec").rglob("*")), "uploaded parts are gone"


def test_a_capture_with_a_record_keeps_no_flat_files_and_routes_to_it(tmp_path):
    gcs = FakeGcs()
    clock = Clock()
    record = Record(tmp_path / "rec", BUCKET, PREFIX, "s1", clock=clock, client_factory=lambda: gcs)
    capture = Capture(tmp_path, record=record)
    assert not list(tmp_path.glob("*.jsonl")), "no flat files beside a record"
    capture.pose({"frame": 0, "atS": 0.0, "keypoints": {"nose": [1, 2, 0.9]}, "wallS": T0})
    capture.pose({"frame": 3, "atS": 0.1, "keypoints": None, "dropped": True, "wallS": T0})
    capture.face_pose({"frame": 3, "atS": 0.1, "keypoints": None, "error": True, "wallS": T0})
    capture.onset(1.2345)
    capture.event(1.0, 2.0)
    capture.payload({"kind": "x"})
    capture.post({"atS": 1.0})
    capture.vocal({"kind": "activation"})
    capture.outbound({"kind": "audio", "atS": 0.5})
    summary = {"counters": {}}
    capture.summary(summary)
    capture.close()
    assert summary["record"]["drained"] is True
    streams = summary["record"]["streams"]
    assert streams["poses"]["rows"] == 1, "the posed row came through the listener, not here"
    assert streams["faces"]["rows"] == 1 and streams["onsets"]["rows"] == 1
    assert streams["events"]["rows"] == 1 and streams["payloads"]["rows"] == 1
    assert streams["posts"]["rows"] == 1 and streams["vocal"]["rows"] == 1
    assert streams["audio"]["rows"] == 1
    assert f"{PREFIX}/summary.json" in gcs.objects
    assert not (tmp_path / "summary.json").exists()
    # Without a record the flat capture is what it always was.
    flat = Capture(tmp_path / "flat")
    flat.pose({"frame": 0, "atS": 0.0, "keypoints": None})
    flat.vocal({"kind": "activation"})
    flat.outbound({"kind": "audio"})
    flat.onset(1.0)
    flat.summary({"a": 1})
    flat.close()
    assert (tmp_path / "flat" / "poses.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "flat" / "onsets.jsonl").read_text() == '{"atS": 1.0}\n'
    assert json.loads((tmp_path / "flat" / "summary.json").read_text()) == {"a": 1}
    assert not (tmp_path / "flat" / "vocal.jsonl").exists()


# -- the pose's tap fans out --------------------------------------------------------


def test_the_full_result_reaches_the_overlay_and_the_record_alike():
    state = ViewState()
    assert state.taps() == ()
    heard: list[str] = []
    state.on_full = lambda *args: heard.append("overlay")

    def bad(*args):
        heard.append("bad")
        raise RuntimeError("boom")

    state.add_listener(bad)
    state.add_listener(lambda *args: heard.append("record"))
    state.add_listener(bad)  # once
    assert len(state.taps()) == 3
    telemetry = Telemetry()
    share_full_result(state.taps(), telemetry, 0, 0.0, None, None, None, None, 0, False)
    assert heard == ["overlay", "bad", "record"], "a raising listener does not stop the next"
    assert telemetry.snapshot()["counters"]["poseListenerErrors"] == 1
    state.remove_listener(bad)
    state.on_full = None
    heard.clear()
    share_full_result(state.taps(), telemetry, 0, 0.0, None, None, None, None, 0, False)
    assert heard == ["record"]
    # One callable still works, as the overlay tests always passed it.
    share_full_result(lambda *args: heard.append("one"), telemetry,
                      0, 0.0, None, None, None, None, 0, False)
    assert heard[-1] == "one"
    share_full_result(None, telemetry, 0, 0.0, None, None, None, None, 0, False)


# -- the session ----------------------------------------------------------------------


def test_a_leased_session_records_both_views_and_every_stream(monkeypatch, tmp_path):
    from test_session_views import (BODY_URL, FACE_URL, HEIGHT, WIDTH, FakeLink,  # noqa: PLC0415
                                    StubDecoder, StubPose, session_args)

    class FullPose(StubPose):
        """The stub pose sharing a full 308-point result like the real one."""

        def step(self, rgb, index, at_s, view=None):
            state = self.view(view)
            state.frame_size = (int(rgb.shape[1]), int(rgb.shape[0]))
            xy, sc = keypoints(index)
            share_full_result(state.taps(), None, index, at_s, xy, sc,
                              (1.0, 2.0, 11.0, 12.0), 0.8, 1, False)
            return super().step(rgb, index, at_s, view=view)

    gcs = FakeGcs()
    pose = FullPose()
    links = []

    def open_link(path, on_message, on_error, **fields):
        link = FakeLink(fields)
        link.connected = True
        link.ready = {"protocol": 3, "version": "2026.09.15-1", "vocal": {"k": 1}}
        link.on_message = on_message
        links.append(link)
        return link

    monkeypatch.setattr(producer, "acquire_gpu_pose", lambda args, telemetry: pose)
    monkeypatch.setattr(producer, "Decoder", StubDecoder)
    monkeypatch.setattr(producer, "open_link", open_link)
    monkeypatch.setattr(record_module.Uploader, "__init__", _uploader_init_with(gcs))
    monkeypatch.setenv("TEE_IMAGE_VERSION", "v0.6.0")
    monkeypatch.setenv("SLOT_NAME", "tee-slot-0")
    StubDecoder.made = []
    StubDecoder.scripts = {BODY_URL: [(0, 0.0), (5, 5 / 30), (10, 10 / 30)],
                           FACE_URL: [(0, 0.0), (10, 10 / 30)]}
    StubDecoder.epochs = {BODY_URL: 1_000.0, FACE_URL: 1_002.0}
    args = session_args(tmp_path, record={"bucket": BUCKET, "prefix": PREFIX,
                                          "partSeconds": 30, "sessionId": "sess-1"})
    telemetry = Telemetry()
    session = producer.Session(args, telemetry)
    assert session.record is not None
    link = links[0]
    # Everything the analysis heard as a `vocal` row goes to the record;
    # a reading and an onset too.
    link.on_message({"kind": "vocal", "row": {"kind": "decision", "atS": 0.1}})
    link.on_message({"kind": "post", "body": {"atS": 0.5, "ratePerMin": 1.0}})
    link.on_message({"kind": "onset", "atS": 0.4})
    session.run()
    summary = session.summary
    assert summary["record"]["drained"] is True
    streams = summary["record"]["streams"]
    posed_body = sorted(i for v, i, _ in pose.steps if v == "body")
    posed_face = sorted(i for v, i, _ in pose.steps if v == "face")
    assert streams["poses"]["rows"] == len(posed_body) >= 1
    assert streams["faces"]["rows"] == len(posed_face) >= 1
    assert streams["vocal"]["rows"] == 1 and streams["posts"]["rows"] == 1
    assert streams["onsets"]["rows"] == 1
    frames = [m for m in link.sent if m["kind"] == "frame"]
    assert streams["frames"]["rows"] == len(frames) >= 1
    objects = sorted(gcs.objects)
    assert f"{PREFIX}/hello.json" in objects and f"{PREFIX}/summary.json" in objects
    hello = json.loads(gcs.objects[f"{PREFIX}/hello.json"][0])
    assert hello["sessionId"] == "sess-1" and hello["producer"]["imageVersion"] == "v0.6.0"
    assert hello["producer"]["slot"] == "tee-slot-0"
    assert hello["hello"]["views"] == ["body", "face"] and "run" not in hello["hello"]
    assert hello["ready"]["version"] == "2026.09.15-1"
    assert hello["sources"]["poses"]["stream"] == BODY_URL
    assert hello["sources"]["faces"]["stream"] == FACE_URL
    assert hello["streams"]["poses"] == {"format": "parquet", "view": "body"}
    poses = [name for name in objects if name.startswith(f"{PREFIX}/poses/")]
    faces = [name for name in objects if name.startswith(f"{PREFIX}/faces/")]
    assert poses and faces
    body_table = pa.concat_tables(
        [pq.read_table(pa.BufferReader(gcs.objects[name][0])) for name in poses]).to_pydict()
    assert sorted(body_table["frame"]) == posed_body
    assert all(len(x) == 308 for x in body_table["x"])
    assert body_table["frameW"] == [WIDTH] * len(posed_body)
    assert body_table["frameH"] == [HEIGHT] * len(posed_body)
    assert body_table["boxW"][0] == 10.0 and body_table["people"][0] == 1
    face_table = pa.concat_tables(
        [pq.read_table(pa.BufferReader(gcs.objects[name][0])) for name in faces]).to_pydict()
    assert sorted(face_table["frame"]) == posed_face and face_table["view"][0] == "face"
    # The frames the record kept are the ones the analysis got, whole.
    frame_rows = []
    for name in objects:
        if name.startswith(f"{PREFIX}/frames/"):
            frame_rows += rows_of(gcs, name)
    assert [r["frame"] for r in frame_rows] == [m["frame"] for m in frames]
    assert all("wallS" in r for r in frame_rows)
    # The overlay's tap was never installed (no publish URL) and the
    # record's listeners are gone with the session.
    assert pose.view("body").taps() == () and pose.view("face").taps() == ()
    # No flat capture beside it.
    assert not (tmp_path / "poses.jsonl").exists()
    assert (summary["record"]["bucket"], summary["record"]["prefix"]) == (BUCKET, PREFIX)


def _uploader_init_with(gcs):
    original = Uploader.__init__

    def __init__(self, bucket, prefix, telemetry=None, client_factory=None, **kwargs):
        original(self, bucket, prefix, telemetry, client_factory=lambda: gcs, **kwargs)

    return __init__


def test_a_session_without_a_record_spec_runs_the_flat_capture(tmp_path):
    args = argparse.Namespace(sink_dir=str(tmp_path), record=None)
    assert producer.open_record(args, Telemetry()) is None
    args.record = {"prefix": PREFIX}  # no bucket: nothing to write to
    assert producer.open_record(args, Telemetry()) is None


def test_sigterm_flushes_the_running_sessions_record(monkeypatch):
    flushed = []

    class FakeSession:
        record = object()

        def flush_record(self, timeout_s):
            flushed.append(timeout_s)

    current = {"session": FakeSession()}
    installed = {}
    monkeypatch.setattr(producer.signal, "signal",
                        lambda signum, handler: installed.update({signum: handler}))
    producer.install_sigterm_flush(current, flush_s=3.0)
    handler = installed[signal.SIGTERM]
    with pytest.raises(SystemExit) as exit_:
        handler(signal.SIGTERM, None)
    assert exit_.value.code == 0 and flushed == [3.0]
    # Without a session (or a record) the exit is immediate.
    current["session"] = None
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    assert flushed == [3.0]
