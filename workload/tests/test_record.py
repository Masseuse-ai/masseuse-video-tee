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
    assert record.hello() is True
    assert record.hello() is False, "hello.json is the record's, written once"
    wait_for(lambda: f"{PREFIX}/hello.json" in gcs.objects)
    hello = json.loads(gcs.objects[f"{PREFIX}/hello.json"][0])
    assert hello["sessionId"] == "s1" and hello["bucket"] == BUCKET and hello["partSeconds"] == 30
    assert hello["startedWallS"] == T0
    assert hello["producer"]["imageVersion"] == "v0.6.0"
    assert hello["keypoints"]["count"] == 308 and hello["keypoints"]["body"]["0"] == "nose"
    assert hello["keypoints"]["rightHand"] == [21, 41] and hello["keypoints"]["leftHand"] == [42, 62]
    assert hello["keypoints"]["armAndNeck"] == [63, 69] and hello["keypoints"]["face"] == [70, 307]
    assert hello["keypoints"]["handOrderVerified"] is True
    names = hello["keypoints"]["names"]
    assert len(names) == 308 and names[:2] == ["nose", "left_eye"]
    assert names[21] == "right_thumb4" and names[41] == "right_wrist" and names[62] == "left_wrist"
    assert names[69] == "neck" and names[178] == "tip_of_nose" and names[272] == "l_center_of_iris"
    assert hello["streams"]["poses"] == {"format": "parquet", "view": "body"}
    assert hello["streams"]["vocal"] == {"format": "jsonl.gz"}
    assert "hello" not in hello and "ready" not in hello, "a run's terms are the run's file"
    # A production run: its own hello under runs/<start>/, the start being
    # the record's clock in UTC.
    run = record.begin_run("estim-s1-20260915T051230Z-a1")
    assert run == "20260915T051230Z"
    record.run_hello(run, hello={"fps": 30, "views": ["body", "face"]},
                     ready={"version": "2026.09.15-1", "vocal": {"x": 1}})
    wait_for(lambda: f"{PREFIX}/runs/{run}/hello.json" in gcs.objects)
    run_hello = json.loads(gcs.objects[f"{PREFIX}/runs/{run}/hello.json"][0])
    assert run_hello["runId"] == run and run_hello["run"] == "estim-s1-20260915T051230Z-a1"
    assert run_hello["sessionId"] == "s1" and run_hello["startedWallS"] == T0
    assert run_hello["ready"]["vocal"] == {"x": 1} and run_hello["hello"]["views"] == ["body", "face"]
    assert run_hello["producer"]["slot"] == "tee-0"

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
    record.append("log", {"source": "producer", "text": "analysis send failed: BrokenPipeError(32)"})
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
            # The shape telemetry.Telemetry.snapshot() has: the timings are
            # `stagesMs`, per stage its percentiles and count.
            return {"atWall": 1.0, "uptimeS": 2.0, "bootMs": {},
                    "counters": {"framesIn": 3}, "gauges": {"streamS": 0.1},
                    "stagesMs": {"audioPitch": {"p50": 4.2, "p95": 6.1, "n": 12}}}

    record.telemetry_from(Source())
    wait_for(lambda: record.streams["telemetry"].snapshot()["rows"] >= 1, timeout_s=3.0)

    # The run ends: its summary lands under runs/<start>/, the parts stay
    # open (nothing but hello files in the bucket yet), the record's
    # counts come back for the run's own printed summary.
    clock.now = T0 + 20.0
    state = record.end_run(run, {"counters": {"framesIn": 3}, "bootMs": {"weights": 1.0}})
    assert state["runId"] == run and state["closed"] is False
    assert state["streams"]["poses"]["rows"] == 2 and state["streams"]["poses"]["parts"] == 0
    wait_for(lambda: f"{PREFIX}/runs/{run}/summary.json" in gcs.objects)
    run_summary = json.loads(gcs.objects[f"{PREFIX}/runs/{run}/summary.json"][0])
    assert run_summary["counters"] == {"framesIn": 3} and run_summary["bootMs"] == {"weights": 1.0}
    assert run_summary["runId"] == run
    assert run_summary["startedWallS"] == T0 and run_summary["endedWallS"] == T0 + 20.0
    assert run_summary["record"]["streams"]["frames"]["rows"] == 1
    assert not any(name.startswith(f"{PREFIX}/frames/") for name in gcs.objects), \
        "the window is still open: its part waits for the next run or the window's end"
    assert record.append("onsets", {"atS": 9.0}) is True, "the record is still open"

    clock.now = T0 + 31.0
    result = record.close({"ended": "stop"})
    assert result["drained"] is True and result["closed"] is True
    names = sorted(gcs.objects)
    part = "part-20260915T051230Z"
    for stream in ("frames", "audio", "segments", "onsets", "events", "payloads", "posts",
                   "vocal", "telemetry", "log"):
        assert f"{PREFIX}/{stream}/{part}.jsonl.gz" in names, stream
    assert f"{PREFIX}/poses/{part}.parquet" in names and f"{PREFIX}/faces/{part}.parquet" in names
    assert f"{PREFIX}/summary.json" in names
    log_rows = rows_of(gcs, f"{PREFIX}/log/{part}.jsonl.gz")
    assert [(r["source"], r["text"]) for r in log_rows] == [
        ("producer", "analysis send failed: BrokenPipeError(32)")]
    assert rows_of(gcs, f"{PREFIX}/onsets/{part}.jsonl.gz") == [
        {"wallS": T0, "atS": 1.5}, {"wallS": T0 + 20.0, "atS": 9.0}], \
        "a row after the run's end rides in the same part"
    assert rows_of(gcs, f"{PREFIX}/frames/{part}.jsonl.gz") == [
        {"wallS": T0, "frame": 1, "atS": 0.033, "keypoints": {}, "fast": None}]
    assert rows_of(gcs, f"{PREFIX}/segments/{part}.jsonl.gz")[0]["error"] == "no-audio"
    assert rows_of(gcs, f"{PREFIX}/vocal/{part}.jsonl.gz")[0]["verdict"] == "vocal"
    telemetry_row = rows_of(gcs, f"{PREFIX}/telemetry/{part}.jsonl.gz")[0]
    assert telemetry_row["counters"] == {"framesIn": 3}
    assert telemetry_row["gauges"] == {"streamS": 0.1}
    # The stage timings under the snapshot's own name; a row has exactly
    # the three blocks and the wall clock, nothing of the summary's.
    assert telemetry_row["stagesMs"] == {"audioPitch": {"p50": 4.2, "p95": 6.1, "n": 12}}
    assert set(telemetry_row) == {"wallS", "counters", "gauges", "stagesMs"}
    poses = read_parquet(gcs, f"{PREFIX}/poses/{part}.parquet")
    assert poses["frame"] == [0, 3] and poses["frameW"] == [1280, None]
    assert poses["dropped"] == [False, True] and len(poses["x"][0]) == 308 and poses["x"][1] is None
    assert gcs.objects[f"{PREFIX}/poses/{part}.parquet"][1] == "application/vnd.apache.parquet"
    faces = read_parquet(gcs, f"{PREFIX}/faces/{part}.parquet")
    assert faces["view"] == ["face", "face"] and faces["frameW"] == [720, None]
    assert faces["error"] == [None, "error"] and faces["people"] == [0, None]
    summary = json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])
    assert summary["ended"] == "stop" and "counters" not in summary, \
        "the record's summary is the record's: the run's numbers are the run's file"
    assert summary["record"]["streams"]["poses"]["rows"] == 2
    assert summary["record"]["streams"]["frames"]["rows"] == 1
    assert summary["record"]["streams"]["onsets"]["rows"] == 2
    assert summary["record"]["endedWallS"] == T0 + 31.0
    assert summary["record"]["startedWallS"] == T0
    assert summary["record"]["files"] == ["hello.json", f"runs/{run}/hello.json",
                                          f"runs/{run}/summary.json"]
    assert summary["record"]["runs"] == [{"id": run, "run": "estim-s1-20260915T051230Z-a1",
                                          "startedWallS": T0, "endedWallS": T0 + 20.0}]
    # Idempotent: the second close is the first's answer, nothing new lands.
    count = len(gcs.objects)
    assert record.close({"again": True})["drained"] is True and len(gcs.objects) == count
    assert record.append("onsets", {"atS": 9.0}) is False
    assert record.hello() is False
    assert telemetry.snapshot()["counters"]["recordPartsUploaded"] == 16  # the log stream is one more part
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
    # The run ended in the record; the record itself is the lease's and
    # stays open (the keeper closes it), so nothing is drained yet.
    run = capture.run_id
    assert run == "20260915T051230Z", "a capture without a run id begins one"
    assert summary["record"]["runId"] == run and summary["record"]["closed"] is False
    streams = summary["record"]["streams"]
    assert streams["poses"]["rows"] == 1, "the posed row came through the listener, not here"
    assert streams["faces"]["rows"] == 1 and streams["onsets"]["rows"] == 1
    assert streams["events"]["rows"] == 1 and streams["payloads"]["rows"] == 1
    assert streams["posts"]["rows"] == 1 and streams["vocal"]["rows"] == 1
    assert streams["audio"]["rows"] == 1
    assert record.closed is False
    wait_for(lambda: f"{PREFIX}/runs/{run}/summary.json" in gcs.objects)
    assert f"{PREFIX}/summary.json" not in gcs.objects
    assert record.close({"ended": "stop"})["drained"] is True
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
    # Without a keeper (no serving slot) the run's record is its own and
    # closes with it, drained.
    assert summary["record"]["drained"] is True and summary["record"]["closed"] is True
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
    run = session.run_id
    assert f"{PREFIX}/hello.json" in objects and f"{PREFIX}/summary.json" in objects
    assert f"{PREFIX}/runs/{run}/hello.json" in objects
    assert f"{PREFIX}/runs/{run}/summary.json" in objects
    hello = json.loads(gcs.objects[f"{PREFIX}/hello.json"][0])
    assert hello["sessionId"] == "sess-1" and hello["producer"]["imageVersion"] == "v0.6.0"
    assert hello["producer"]["slot"] == "tee-slot-0"
    assert hello["streams"]["poses"] == {"format": "parquet", "view": "body"}
    run_hello = json.loads(gcs.objects[f"{PREFIX}/runs/{run}/hello.json"][0])
    assert run_hello["hello"]["views"] == ["body", "face"] and "run" not in run_hello["hello"]
    assert run_hello["ready"]["version"] == "2026.09.15-1"
    assert run_hello["sources"]["poses"]["stream"] == BODY_URL
    assert run_hello["sources"]["faces"]["stream"] == FACE_URL
    run_summary = json.loads(gcs.objects[f"{PREFIX}/runs/{run}/summary.json"][0])
    assert run_summary["runId"] == run and "bootMs" in run_summary
    assert run_summary["record"]["streams"]["poses"]["rows"] == len(posed_body)
    record_summary = json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])
    assert record_summary["ended"] == "run" and len(record_summary["record"]["runs"]) == 1
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


def test_a_sessions_production_runs_share_the_leases_record(monkeypatch, tmp_path):
    """The trainer opens a new /produce when the camera changes (the
    phone's picture, then a fixed camera: 2026-09-15's first real session
    had two runs in one lease). Under a keeper both runs write the same
    record: one hello.json, one part per window even where the runs meet,
    a runs/<start>/ pair each, and one summary.json when the lease ends -
    where before the second run's files were refused as already there."""
    from test_session_views import (BODY_URL, FakeLink, StubDecoder,  # noqa: PLC0415
                                    StubPose, session_args)

    class FullPose(StubPose):
        def step(self, rgb, index, at_s, view=None):
            state = self.view(view)
            state.frame_size = (int(rgb.shape[1]), int(rgb.shape[0]))
            xy, sc = keypoints(index)
            share_full_result(state.taps(), None, index, at_s, xy, sc,
                              (1.0, 2.0, 11.0, 12.0), 0.8, 1, False)
            return super().step(rgb, index, at_s, view=view)

    gcs = FakeGcs()
    pose = FullPose()
    clock = Clock()

    def open_link(path, on_message, on_error, **fields):
        link = FakeLink(fields)
        link.connected = True
        link.ready = {"protocol": 3, "version": "2026.09.15-1"}
        link.on_message = on_message
        return link

    monkeypatch.setattr(producer, "acquire_gpu_pose", lambda args, telemetry: pose)
    monkeypatch.setattr(producer, "Decoder", StubDecoder)
    monkeypatch.setattr(producer, "open_link", open_link)
    monkeypatch.setattr(record_module.Uploader, "__init__", _uploader_init_with(gcs))
    monkeypatch.setattr(record_module.time, "time", clock)
    StubDecoder.made = []
    StubDecoder.scripts = {BODY_URL: [(0, 0.0), (5, 5 / 30)]}
    StubDecoder.epochs = {BODY_URL: 1_000.0}
    spec = {"bucket": BUCKET, "prefix": PREFIX, "partSeconds": 30, "sessionId": "sess-1"}
    keeper = record_module.RecordKeeper(log=lambda *a, **k: None)
    telemetry = Telemetry()

    def session(run: str):
        return producer.Session(
            session_args(tmp_path, record=spec, run=run, face_stream="", audio_stream=""),
            telemetry, records=keeper)

    first = session("estim-sess-1-a1")
    first.run()
    assert first.summary["record"]["closed"] is False, "the record is the lease's"
    record = first.record
    assert keeper.open_records() == [record]
    assert first.run_id == "20260915T051230Z"

    clock.now = T0 + 12.0  # the same window: the phone's picture gave way to the camera
    second = session("estim-sess-1-a2")
    assert second.record is record, "the same open record"
    assert second.run_id == "20260915T051242Z"
    second.run()
    assert second.summary["record"]["closed"] is False

    hello_files = [name for name in gcs.objects if name.endswith("/hello.json")]
    assert sorted(hello_files) == [f"{PREFIX}/hello.json",
                                   f"{PREFIX}/runs/20260915T051230Z/hello.json",
                                   f"{PREFIX}/runs/20260915T051242Z/hello.json"]
    assert f"{PREFIX}/summary.json" not in gcs.objects, "not until the lease ends"
    assert not any(name.startswith(f"{PREFIX}/poses/") for name in gcs.objects), \
        "the window both runs wrote is one open part"

    # The lease ends (/stop): the keeper closes the record, and the one
    # part of the shared window carries both runs' rows.
    clock.now = T0 + 40.0
    results = keeper.close_all("stop")
    assert len(results) == 1 and results[0]["drained"] is True
    assert keeper.open_records() == []
    part = f"{PREFIX}/poses/part-20260915T051230Z.parquet"
    poses = [name for name in gcs.objects if name.startswith(f"{PREFIX}/poses/")]
    assert poses == [part], poses
    rows = read_parquet(gcs, part)
    body_steps = [i for v, i, _ in pose.steps if v == "body"]
    assert len(rows["frame"]) == len(body_steps) and len(body_steps) >= 2
    assert min(rows["wallS"]) == T0 and max(rows["wallS"]) == T0 + 12.0
    summary = json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])
    assert summary["ended"] == "stop"
    assert [r["id"] for r in summary["record"]["runs"]] == ["20260915T051230Z", "20260915T051242Z"]
    assert [r["run"] for r in summary["record"]["runs"]] == ["estim-sess-1-a1", "estim-sess-1-a2"]
    assert all(r["endedWallS"] is not None for r in summary["record"]["runs"])
    assert summary["record"]["upload"]["existed"] == 0, "nothing was refused as already there"
    assert telemetry.snapshot()["counters"].get("recordPartsExisted", 0) == 0
    # A third run after the lease ended would open a fresh record under
    # the prefix (a re-lease of the same session); the closed one is gone
    # from the keeper.
    reopened = keeper.open(BUCKET, PREFIX, lambda: Record(
        tmp_path / "rec2", BUCKET, PREFIX, "sess-1", clock=clock, client_factory=lambda: gcs))
    assert reopened is not record and keeper.open_records() == [reopened]
    reopened.close()


def test_the_keeper_holds_one_record_per_prefix_and_closes_the_rest_on_a_new_lease(tmp_path):
    gcs = FakeGcs()
    clock = Clock()
    said = []
    keeper = record_module.RecordKeeper(log=lambda text, **k: said.append(text))
    other = f"{ACCOUNT}/estim_sessions/019966a0-0000-7000-8000-000000000003/enclave"

    def factory(prefix, name):
        return lambda: Record(tmp_path / name, BUCKET, prefix, name, clock=clock,
                              client_factory=lambda: gcs)

    a = keeper.open(BUCKET, PREFIX, factory(PREFIX, "a"))
    assert keeper.open(BUCKET, PREFIX + "/", factory(PREFIX, "a2")) is a, "a trailing slash is the same prefix"
    b = keeper.open(BUCKET, other, factory(other, "b"))
    assert b is not a and keeper.get(BUCKET, other) is b
    assert set(map(id, keeper.open_records())) == {id(a), id(b)}
    a.append("onsets", {"atS": 1.0})
    # A lease for the second session: the first's record closes, the
    # second's is kept; then the same again changes nothing.
    thread = keeper.close_in_background("lease", keep=(BUCKET, other))
    thread.join(5.0)
    assert a.closed is True and b.closed is False
    assert keeper.open_records() == [b]
    assert f"{PREFIX}/summary.json" in gcs.objects
    assert json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])["ended"] == "lease"
    assert f"{PREFIX}/onsets/part-20260915T051230Z.jsonl.gz" in gcs.objects
    assert keeper.close_all("lease", keep=(BUCKET, other)) == []
    assert any("closed" in line and "1 rows in 1 parts" in line for line in said)
    # The rest: close one by name, an unknown is None, close_all is idempotent.
    assert keeper.close(BUCKET, "nobody/estim_sessions/x/enclave", "stop") is None
    assert keeper.close(BUCKET, other, "stop")["closed"] is True
    assert keeper.close(BUCKET, other, "stop") is None
    assert keeper.close_all("exit") == [] and keeper.open_records() == []
    assert json.loads(gcs.objects[f"{other}/summary.json"][0])["ended"] == "stop"


def test_the_exits_close_the_leases_record_first(tmp_path):
    gcs = FakeGcs()
    clock = Clock()
    keeper = record_module.RecordKeeper(log=lambda *a, **k: None)
    record = keeper.open(BUCKET, PREFIX, lambda: Record(
        tmp_path / "rec", BUCKET, PREFIX, "s", clock=clock, client_factory=lambda: gcs))
    record.append("posts", {"atS": 1.0})
    exits = []
    exit_ = producer.closing_exit(keeper, exit_impl=exits.append, close_s=5.0)
    exit_(0)
    assert exits == [0]
    assert record.closed is True and f"{PREFIX}/summary.json" in gcs.objects
    assert json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])["ended"] == "exit"
    assert f"{PREFIX}/posts/part-20260915T051230Z.jsonl.gz" in gcs.objects
    # Nothing open: the exit is just the exit.
    exit_(3)
    assert exits == [0, 3]
    # The teardown and the idle exit take it as their exit_impl.
    teardown = producer.Teardown(threading.Lock(), drain_s=0.0, exit_impl=exit_,
                                 sleep=lambda s: None)
    keeper.open(BUCKET, PREFIX, lambda: Record(
        tmp_path / "rec2", BUCKET, PREFIX, "s", clock=clock, client_factory=lambda: gcs))
    code, body = teardown.request("now")
    assert code == 200
    wait_for(lambda: exits == [0, 3, 0])
    assert keeper.open_records() == []


def test_sigterm_flushes_the_leases_record_between_runs(monkeypatch, tmp_path):
    gcs = FakeGcs()
    clock = Clock()
    keeper = record_module.RecordKeeper(log=lambda *a, **k: None)
    record = keeper.open(BUCKET, PREFIX, lambda: Record(
        tmp_path / "rec", BUCKET, PREFIX, "s", clock=clock, client_factory=lambda: gcs))
    record.append("events", {"startS": 1.0, "releaseS": 2.0})
    installed = {}
    monkeypatch.setattr(producer.signal, "signal",
                        lambda signum, handler: installed.update({signum: handler}))
    producer.install_sigterm_flush({"session": None}, flush_s=3.0, records=keeper)
    with pytest.raises(SystemExit) as exit_:
        installed[signal.SIGTERM](signal.SIGTERM, None)
    assert exit_.value.code == 0
    assert record.closed is True
    assert json.loads(gcs.objects[f"{PREFIX}/summary.json"][0])["ended"] == "flush"
    assert f"{PREFIX}/events/part-20260915T051230Z.jsonl.gz" in gcs.objects


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
