"""A /produce session ends with its client.

The open SSE request is the session: whoever holds it is the one consuming
the readings, and when it goes away there is nobody to produce for. Locked
down here: a client that disconnects stops the session and the runner is
joined before the pump returns (so the busy lock released after it never
frees a GPU that is still working); an explicit POST /stop ends it the same
way for the caller whose disconnect may not reach the container; a session
that ends on its own still writes its summary event; a runner that will not
wind down in time is reported, not waited on forever. And the stream speaks
before the boot: a `hello` event the moment the headers are out and a
keepalive comment every few seconds while Session() waits on the GPU pose
lock, so a cold instance's minute of boot is never a silent stream.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import producer  # noqa: E402
from producer import (  # noqa: E402
    BootKeepalive, gpu_pose_state, pump_session, stop_session,
)
from telemetry import Telemetry  # noqa: E402


class FakeSession:
    def __init__(self, run_name: str = ""):
        self.stopping = threading.Event()
        self.summary = None
        self.run_name = run_name
        self.error = None


class Wire:
    """The client's end of the SSE response: records every write and, if
    asked, goes away on the Nth one the way a closed socket does."""

    def __init__(self, break_on_write: int | None = None):
        self.writes: list[bytes] = []
        self.break_on_write = break_on_write

    def write(self, data: bytes) -> None:
        if (self.break_on_write is not None
                and len(self.writes) + 1 >= self.break_on_write):
            raise BrokenPipeError("client went away")
        self.writes.append(data)

    def flush(self) -> None:
        return None

    def events(self) -> list[dict]:
        return [json.loads(chunk[len(b"data: "):].strip())
                for chunk in self.writes if chunk.startswith(b"data: ")]


def start_runner(session: FakeSession, telemetry: Telemetry,
                 frames: int | None = None, slow_stop_s: float = 0.0):
    """A stand-in for Session.run on its thread: emits a payload per
    'frame' until `frames` run out or `stopping` is set."""
    def run() -> None:
        at = 0
        while frames is None or at < frames:
            if session.stopping.is_set():
                break
            telemetry.emit("payload", {"atS": at, "calibrationReady": False})
            at += 1
            time.sleep(0.01)
        if session.stopping.is_set() and slow_stop_s:
            time.sleep(slow_stop_s)
        session.summary = {"counters": {"framesIn": at},
                           "captureDir": "/tmp/capture", "bootMs": {}}

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    return runner


def test_a_departed_client_stops_the_session_and_the_runner_is_joined():
    telemetry = Telemetry()
    session = FakeSession()
    subscription = telemetry.subscribe()
    runner = start_runner(session, telemetry)  # would run forever
    wire = Wire(break_on_write=3)
    lines: list[str] = []

    outcome = pump_session(session, runner, wire, subscription, telemetry,
                           wait_s=0.02, join_timeout_s=2.0,
                           log=lambda msg, **_: lines.append(msg))

    assert outcome == "client_left"
    assert session.stopping.is_set()
    assert not runner.is_alive(), "the runner must be joined before return"
    assert subscription not in telemetry._listeners
    assert lines and lines[0].startswith("produce: client left after")
    assert lines[0].endswith("session stopped")


def test_a_session_that_ends_on_its_own_writes_its_summary():
    telemetry = Telemetry()
    session = FakeSession(run_name="session-abc")
    subscription = telemetry.subscribe()
    runner = start_runner(session, telemetry, frames=3)
    wire = Wire()

    outcome = pump_session(session, runner, wire, subscription, telemetry,
                           wait_s=0.02, upload_result=lambda: {"ok": True})

    assert outcome == "ended"
    assert not session.stopping.is_set()
    events = wire.events()
    assert [e["atS"] for e in events if e["kind"] == "payload"] == [0, 1, 2]
    final = events[-1]
    assert final["kind"] == "summary"
    assert final["run"] == "session-abc"
    assert final["counters"] == {"framesIn": 3}
    assert final["upload"] == {"ok": True}
    assert subscription not in telemetry._listeners


def test_an_explicit_stop_ends_the_session_through_the_normal_summary():
    """The caller whose disconnect may never reach the container says
    stop out loud; the pump then ends as for a session that ran out."""
    telemetry = Telemetry()
    session = FakeSession(run_name="session-xyz")
    holder = {"session": session}
    subscription = telemetry.subscribe()
    runner = start_runner(session, telemetry)  # would run forever
    wire = Wire()
    done: dict = {}

    def pump() -> None:
        done["outcome"] = pump_session(session, runner, wire, subscription,
                                       telemetry, wait_s=0.02)

    pumping = threading.Thread(target=pump, daemon=True)
    pumping.start()
    time.sleep(0.05)

    assert stop_session(holder) == (200, {"status": "stopping",
                                          "run": "session-xyz"})
    pumping.join(timeout=3.0)
    assert done["outcome"] == "ended"
    assert wire.events()[-1]["kind"] == "summary"
    assert stop_session({"session": None}) == (
        404, {"error": "no session is running"})


def test_a_stop_given_the_slot_lock_answers_stopped_once_the_run_let_go():
    """The trainer restarts production on a camera change: its /stop is
    answered once the slot is free, so the /produce that follows lands
    first time rather than meeting the 409 of a run still winding down."""
    from producer import STOP_POLL_S, STOP_WAIT_S

    clock = {"now": 0.0}
    slept: list[float] = []

    def sleeper(lock: threading.Lock, release_at: float | None):
        def sleep(s: float) -> None:
            slept.append(s)
            clock["now"] += s
            if release_at is not None and clock["now"] >= release_at and lock.locked():
                lock.release()  # the pump letting go once the runner wound down
        return sleep

    session = FakeSession(run_name="session-xyz")
    busy = threading.Lock()
    busy.acquire()  # the /produce handler holds it for the run
    code, body = stop_session({"session": session}, busy, clock=lambda: clock["now"],
                              sleep=sleeper(busy, release_at=1.2))
    assert (code, body) == (200, {"status": "stopped", "run": "session-xyz"})
    assert session.stopping.is_set()
    assert 1.2 <= clock["now"] < 1.2 + 2 * STOP_POLL_S
    assert slept and max(slept) <= STOP_POLL_S

    # A run that will not let go within the wait is reported `stopping`,
    # the answer of old; the caller goes on as before.
    session = FakeSession(run_name="session-slow")
    slow = threading.Lock()
    slow.acquire()
    clock["now"] = 0.0
    code, body = stop_session({"session": session}, slow, clock=lambda: clock["now"],
                              sleep=sleeper(slow, release_at=None))
    assert (code, body) == (200, {"status": "stopping", "run": "session-slow"})
    assert STOP_WAIT_S <= clock["now"] < STOP_WAIT_S + STOP_POLL_S
    slow.release()

    # A slot already free answers `stopped` without a wait.
    clock["now"] = 0.0
    free = threading.Lock()
    code, body = stop_session({"session": FakeSession(run_name="r")}, free,
                              clock=lambda: clock["now"], sleep=sleeper(free, release_at=None))
    assert (code, body["status"], clock["now"]) == (200, "stopped", 0.0)


def test_a_runner_that_will_not_wind_down_is_reported_not_awaited_forever():
    telemetry = Telemetry()
    session = FakeSession()
    subscription = telemetry.subscribe()
    runner = start_runner(session, telemetry, slow_stop_s=1.5)
    wire = Wire(break_on_write=2)
    lines: list[str] = []

    started = time.monotonic()
    outcome = pump_session(session, runner, wire, subscription, telemetry,
                           wait_s=0.02, join_timeout_s=0.2,
                           log=lambda msg, **_: lines.append(msg))

    assert outcome == "client_left"
    assert time.monotonic() - started < 1.0
    assert runner.is_alive()
    assert "still winding down" in lines[0]
    runner.join(timeout=3.0)


# -- the stream speaks before the boot ---------------------------------------


def test_hello_is_the_first_event_and_keepalives_follow_until_stopped():
    wire = Wire()
    keepalive = BootKeepalive(wire, interval_s=0.02)
    keepalive.hello("booting")
    assert wire.writes[0] == b'data: {"kind": "hello", "state": "booting"}\n\n'

    keepalive.start()
    time.sleep(0.15)
    keepalive.stop()
    comments = [chunk for chunk in wire.writes if chunk == b": keepalive\n\n"]
    assert len(comments) >= 3
    # Stopped means stopped: nothing more arrives once Session() is back.
    written = len(wire.writes)
    time.sleep(0.1)
    assert len(wire.writes) == written
    assert keepalive.failed is False


def test_a_client_that_leaves_during_the_boot_is_noticed_and_the_thread_ends():
    wire = Wire(break_on_write=2)  # the hello lands, the first comment does not
    keepalive = BootKeepalive(wire, interval_s=0.02)
    keepalive.hello("booting")
    keepalive.start()
    time.sleep(0.1)
    assert keepalive.failed is True
    keepalive.stop()
    assert len(wire.writes) == 1


def test_gpu_pose_state_reads_booted_and_not_mid_boot_as_ready():
    saved = producer._GPU_POSE["pose"]
    try:
        producer._GPU_POSE["pose"] = None
        assert gpu_pose_state() == "booting"
        producer._GPU_POSE["pose"] = object()
        assert gpu_pose_state() == "ready"
        # A /warmup mid-boot holds the lock: the session will join that boot.
        with producer._GPU_POSE_LOCK:
            assert gpu_pose_state() == "booting"
        assert gpu_pose_state() == "ready"
    finally:
        producer._GPU_POSE["pose"] = saved


def test_a_session_that_ended_early_says_why_ahead_of_its_summary():
    """run() raising (the stream had no video track, say) reaches the
    client as an `error` event the trainer shows as lastError, then the
    summary as usual."""
    telemetry = Telemetry()
    session = FakeSession(run_name="session-xyz")
    subscription = telemetry.subscribe()
    runner = start_runner(session, telemetry, frames=1)
    runner.join()
    session.error = "RuntimeError: no video track on rtsp://127.0.0.1:8554/cam"
    wire = Wire()

    outcome = pump_session(session, runner, wire, subscription, telemetry,
                           wait_s=0.02, upload_result=lambda: None)

    assert outcome == "ended"
    kinds = [event["kind"] for event in wire.events()]
    assert kinds[-2:] == ["error", "summary"]
    assert wire.events()[-2]["error"] == session.error
