"""tee_mode.IdleExit: the slot ends itself when nobody holds it.

A boot nobody leases exits after boot_idle_s; a /warmup (touch) restarts
that clock; a lease, or a boot or /produce in progress, holds the slot;
once a lease has been held the shorter idle_s applies from the moment the
slot is idle; an expired lease is as idle as a cleared one; the exit is 0
through the injected exit; the producer wires it in TEE mode only, on the
configured values, and /warmup touches it.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

import producer  # noqa: E402
import tee_mode  # noqa: E402
from telemetry import Telemetry  # noqa: E402
from test_tee_mode import Slot, make_tee, server_args  # noqa: E402

pytest.importorskip("cryptography")


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeTeardown:
    def __init__(self):
        self.working = False

    def is_working(self) -> bool:
        return self.working


def make(clock: Clock, **overrides):
    teardown = FakeTeardown()
    lease = tee_mode.Lease(clock=clock)
    exits: list[int] = []
    logged: list[str] = []
    idle = tee_mode.IdleExit(
        teardown, lease, idle_s=60, boot_idle_s=300, clock=clock,
        exit_impl=exits.append, sleep=lambda s: None,
        log=lambda message, **kw: logged.append(message), **overrides)
    return idle, teardown, lease, exits, logged


def grant(lease: tee_mode.Lease, clock: Clock, ttl_s: float = 3600) -> None:
    code, _ = lease.grant({"sessionId": "sess-1", "capabilityHash": "ab" * 32,
                           "expiresAt": clock() + ttl_s})
    assert code == 200


def test_a_boot_nobody_leases_exits_after_boot_idle_s():
    clock = Clock()
    idle, _, _, _, _ = make(clock)
    clock.advance(299)
    assert idle.due() is None
    assert idle.snapshot()["exitInS"] == 1.0
    clock.advance(1)
    assert idle.due() == "no lease 300s after boot"
    assert idle.snapshot()["everLeased"] is False


def test_a_warmup_restarts_the_boot_clock():
    clock = Clock()
    idle, _, _, _, _ = make(clock)
    clock.advance(250)
    idle.touch()
    clock.advance(250)
    assert idle.due() is None, "the trainer is polling: a phone is waiting"
    clock.advance(50)
    assert idle.due() == "no lease 300s after boot"


def test_work_in_progress_holds_the_slot_and_idle_counts_from_its_end():
    clock = Clock()
    idle, teardown, _, _, _ = make(clock)
    teardown.working = True
    clock.advance(1000)
    assert idle.due() is None
    teardown.working = False
    assert idle.due() is None
    clock.advance(299)
    assert idle.due() is None
    clock.advance(1)
    assert idle.due() is not None


def test_a_lease_holds_the_slot_and_the_short_clock_runs_after_it():
    clock = Clock()
    idle, _, lease, _, _ = make(clock)
    clock.advance(200)
    grant(lease, clock, ttl_s=tee_mode.LEASE_MAX_S)
    clock.advance(5 * 3600)
    assert idle.due() is None
    assert idle.snapshot() == {"everLeased": True, "idleForS": 0.0, "exitInS": None,
                               "idleExitS": 60.0, "bootIdleS": 300.0}
    lease.clear()  # /stop or /teardown
    assert idle.due() is None
    clock.advance(59)
    assert idle.due() is None
    clock.advance(1)
    assert idle.due() == "no lease and no session for 60s"


def test_an_expired_lease_is_idle():
    clock = Clock()
    idle, _, lease, _, _ = make(clock)
    grant(lease, clock, ttl_s=100)
    clock.advance(50)
    assert idle.due() is None, "the lease holds the slot while it is active"
    clock.advance(50)  # the lease has just expired without a /stop
    assert lease.active() is False
    assert idle.due() is None, "the idle clock starts at expiry, not boot"
    clock.advance(60)
    # ever_granted survives the expiry, so it is the short clock, not boot.
    assert idle.due() == "no lease and no session for 60s"


def test_a_produce_session_after_the_lease_holds_the_slot():
    clock = Clock()
    idle, teardown, lease, _, _ = make(clock)
    grant(lease, clock)
    lease.clear()  # /stop cleared the lease, but the session is still up
    teardown.working = True  # /produce holds working()
    clock.advance(3600)
    assert idle.due() is None
    teardown.working = False
    assert idle.due() is None, "the idle clock starts when the session ends"
    clock.advance(60)
    assert idle.due() == "no lease and no session for 60s"


def test_run_exits_zero_and_says_why():
    clock = Clock()
    idle, _, _, exits, logged = make(clock)
    clock.advance(300)
    idle._run()
    assert exits == [0]
    assert logged == ["tee: idle exit: no lease 300s after boot; exiting so the VM stops"]


def test_run_keeps_ticking_while_held():
    clock = Clock()
    idle, _, lease, exits, _ = make(clock, tick_s=0)
    grant(lease, clock)
    ticks = {"n": 0}

    def sleep(_s):
        ticks["n"] += 1
        if ticks["n"] == 3:
            idle.stop()

    idle.sleep = sleep
    idle._run()
    assert exits == [] and ticks["n"] == 3


def test_config_reads_the_env(monkeypatch):
    env = {"TEE_PUBLIC_HOST": "slot-0.tee.masseuse.ai",
           "TRAINER_INVOKER_SERVICE_ACCOUNT": "t@x.iam.gserviceaccount.com"}
    config = tee_mode.TeeConfig.from_env(env)
    assert (config.idle_exit_s, config.boot_idle_s) == (60.0, 300.0)
    config = tee_mode.TeeConfig.from_env({**env, "TEE_IDLE_EXIT_S": "0",
                                          "TEE_BOOT_IDLE_S": "90"})
    assert (config.idle_exit_s, config.boot_idle_s) == (0.0, 90.0)


def test_the_producer_wires_it_in_tee_mode_only_and_warmup_touches():
    plain = producer.build_server(server_args(tee=False), Telemetry(), tee=None)
    try:
        assert plain.idle_exit is None
    finally:
        plain.server_close()
    with Slot() as slot:
        idle = slot.server.idle_exit
        assert isinstance(idle, tee_mode.IdleExit)
        assert (idle.idle_s, idle.boot_idle_s) == (60.0, 300.0)
        assert idle.lease is slot.tee.lease
        assert idle.teardown.is_working() in (True, False)
        before = idle._last_touch
        # Not started by build_server: a test's server must never exit.
        assert not any(t.name == "tee-idle-exit" for t in threading.enumerate())
        code, _, body = slot.as_trainer("GET", "/warmup")
        assert code == 200
        assert idle._last_touch >= before
        code, _, body = slot.as_trainer("GET", "/statz")
        assert code == 200
        snapshot = json.loads(body)["tee"]["idleExit"]
        assert snapshot["everLeased"] is False
        assert snapshot["bootIdleS"] == 300.0


def test_configured_values_reach_the_watchdog():
    tee = make_tee()
    tee.config = tee_mode.TeeConfig(
        public_host=tee.config.public_host,
        trainer_invoker_service_accounts=tee.config.trainer_invoker_service_accounts,
        allowed_origins=tee.config.allowed_origins, tls_cert_dir="/nonexistent",
        idle_exit_s=5, boot_idle_s=42)
    server = producer.build_server(server_args(), Telemetry(), tee=tee)
    try:
        assert (server.idle_exit.idle_s, server.idle_exit.boot_idle_s) == (5.0, 42.0)
    finally:
        server.server_close()
