"""The /teardown decision: never under an open session, and never answered
by a replacement instance.

Teardown exists so a cancelled session stops Blackwell billing rather
than idling through Cloud Run's keep-warm. Exiting is the easy half. The
autoscaler's CPU driver still recommends one instance for a minute after
the last busy second, and an instance that dies inside that minute is
re-created and billed until the lookback clears. So a drain teardown waits
until the instance has been quiet for the window, while `mode=now` keeps
the immediate exit for callers that want a fresh instance anyway.

Properties locked down here: an open /produce session refuses it; the
drain delay is measured from the last busy moment; a /warmup or /produce
inside the drain cancels it; the watchdog will not exit under a session
that took the lock; `mode=now` holds the lock to the grave.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "producer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

from producer import Teardown  # noqa: E402


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Gate:
    """A sleep the test controls: the watchdog parks here until opened."""

    def __init__(self):
        self.parked = threading.Event()
        self.opened = threading.Event()

    def __call__(self, _seconds: float) -> None:
        self.parked.set()
        self.opened.wait(timeout=2.0)


class Exit:
    def __init__(self):
        self.fired = threading.Event()
        self.code = None

    def __call__(self, code: int) -> None:
        self.code = code
        self.fired.set()


def make(drain_s: float = 90.0, sleep=None):
    busy = threading.Lock()
    clock = Clock()
    exit_ = Exit()
    gate = sleep if sleep is not None else Gate()
    teardown = Teardown(busy, drain_s=drain_s, clock=clock, exit_impl=exit_,
                        sleep=gate)
    return teardown, busy, clock, exit_, gate


def wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_teardown_refuses_while_a_session_is_running():
    teardown, busy, _, exit_, _ = make()
    busy.acquire()

    code, body = teardown.request()

    assert code == 409
    assert body == {"error": "a session is running"}
    assert teardown.snapshot()["draining"] is False
    assert not exit_.fired.is_set()


def test_a_fresh_session_end_drains_for_the_full_window():
    teardown, busy, clock, exit_, gate = make()
    with teardown.working():
        clock.now += 133.0  # the session itself
    # No time has passed since the session ended.

    code, body = teardown.request()

    assert code == 200
    assert body == {"status": "draining", "exitInS": 90.0, "drainS": 90.0}
    assert teardown.snapshot() == {"draining": True, "exitInS": 90.0,
                                   "quietForS": 0.0, "drainS": 90.0}
    # The drain does not hold the lock: a session may start on the warm
    # pipeline instead of waiting for a cold boot on a replacement.
    assert busy.acquire(blocking=False) is True
    busy.release()
    assert gate.parked.wait(timeout=2.0)
    assert not exit_.fired.is_set()
    gate.opened.set()


def test_an_instance_already_quiet_past_the_window_exits_at_once(capsys):
    teardown, busy, clock, exit_, gate = make()
    with teardown.working():
        pass
    clock.now += 100.0

    code, body = teardown.request()

    assert (code, body["status"], body["exitInS"]) == (200, "draining", 0.0)
    assert exit_.fired.wait(timeout=2.0)
    assert exit_.code == 0
    # The exit took the lock, so nothing starts on the dying instance.
    assert busy.acquire(blocking=False) is False
    assert "teardown: exiting after 90s quiet" in capsys.readouterr().out


def test_the_drain_counts_from_the_last_busy_moment_not_the_request():
    teardown, _, clock, exit_, gate = make()
    with teardown.working():
        pass
    clock.now += 30.0

    _, body = teardown.request()
    assert body["exitInS"] == 60.0

    # The watchdog wakes, sees the remaining window, and keeps waiting.
    assert gate.parked.wait(timeout=2.0)
    assert not exit_.fired.is_set()
    clock.now += 60.0
    gate.opened.set()
    assert exit_.fired.wait(timeout=2.0)


def test_working_resets_the_quiet_clock_and_nests():
    teardown, _, clock, _, _ = make()
    clock.now += 500.0
    with teardown.working():
        with teardown.working():
            clock.now += 10.0
            assert teardown.quiet_for_s() == 0.0
        clock.now += 10.0
        assert teardown.quiet_for_s() == 0.0  # the outer scope is still busy
    clock.now += 5.0
    assert teardown.quiet_for_s() == 5.0
    assert teardown.exit_in_s() == 85.0


def test_a_warmup_during_the_drain_cancels_it(capsys):
    teardown, _, clock, exit_, gate = make()
    teardown.request()
    assert gate.parked.wait(timeout=2.0)

    assert teardown.cancel("/warmup") is True
    assert teardown.cancel("/warmup") is False  # nothing left to cancel
    assert teardown.snapshot()["draining"] is False
    assert teardown.snapshot()["exitInS"] is None
    assert "teardown: drain cancelled by /warmup" in capsys.readouterr().out

    # Even with the window long expired, the cancelled watchdog stays dead.
    clock.now += 500.0
    gate.opened.set()
    time.sleep(0.05)
    assert not exit_.fired.is_set()


def test_a_second_drain_request_replaces_the_first_watchdog():
    teardown, _, clock, exit_, gate = make()
    teardown.request()
    assert gate.parked.wait(timeout=2.0)
    teardown.cancel("/warmup")
    code, body = teardown.request()
    assert (code, body["status"]) == (200, "draining")
    clock.now += 100.0
    gate.opened.set()
    assert exit_.fired.wait(timeout=2.0)


def test_the_watchdog_yields_to_a_session_that_took_the_lock(capsys):
    teardown, busy, clock, exit_, gate = make()
    teardown.request()
    assert gate.parked.wait(timeout=2.0)

    # A /produce slipped in without calling cancel (belt and braces).
    busy.acquire()
    clock.now += 100.0
    gate.opened.set()

    assert wait_until(lambda: teardown.snapshot()["draining"] is False)
    assert not exit_.fired.is_set()
    assert "teardown: drain cancelled by a session" in capsys.readouterr().out


def test_mode_now_exits_immediately_and_holds_the_lock(capsys):
    teardown, busy, _, exit_, _ = make(sleep=lambda _s: None)

    code, body = teardown.request("now")

    assert code == 200
    assert body == {"status": "terminating"}
    # Held to the grave: a /produce arriving in the exit window sees 409.
    assert busy.acquire(blocking=False) is False
    assert exit_.fired.wait(timeout=2.0)
    assert teardown.snapshot()["draining"] is False
    assert "teardown: exiting now" in capsys.readouterr().out


def test_mode_now_also_refuses_under_a_session():
    teardown, busy, _, exit_, _ = make(sleep=lambda _s: None)
    busy.acquire()
    code, _ = teardown.request("now")
    assert code == 409
    assert not exit_.fired.is_set()


def test_the_request_does_not_block_the_caller():
    teardown, _, _, _, _ = make()
    started = time.monotonic()
    teardown.request()
    assert time.monotonic() - started < 0.2
