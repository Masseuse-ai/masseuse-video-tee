"""Two views on one clock (producer/sync.py).

Anchored on their decoders' epochs - the senders' own clocks - the mapping
between the views is exact; without an anchor a view's clock is estimated
from its frames' arrivals as the fastest recent arrival, which jitter
cannot pull around and a path change moves within the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "producer"))

from sync import PAIR_TOLERANCE_S, SourceClock, ViewSync  # noqa: E402


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_anchored_views_map_exactly_by_their_epochs():
    sync = ViewSync(clock=_Clock())
    assert not sync.ready and sync.timing == "arrival"
    # The body view's first frame was at T, the face view's 1.25 s later
    # on the same (senders') clock.
    sync.body.anchor(1_757_500_000.0)
    sync.face.anchor(1_757_500_001.25)
    assert sync.ready and sync.timing == "ntp"
    assert sync.face_at_s(2.0) == 0.75
    assert sync.body_at_s(0.75) == 2.0
    assert sync.skew_s() == 1.25
    # Arrivals no longer move it.
    sync.body.note(3.0, wall=5_000.0)
    assert sync.body.offset == 1_757_500_000.0 and sync.body.samples == 1
    assert sync.body.anchored


def test_an_estimated_view_uses_its_fastest_recent_arrival():
    clock = _Clock()
    source = SourceClock(window_s=2.0, clock=clock)
    assert source.offset is None and source.wall_of(0.0) is None
    # Frames at 30 fps arriving 0.4 s after their media time, one of them
    # held up by 0.2 s of jitter: the offset is the fastest, 0.4.
    for k in range(60):
        at_s = k / 30
        clock.now = 1000.0 + at_s + 0.4 + (0.2 if k == 17 else 0.0)
        source.note(at_s)
    assert abs(source.offset - 1000.4) < 1e-9
    assert abs(source.wall_of(1.0) - 1001.4) < 1e-9
    assert abs(source.at_s_of(1001.4) - 1.0) < 1e-9
    # The path shortens by 0.1 s: within the window the minimum follows.
    for k in range(60, 150):
        at_s = k / 30
        clock.now = 1000.0 + at_s + 0.3
        source.note(at_s)
    assert abs(source.offset - 1000.3) < 1e-9
    source.reset()
    assert source.offset is None and not source.anchored


def test_a_reset_keeps_an_anchor_and_none_lifts_it():
    source = SourceClock(clock=_Clock())
    source.anchor(10.0)
    source.reset()
    assert source.offset == 10.0 and source.anchored
    source.anchor(None)
    assert source.offset is None and not source.anchored
    source.note(0.0, wall=12.0)
    assert source.offset == 12.0


def test_mixed_views_are_arrival_timed_and_still_map():
    sync = ViewSync(clock=_Clock())
    sync.body.anchor(1_000.0)
    sync.face.note(0.0, wall=1_000.5)
    assert sync.ready and sync.timing == "arrival"
    assert abs(sync.face_at_s(1.0) - 0.5) < 1e-9


def test_pairing_tolerance():
    sync = ViewSync(clock=_Clock())
    assert sync.paired(1.0, 1.0 + PAIR_TOLERANCE_S)
    assert not sync.paired(1.0, 1.0 + PAIR_TOLERANCE_S + 0.01)
    assert sync.paired(1.0, 1.0 - PAIR_TOLERANCE_S)
