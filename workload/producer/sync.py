"""Two camera streams on one clock.

A session with a fixed camera behind the user reads two streams: the fixed
camera's (the body view, the session's primary) and the phone's own (the
face view, drawn as an inset). Each is decoded on its own, and each
decoder's media time (`at_s`) counts from its own first frame. The two
counts say nothing about each other; lining the views up needs a clock
they share.

The clock they share, when it can be had, is the senders' own. Both the
phone and the connector time every frame with their wall clock, carried
as the RTCP sender report's NTP time beside the RTP timestamps; the relay
keeps it per path (`useAbsoluteTimestamp`) and `stream-reader`
(workload/reader) hands it to the decoder with each frame. The decoder
lays its frames on a 30 fps grid whose slot zero is the first frame's
sender time, its `epoch` (`producer.Decoder`), so a view's media time is
`sender_time - epoch` exactly, and `SourceClock.anchor(epoch)` makes the
conversion exact: a moment on one view's timeline is the same moment on
the other's to the precision of the two senders' clocks, which both keep
by NTP.

When a view has no sender time - the reader is not there, the sender does
not report, or its clock is plainly off - the fallback is this host's
clock. Every frame lands in the producer at a wall time, and for one stream
the difference between arrival and media time is a constant - the path's
latency - plus the jitter of that path, which only ever adds. So the
fastest arrival seen recently is the constant, and `SourceClock.note`
keeps it as the minimum of `wall - at_s` over a sliding window of media
time; what it cannot see is the path's own latency, which then shows as a
fixed offset between the views of a few frames. The session says which of
the two a view is on (`timing`: `ntp` or `arrival`) in its status.

`ViewSync` holds a clock per view and maps a moment on one onto the other:
the face frame to draw beside body frame `t` is the one whose time puts it
at the same moment.
"""

from __future__ import annotations

import time
from collections import deque

# How much media time the arrival offset is estimated over. Long enough
# that a tick's jitter cannot pull the minimum around (60 frames at 30
# fps), short enough that a path change is followed in that many frames.
WINDOW_S = 2.0
# How far apart two frames may be, in wall time, and still be drawn as one
# moment. A quarter of a second is about eight frames of the body view:
# the face lags or leads by less than the eye reads as separate.
PAIR_TOLERANCE_S = 0.25


class SourceClock:
    """One stream's media time against the wall: `wall_of` and `at_s_of`
    convert once `anchor` or `note` has placed it."""

    def __init__(self, window_s: float = WINDOW_S, clock=time.monotonic):
        self.window_s = float(window_s)
        self.clock = clock
        # (at_s, wall - at_s), oldest first, over the last `window_s` of
        # media time; unused once anchored.
        self._samples: deque[tuple[float, float]] = deque()
        self._offset: float | None = None
        self._anchor: float | None = None
        self.samples = 0

    def anchor(self, epoch: float | None) -> None:
        """Media time zero is `epoch` exactly, on the sender's clock: the
        arrival estimate is set aside. None returns to estimating."""
        self._anchor = None if epoch is None else float(epoch)
        self._samples.clear()
        self._offset = self._anchor

    @property
    def anchored(self) -> bool:
        return self._anchor is not None

    def note(self, at_s: float, wall: float | None = None) -> None:
        """A frame at media time `at_s` arrived at `wall` (now, by default).
        Counted but otherwise ignored while anchored."""
        self.samples += 1
        if self._anchor is not None:
            return
        if wall is None:
            wall = self.clock()
        self._samples.append((at_s, wall - at_s))
        while self._samples and self._samples[0][0] < at_s - self.window_s:
            self._samples.popleft()
        self._offset = min(offset for _, offset in self._samples)

    @property
    def offset(self) -> float | None:
        """Wall time of media time zero, or None before the first frame."""
        return self._offset

    def wall_of(self, at_s: float) -> float | None:
        return None if self._offset is None else at_s + self._offset

    def at_s_of(self, wall: float) -> float | None:
        return None if self._offset is None else wall - self._offset

    def reset(self) -> None:
        """Forget the arrival estimate (a path change); an anchor stays."""
        self._samples.clear()
        self._offset = self._anchor


class ViewSync:
    """The face view's clock against the body view's."""

    def __init__(self, tolerance_s: float = PAIR_TOLERANCE_S,
                 window_s: float = WINDOW_S, clock=time.monotonic):
        self.clock = clock
        self.tolerance_s = float(tolerance_s)
        self.body = SourceClock(window_s, clock)
        self.face = SourceClock(window_s, clock)

    @property
    def ready(self) -> bool:
        return self.body.offset is not None and self.face.offset is not None

    @property
    def timing(self) -> str:
        """`ntp` when both views are on their senders' clocks, else
        `arrival`."""
        return "ntp" if self.body.anchored and self.face.anchored else "arrival"

    def face_at_s(self, body_at_s: float) -> float | None:
        """The face view's media time for a moment on the body view's."""
        wall = self.body.wall_of(body_at_s)
        return None if wall is None else self.face.at_s_of(wall)

    def body_at_s(self, face_at_s: float) -> float | None:
        """The body view's media time for a moment on the face view's."""
        wall = self.face.wall_of(face_at_s)
        return None if wall is None else self.body.at_s_of(wall)

    def skew_s(self) -> float | None:
        """How much later than the body's the face's frames arrive for the
        same media time: the difference of the two offsets. Descriptive
        (the HUD, the status body); the mapping above already removes it."""
        if not self.ready:
            return None
        return self.face.offset - self.body.offset

    def paired(self, wanted_at_s: float, found_at_s: float) -> bool:
        """Whether a frame found at `found_at_s` stands for `wanted_at_s`."""
        return abs(found_at_s - wanted_at_s) <= self.tolerance_s
