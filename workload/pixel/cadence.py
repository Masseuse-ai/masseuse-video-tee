"""Which frames of a fixed grid go to the pose model: the cadence picker.

The producer lays every live view's frames on a 30 fps grid (producer.py,
`FPS`) and sends a subset of them to the keypoint model. `index % stride
== 0` serves a cadence that divides the grid (6 of 30 is every fifth
frame) and nothing else: 9 of 30 would round to every third frame, which
is 10 fps. `CadencePicker` picks, for the j-th pose frame, the grid slot
nearest the ideal instant j / pose_fps, so 9 fps is slots 0, 3, 7, 10, 13,
17, ...: a 3-4-3 pattern that is exactly nine per thirty frames, every
pick within a third of a frame of its ideal instant, and a cadence that
divides the grid is the stride it always was.

Whether a slot is picked is a function of its index alone. There is no
state, so a stream and an offline pass over the same frames agree, two
views on their own grids pick the same slots, and a gap in the grid (a
reconnect, a pause) resumes on the slots that would have been picked
without it. The producer's decode loops and the boot-time load bench
(`pose_load`) use the same picker, so the bench's arrivals are the
session's.

`FreshPicker` wraps it for a live source slower than the grid: the decode
conforms every stream to 30 fps (`fps=30`), so a phone uploading 15 fps
arrives as every frame twice and 10 fps as every frame three times. A
slot that lands on a repeat of the frame last posed is deferred to the
next frame that differs, so the picks are distinct pictures at any upload
of 10 fps or more, each within a grid slot of its ideal instant.
"""

from __future__ import annotations

import math

import numpy as np


class CadencePicker:
    """The grid slots of a pose cadence that need not divide the grid.

    `fps` is the grid's rate, `pose_fps` the cadence wanted; a cadence
    above the grid's is the grid's (every slot).
    """

    def __init__(self, fps: float, pose_fps: float):
        pose_fps = float(pose_fps)
        if not pose_fps > 0:
            raise ValueError(f"pose_fps must be positive, got {pose_fps!r}")
        self.fps = float(fps)
        self.pose_fps = min(pose_fps, self.fps)
        # Grid frames per pose interval; 1 or more.
        self.ratio = self.fps / self.pose_fps

    def slot(self, j: int) -> int:
        """The grid slot of the j-th pose frame: the one nearest j * ratio
        (halves round up, so the pattern is the same on every platform)."""
        return int(math.floor(j * self.ratio + 0.5))

    def take(self, index: int) -> bool:
        """Whether grid slot `index` is a pose frame."""
        # The nearest ideal instant is one of the two that bracket
        # index / ratio; with ratio >= 1 no other j can round to `index`.
        base = int(index / self.ratio)
        return any(self.slot(j) == index for j in (base, base + 1))

    def slots(self, count: int) -> list[int]:
        """The first `count` pose slots, for tests and tools."""
        return [self.slot(j) for j in range(max(0, count))]

    @property
    def interval_s(self) -> float:
        """The mean pose interval in seconds."""
        return 1.0 / self.pose_fps

    def __repr__(self) -> str:
        return f"CadencePicker(fps={self.fps:g}, pose_fps={self.pose_fps:g})"


def same_frame(a, b) -> bool:
    """Whether two frame buffers hold the same picture: an element-wise
    equality (numpy's, a vectorised compare well under 0.1 ms for a
    720x1280 4:2:0 frame), or plain `==` for anything that is not an
    array."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b))
    return a == b


class FreshPicker:
    """`CadencePicker`'s slots, taken on distinct frames.

    `take(index, frame)` is the picker's answer for the slot, except that
    a slot whose frame is the same picture as the frame last taken is
    deferred to the next frame that differs. The deferral is bounded by
    the next slot: when that comes round and the picture has still not
    changed, the source is frozen (a paused camera, a link the relay
    bridges by repeating the last frame) and the slot is taken as it
    would have been, so a frozen source yields its cadence of repeats
    rather than nothing and the rows downstream keep their rhythm. The
    first fresh frame after that ends the freeze at once.

    Stateful, unlike `CadencePicker`: the frame last taken is held (the
    decoder hands out a fresh buffer per frame, so it is never
    overwritten under us) and one owed pick is remembered. `deferred`
    counts the slots whose pick moved to a later frame; `repeats` the
    picks taken on a repeated frame because none differed in time.
    """

    def __init__(self, picker: CadencePicker, same=same_frame):
        self.picker = picker
        self.same = same
        self.last = None
        self.owed = False
        self.frozen = False
        self.deferred = 0
        self.repeats = 0

    def take(self, index: int, frame) -> bool:
        due = self.picker.take(index)
        if not due and not self.owed:
            return False
        fresh = self.last is None or not self.same(frame, self.last)
        if fresh:
            self.last = frame
            self.owed = False
            self.frozen = False
            return True
        if not due:
            return False  # owed, and still the same picture: wait
        if self.owed or self.frozen:
            # A whole slot without a change: the source is frozen. The
            # repeat is taken so the cadence goes on.
            self.owed = False
            self.frozen = True
            self.repeats += 1
            return True
        self.owed = True
        self.deferred += 1
        return False

    def __repr__(self) -> str:
        return f"FreshPicker({self.picker!r}, deferred={self.deferred}, repeats={self.repeats})"
