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
"""

from __future__ import annotations

import math


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
