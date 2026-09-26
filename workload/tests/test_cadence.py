"""The cadence picker: which 30 fps grid slots go to the pose model.

What is pinned: 9 fps is the 3-4-3 pattern (0, 3, 7, 10, ...), exactly
nine slots per thirty frames, each within a third of a frame of its ideal
instant; a cadence that divides the grid is the stride it always was; the
decision is a function of the slot index alone, so a gap in the grid, a
second view on its own grid and an offline pass all pick the same slots;
and a cadence above the grid's takes every frame.
"""

from __future__ import annotations

import pytest

from cadence import CadencePicker

GRID = 30.0


def picks(pose_fps: float, slots: int = 300) -> list[int]:
    picker = CadencePicker(GRID, pose_fps)
    return [index for index in range(slots) if picker.take(index)]


def test_nine_fps_is_the_three_four_three_pattern():
    chosen = picks(9.0)
    assert chosen[:10] == [0, 3, 7, 10, 13, 17, 20, 23, 27, 30]
    # Exactly nine per thirty frames, over ten seconds.
    assert len(chosen) == 90
    assert all(len([c for c in chosen if 30 * s <= c < 30 * (s + 1)]) == 9
               for s in range(10))
    gaps = [b - a for a, b in zip(chosen, chosen[1:])]
    assert set(gaps) == {3, 4} and gaps[:9] == [3, 4, 3, 3, 4, 3, 3, 4, 3]


def test_every_pick_is_the_slot_nearest_its_ideal_instant():
    for pose_fps in (9.0, 7.5, 12.0, 4.0, 11.0):
        picker = CadencePicker(GRID, pose_fps)
        chosen = picks(pose_fps)
        assert chosen == picker.slots(len(chosen))
        for j, slot in enumerate(chosen):
            ideal_s = j / pose_fps
            # Within half a grid frame of the ideal instant, always; at 9
            # fps the worst case is a third of a frame (11 ms).
            assert abs(slot / GRID - ideal_s) <= 0.5 / GRID + 1e-9
        if pose_fps == 9.0:
            worst = max(abs(slot / GRID - j / pose_fps) for j, slot in enumerate(chosen))
            assert worst == pytest.approx(1 / 90, abs=1e-9)


def test_a_cadence_that_divides_the_grid_is_its_stride():
    for pose_fps, stride in ((6.0, 5), (10.0, 3), (15.0, 2), (30.0, 1), (5.0, 6), (3.0, 10)):
        assert picks(pose_fps) == list(range(0, 300, stride)), pose_fps


def test_the_decision_is_a_function_of_the_slot_alone():
    """A gap in the grid (a reconnect) and a second picker on its own grid
    pick the same slots as an uninterrupted pass: no state to resume."""
    picker = CadencePicker(GRID, 9.0)
    uninterrupted = {index for index in range(300) if picker.take(index)}
    # The same picker asked out of order, with a gap, answers the same.
    with_gap = {index for index in [*range(40), *range(200, 300)] if picker.take(index)}
    assert with_gap == {i for i in uninterrupted if i < 40 or i >= 200}
    other = CadencePicker(GRID, 9.0)
    assert {index for index in range(300) if other.take(index)} == uninterrupted


def test_a_cadence_above_the_grid_takes_every_frame():
    picker = CadencePicker(GRID, 45.0)
    assert picker.pose_fps == GRID and picker.ratio == 1.0
    assert picks(45.0) == list(range(300))


def test_the_interval_and_the_repr():
    picker = CadencePicker(GRID, 9.0)
    assert picker.interval_s == pytest.approx(1 / 9)
    assert repr(picker) == "CadencePicker(fps=30, pose_fps=9)"
    with pytest.raises(ValueError):
        CadencePicker(GRID, 0.0)
    with pytest.raises(ValueError):
        CadencePicker(GRID, -3.0)
