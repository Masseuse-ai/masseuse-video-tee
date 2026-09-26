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

from cadence import CadencePicker, FreshPicker, same_frame

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


# -- distinct frames -------------------------------------------------------------


def conformed(source_fps: float, seconds: float = 10.0, phase: int = 0) -> list[int]:
    """A source at `source_fps` laid on the 30 fps grid the way ffmpeg's
    `fps=30` does it: each grid slot shows the source frame current at
    its instant, so a slower source is repeats. Returns, per grid slot,
    the id of the source frame shown. `phase` shifts where the source's
    frames change relative to the grid."""
    slots = int(seconds * GRID)
    return [int((index + phase) * source_fps / GRID) for index in range(slots)]


def fresh_picks(source_fps: float, pose_fps: float = 9.0, phase: int = 0,
                seconds: float = 10.0) -> tuple[list[int], list[int], FreshPicker]:
    """(grid slots taken, source frame ids taken, the picker)."""
    picker = FreshPicker(CadencePicker(GRID, pose_fps))
    shown = conformed(source_fps, seconds, phase)
    taken = [index for index, frame in enumerate(shown) if picker.take(index, frame)]
    return taken, [shown[index] for index in taken], picker


@pytest.mark.parametrize("source_fps", [30.0, 20.0, 15.0, 10.0])
@pytest.mark.parametrize("phase", [0, 1, 2])
def test_a_slower_source_conformed_to_the_grid_yields_distinct_picks(source_fps, phase):
    """A 15 fps upload arrives as every frame twice, 10 fps as three
    times; the 9 fps picks are nine distinct pictures a second all the
    same, each within one grid slot of the 3-4-3 slot it would have been."""
    slots, frames, picker = fresh_picks(source_fps, phase=phase)
    assert len(frames) == len(set(frames)), "a frame was posed twice"
    assert len(slots) == 90, "nine picks a second over ten seconds"
    ideal = CadencePicker(GRID, 9.0).slots(len(slots))
    assert all(0 <= slot - want <= 1 for slot, want in zip(slots, ideal)), \
        "a pick moved by more than one slot, or backwards"
    assert picker.repeats == 0


def test_a_ten_fps_source_with_jitter_moves_the_pick_not_the_cadence():
    """The relay's conform is not phase-locked to the phone: a frame held
    for four slots instead of three puts a pick on a repeat, which moves
    to the next distinct frame and is counted once."""
    shown = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9]
    picker = FreshPicker(CadencePicker(GRID, 9.0))
    taken = [index for index, frame in enumerate(shown) if picker.take(index, frame)]
    # Slot 3 shows frame 0 again (taken at slot 0): deferred to slot 4.
    assert taken == [0, 4, 7, 10, 13, 17, 20, 23, 27]
    assert [shown[i] for i in taken] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert picker.deferred == 1 and picker.repeats == 0


def test_a_frozen_source_yields_its_cadence_of_repeats_after_one_slot():
    """The picture stops changing (a paused camera): the first slot on it
    waits one slot for a change that does not come, then the cadence goes
    on with repeats, and the first fresh frame is taken at once."""
    picker = FreshPicker(CadencePicker(GRID, 9.0))
    shown = [0, 0, 0, 1, 1, 1, 1] + [1] * 23 + [2, 2, 2, 3]
    taken = [index for index, frame in enumerate(shown) if picker.take(index, frame)]
    # 0 at slot 0; 1 at slot 3; slot 7 repeats 1: deferred; slot 10 still 1:
    # frozen, taken; from then on every slot of the cadence, as repeats.
    assert taken[:6] == [0, 3, 10, 13, 17, 20]
    assert 27 in taken and 7 not in taken
    # Frame 2 arrives at slot 30 (a cadence slot): taken as fresh, the
    # freeze ended; frame 3 at slot 33 likewise.
    assert taken[-2:] == [30, 33]
    assert picker.deferred == 1
    assert picker.repeats == len([s for s in taken if 10 <= s <= 27])
    assert picker.frozen is False


def test_an_owed_pick_is_taken_on_the_first_fresh_frame_between_slots():
    picker = FreshPicker(CadencePicker(GRID, 9.0))
    # Slot 3 repeats frame 0; frame 1 appears at slot 5, before slot 7.
    shown = [0, 0, 0, 0, 0, 1, 1, 2, 2]
    taken = [index for index, frame in enumerate(shown) if picker.take(index, frame)]
    assert taken == [0, 5, 7]
    assert picker.deferred == 1 and picker.repeats == 0


def test_same_frame_compares_pictures_not_identities():
    import numpy as np
    a = np.zeros((6, 4), np.uint8)
    b = np.zeros((6, 4), np.uint8)
    assert same_frame(a, b) and same_frame(a, a)
    b[0, 0] = 1
    assert not same_frame(a, b)
    assert not same_frame(a, np.zeros((4, 6), np.uint8))
    assert same_frame(b"x", b"x") and not same_frame(b"x", b"y")
    picker = FreshPicker(CadencePicker(GRID, 9.0))
    assert picker.take(0, a) and not picker.take(3, np.zeros((6, 4), np.uint8))
    assert "deferred=1" in repr(picker)
