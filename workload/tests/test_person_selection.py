"""Tests for choosing which detected box is the tracked person.

The detector does not return one clean person. On the 2026-05-30 reference
clip it returns the whole body and, separately, the upper half sharing the
same top-left corner, and the full-body box's score wanders across the
acceptance threshold frame to frame while the half-body box scores 0.86.
Selection therefore has to survive two things that really happened: one
person being counted as four people, and the crop handed to Sapiens2
flipping between the whole body and half of it every other frame.

The boxes below are the real ones, read off frames 744-775 of that clip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

from pose_rows import (  # noqa: E402
    MERGE_CONTAINMENT, MERGE_IOU, PERSON_SCORE, SCENERY_MOTION_RATIO,
    TRACK_IOU, TRACK_SCORE, _fuse_rtdetrv4_fragments, choose_person,
    containment, find_scenery, iou,
)


# Frame 771: the full body, and the upper half nested inside it.
FULL = [1318.0, 1279.0, 2338.0, 1752.0]
HALF = [1327.0, 1276.0, 1765.0, 1757.0]


class FakeTracker:
    """Just the selection logic, with detection stubbed out.

    The real Tracker loads two networks onto a GPU, which a unit test has no
    business doing. The selection is pure geometry over candidate boxes
    (`pose_rows.choose_person`), so it is exercised directly with the same
    identity state the tracker carries.
    """

    def __init__(self, scenery=()):
        self.previous = None
        self.scenery = list(scenery)

    def choose(self, candidates):
        box, score, people, unresolved, self.previous = choose_person(
            candidates, self.previous, self.scenery
        )
        return box, score, people, unresolved


def choose(tracker, candidates):
    return tracker.choose(candidates)


def test_a_nested_half_body_box_is_not_a_second_person():
    """The bug that reported four people in a room containing one."""
    assert iou(FULL, HALF) < MERGE_IOU, "IoU alone cannot see the nesting"
    assert containment(FULL, HALF) >= MERGE_CONTAINMENT

    tracker = FakeTracker()
    _, _, people, unresolved = choose(tracker, [(FULL, 0.389), (HALF, 0.837)])
    assert people == 1
    assert not unresolved


def test_the_full_body_wins_even_when_the_half_body_scores_higher():
    tracker = FakeTracker()
    box, _, _, _ = choose(tracker, [(FULL, 0.389), (HALF, 0.837)])
    assert box == pytest.approx([FULL[0], FULL[1],
                                 FULL[2] - FULL[0], FULL[3] - FULL[1]])


def test_a_weak_full_body_box_is_kept_when_it_matches_the_previous_frame():
    """Frame 772, where the full body scored 0.308 and was thrown away.

    Losing it handed Sapiens2 a half-body crop, which is how the pelvis series
    acquired step changes that no movement produced.
    """
    tracker = FakeTracker()
    choose(tracker, [(FULL, 0.699), (HALF, 0.616)])

    weak = 0.308
    assert weak < PERSON_SCORE and weak >= TRACK_SCORE
    box, score, _, _ = choose(tracker, [(FULL, weak), (HALF, 0.856)])
    assert score == pytest.approx(weak)
    assert box[2] == pytest.approx(FULL[2] - FULL[0])


def test_a_half_body_lock_recovers_on_the_next_frame():
    """Continuity must not be self-reinforcing.

    Ranking candidates by overlap with the previous box makes a wrong pick
    permanent: once locked to the upper half, the half box overlaps itself
    better than the full body does. Containment is what breaks the lock.
    """
    tracker = FakeTracker()
    tracker.previous = HALF

    assert iou(FULL, HALF) < TRACK_IOU, "overlap alone would not requalify it"
    assert containment(FULL, HALF) >= MERGE_CONTAINMENT

    box, _, _, _ = choose(tracker, [(FULL, 0.389), (HALF, 0.837)])
    assert box[2] == pytest.approx(FULL[2] - FULL[0])


def test_two_genuinely_separate_people_stay_two():
    """The merge must not collapse a real second person into the first."""
    other = [3000.0, 1279.0, 3600.0, 1752.0]
    assert iou(FULL, other) == 0.0
    assert containment(FULL, other) == 0.0

    tracker = FakeTracker()
    _, _, people, unresolved = choose(tracker, [(FULL, 0.9), (other, 0.9)])
    assert people == 2
    assert unresolved


def test_a_far_away_weak_box_cannot_hijack_the_track():
    tracker = FakeTracker()
    choose(tracker, [(FULL, 0.699)])
    elsewhere = [3000.0, 1279.0, 3600.0, 1752.0]

    box, _, _, _ = choose(tracker, [(elsewhere, TRACK_SCORE + 0.01)])
    assert box is None, "a sub-threshold box that is not the tracked person is not a person"


def test_nothing_detected_clears_the_anchor():
    """A dropped frame must not anchor identity to a stale position."""
    tracker = FakeTracker()
    choose(tracker, [(FULL, 0.699)])
    choose(tracker, [])
    assert tracker.previous is None


def test_v4_low_score_leg_fragment_repairs_the_sapiens_crop():
    """A weak fragment may extend a person, but never become one itself."""
    torso = [1324.8, 1255.4, 1926.2, 1711.6]
    legs = [1516.2, 1331.0, 2253.3, 1634.7]
    fused = _fuse_rtdetrv4_fragments(
        [(torso, 0.8681), (legs, 0.0316)], TRACK_SCORE
    )

    assert (legs, 0.0316) not in fused
    assert (torso, 0.8681) in fused
    envelope, score = max(fused, key=lambda row: row[0][2] - row[0][0])
    assert envelope == pytest.approx([1324.8, 1255.4, 2253.3, 1711.6])
    assert score == pytest.approx(0.8681)


def test_v4_giant_weak_proposal_cannot_expand_the_person():
    torso = [1334.5, 1277.7, 1869.8, 1735.6]
    unrelated = [198.2, 901.9, 1605.4, 2144.9]
    fused = _fuse_rtdetrv4_fragments(
        [(torso, 0.8830), (unrelated, 0.0298)], TRACK_SCORE
    )

    assert fused == [(torso, 0.8830)]


def test_v4_upright_full_body_box_does_not_absorb_a_fragment():
    full_body = [1345.0, 1291.8, 1978.9, 1893.1]
    weak_right = [1574.3, 1338.3, 2251.2, 1610.1]
    fused = _fuse_rtdetrv4_fragments(
        [(full_body, 0.8963), (weak_right, 0.0378)], TRACK_SCORE
    )

    assert fused == [(full_body, 0.8963)]


# The clothing pile the detector scored 0.37-0.42 as a person, all session.
CLOTHES = [2309.0, 1260.0, 2663.0, 1473.0]


def _clip(*boxes, frames=100):
    return [list(boxes) for _ in range(frames)]


def test_an_unmoving_object_is_scenery():
    still = find_scenery(
        _clip(FULL, CLOTHES),
        motion=lambda box: 0.2 if box is CLOTHES else 5.7,
    )
    assert still == [CLOTHES]


def test_a_man_lying_still_is_not_scenery():
    """The failure that made the first attempt at this delete the person.

    The person is prone on a bed for the whole clip, so their box holds
    position as well as the furniture does. Only their pixels give them away.
    """
    still = find_scenery(
        _clip(FULL, CLOTHES),
        motion=lambda box: 0.2 if box is CLOTHES else 5.7,
    )
    assert FULL not in still


def test_scenery_needs_a_real_gap_not_just_the_lowest_motion():
    """Two moving boxes must both survive, however they rank against each other."""
    other = [3000.0, 1279.0, 3600.0, 1752.0]
    energies = {id(FULL): 5.7, id(other): 5.7 * SCENERY_MOTION_RATIO * 1.5}
    assert find_scenery(_clip(FULL, other),
                        motion=lambda box: energies[id(box)]) == []


def test_a_box_seen_only_briefly_is_never_scenery():
    """Someone passing through must not be suppressed for being still."""
    frames = _clip(FULL, frames=100)
    frames[0] = [FULL, CLOTHES]
    assert find_scenery(frames, motion=lambda box: 0.0) == []


def test_no_candidates_yields_no_scenery():
    assert find_scenery([[] for _ in range(50)], motion=lambda box: 0.0) == []


def test_known_scenery_stops_inflating_the_people_count():
    """The wiring: finding the clothing pile only helps if selection drops it."""
    counted = FakeTracker()
    _, _, before, _ = choose(counted, [(FULL, 0.9), (CLOTHES, 0.42)])
    assert before == 2

    filtered = FakeTracker(scenery=[CLOTHES])
    box, _, after, unresolved = choose(filtered, [(FULL, 0.9), (CLOTHES, 0.42)])
    assert after == 1
    assert not unresolved
    assert box[2] == pytest.approx(FULL[2] - FULL[0]), "and the person is still chosen"
