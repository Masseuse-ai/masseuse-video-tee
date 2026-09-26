"""The detector's host-side selection over its packed rows.

`candidates_from_packed` replaces the tensor indexing the tracker used to do
on the device (`labels == person`, `scores >= floor`, `boxes[keep]`, then a
float per coordinate): the same candidates must come out of the normalised
rows copied back in one piece.
"""

from __future__ import annotations

import random

import pytest

from pose_rows import (
    PERSON_SCORE, RTDETRV4_FRAGMENT_SCORE, TRACK_SCORE,
    _fuse_rtdetrv4_fragments, candidates_from_packed,
)

PERSON = 0
WIDTH, HEIGHT = 640.0, 360.0


def _previous_selection(labels, boxes, scores, threshold):
    """What `_RtdetrV4Detector.candidates` computed before, over tensors."""
    floor = min(threshold, RTDETRV4_FRAGMENT_SCORE)
    raw = [
        ([float(v) for v in box], float(score))
        for label, box, score in zip(labels, boxes, scores)
        if label == PERSON and score >= floor
    ]
    return _fuse_rtdetrv4_fragments(raw, threshold)


def _queries(seed: int, count: int = 300):
    rng = random.Random(seed)
    labels, boxes, scores = [], [], []
    for _ in range(count):
        labels.append(rng.choice([PERSON, PERSON, 1, 56, 62]))
        x0, y0 = rng.uniform(0, WIDTH - 40), rng.uniform(0, HEIGHT - 40)
        boxes.append([x0, y0, x0 + rng.uniform(20, WIDTH - x0),
                      y0 + rng.uniform(20, HEIGHT - y0)])
        scores.append(rng.choice([rng.uniform(0, 1), rng.uniform(0, 0.05)]))
    return labels, boxes, scores


def _packed(labels, boxes, scores):
    return [
        [x0 / WIDTH, y0 / HEIGHT, x1 / WIDTH, y1 / HEIGHT, score, float(label)]
        for label, (x0, y0, x1, y1), score in zip(labels, boxes, scores)
    ]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("threshold", [min(PERSON_SCORE, TRACK_SCORE), 0.5])
def test_packed_rows_select_the_same_candidates(seed, threshold):
    labels, boxes, scores = _queries(seed)
    expected = _previous_selection(labels, boxes, scores, threshold)
    actual = candidates_from_packed(
        _packed(labels, boxes, scores), WIDTH, HEIGHT, PERSON, threshold)
    assert len(actual) == len(expected) > 0
    for (box, score), (want_box, want_score) in zip(actual, expected):
        assert box == pytest.approx(want_box, abs=1e-9)
        assert score == pytest.approx(want_score)


def test_only_the_person_label_at_or_above_the_floor_qualifies():
    rows = [
        [0.1, 0.1, 0.5, 0.9, 0.90, 0.0],   # a person
        [0.1, 0.1, 0.5, 0.9, 0.95, 1.0],   # a bicycle, stronger: ignored
        [0.2, 0.2, 0.4, 0.8, RTDETRV4_FRAGMENT_SCORE, 0.0],  # at the floor
        [0.2, 0.2, 0.4, 0.8, RTDETRV4_FRAGMENT_SCORE / 2, 0.0],  # below it
    ]
    accepted = candidates_from_packed(rows, WIDTH, HEIGHT, PERSON, PERSON_SCORE)
    assert accepted == [([64.0, 36.0, 320.0, 324.0], 0.9)]
    # The floor is the lower of the threshold and the fragment score: a
    # fragment below the person threshold is kept for fusion, not returned.
    assert candidates_from_packed(
        rows, WIDTH, HEIGHT, PERSON, RTDETRV4_FRAGMENT_SCORE) == [
        ([64.0, 36.0, 320.0, 324.0], 0.9),
        ([128.0, 72.0, 256.0, 288.0], RTDETRV4_FRAGMENT_SCORE),
    ]


def test_labels_arrive_as_floats_from_the_packed_tensor():
    rows = [[0.0, 0.0, 0.5, 0.5, 0.8, 0.0], [0.0, 0.0, 0.5, 0.5, 0.8, 2.0]]
    assert len(candidates_from_packed(rows, WIDTH, HEIGHT, PERSON, 0.5)) == 1
    assert len(candidates_from_packed(rows, WIDTH, HEIGHT, 2, 0.5)) == 1
    assert candidates_from_packed(rows, WIDTH, HEIGHT, 1, 0.5) == []
