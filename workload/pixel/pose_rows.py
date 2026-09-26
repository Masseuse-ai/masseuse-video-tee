"""Person selection and the pose row shape - the tracker's geometry, no torch.

`pose_track.Tracker` runs the detector and Sapiens2 on the producer's GPU;
everything it decides *about* the boxes it gets back lives here, over plain
lists of floats, so the unit tests and the CPU analysis tools exercise the
exact production logic without importing a deep-learning stack. Rows written
by `row_for` / `missing_row` are the producer's `poses.jsonl` format.
"""

from __future__ import annotations

from collections.abc import Callable

from keypoints import BODY, KEYPOINT_NAMES, LOWER_BODY, MIN_KEYPOINT_SCORE

# On prone frames RT-DETRv4-X sometimes emits the torso as the high-confidence
# person and the legs as an overlapping low-confidence person query. The latter
# must not qualify as a person independently, but its extent repairs the crop
# that Sapiens2 receives. These guards were pinned on the three-frame smoke
# gate: the true fragment is 0.82-1.50x the torso area and 55-92% contained;
# a giant unrelated 0.03-score proposal is 7.1x the torso and is therefore
# excluded.
RTDETRV4_FRAGMENT_SCORE = 0.025
RTDETRV4_FRAGMENT_CONTAINMENT = 0.45
RTDETRV4_FRAGMENT_MAX_AREA_RATIO = 2.0
RTDETRV4_FRAGMENT_MIN_SEED_ASPECT = 1.10
PERSON_SCORE = 0.35

# Two boxes this close together are one person the detector split, or two people
# it could not separate. Either way the frame's identity is unresolved.
MERGE_IOU = 0.55

# A box this far inside another is a part of that person, not a second person.
MERGE_CONTAINMENT = 0.80

# Continuity: a candidate overlapping the previous frame's person this much is
# the same person again, and is accepted down to TRACK_SCORE rather than
# PERSON_SCORE. The full-body box scores 0.31-0.70 frame to frame while a
# spurious upper-half box
# scores 0.86, so a fixed threshold alone silently prefers the wrong one.
TRACK_IOU = 0.50
TRACK_SCORE = 0.25

# Furniture the detector calls a person. On this clip a pile of clothing on the
# chair and a body pillow both score above PERSON_SCORE all session long.
#
# They are not separated by holding still in the frame: the person is prone on
# a bed and their box moves as little as the objects' do, so a position-based
# test deletes the person from 397 of 796 frames. What separates them is that
# the person's pixels change and the objects' do not, so the test is motion
# inside the box, measured relative to the motion inside the person's. An
# object sits near zero; a person cannot, because they are breathing at the
# very least.
SCENERY_IOU = 0.70
SCENERY_FRACTION = 0.40
SCENERY_MOTION_RATIO = 0.25

Box = list[float]
Candidate = tuple[Box, float]


def _intersection(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return 0.0 if x1 <= x0 or y1 <= y0 else (x1 - x0) * (y1 - y0)


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    """Overlap of two xyxy boxes."""
    overlap = _intersection(a, b)
    return 0.0 if not overlap else overlap / (_area(a) + _area(b) - overlap)


def containment(a: Box, b: Box) -> float:
    """How much of the smaller box lies inside the larger one.

    IoU cannot see nesting. On this footage the detector routinely returns the
    whole body and, separately, the upper half sharing the same top-left
    corner: the smaller box is 99% inside the larger, yet their IoU is only
    0.43-0.50 because of the size difference. Judged on IoU alone those score
    as two people, which is how a man alone in a room came to be counted as
    four.
    """
    overlap = _intersection(a, b)
    return 0.0 if not overlap else overlap / min(_area(a), _area(b))


def recurring_boxes(per_frame: list[list[Box]]) -> list[tuple[Box, int]]:
    """Candidate boxes that keep appearing in the same place, with their counts."""
    clusters: list[list] = []
    for boxes in per_frame:
        for box in boxes:
            for cluster in clusters:
                if iou(box, cluster[0]) >= SCENERY_IOU:
                    cluster[1] += 1
                    break
            else:
                clusters.append([box, 1])
    needed = max(2, int(len(per_frame) * SCENERY_FRACTION))
    return [(box, seen) for box, seen in clusters if seen >= needed]


def find_scenery(
    per_frame: list[list[Box]],
    motion: Callable[[Box], float],
) -> list[Box]:
    """Recurring boxes whose contents never change: objects, not people.

    `motion` returns mean inter-frame pixel change inside a box. The threshold
    is relative to the liveliest recurring box, which is the person, so it needs no
    absolute calibration and survives a change of camera, exposure or codec.

    This cannot suppress a second person unless they hold utterly still for
    most of the clip, which would also mean they were not breathing.
    """
    recurring = recurring_boxes(per_frame)
    if not recurring:
        return []
    energies = {id(box): motion(box) for box, _ in recurring}
    liveliest = max(energies.values())
    if liveliest <= 0.0:
        return []
    return [
        box for box, _ in recurring
        if energies[id(box)] / liveliest < SCENERY_MOTION_RATIO
    ]


def _fuse_rtdetrv4_fragments(
    candidates: list[Candidate], threshold: float
) -> list[Candidate]:
    """Add full-body envelopes without admitting weak fragments as people.

    RT-DETRv4-X is very confident about the visible torso, but on the
    fixed camera's prone geometry it can represent the legs as a
    separate low-score person query. A synthesized envelope retains the seed's
    confidence. The raw fragment is never returned, so the shared person
    threshold and multi-person safety behavior remain unchanged.
    """
    accepted = [(box, score) for box, score in candidates if score >= threshold]
    fused: list[Candidate] = []
    for seed, seed_score in accepted:
        seed_area = _area(seed)
        seed_aspect = (seed[2] - seed[0]) / (seed[3] - seed[1])
        if seed_aspect < RTDETRV4_FRAGMENT_MIN_SEED_ASPECT:
            continue
        envelope = list(seed)
        for fragment, fragment_score in candidates:
            if not RTDETRV4_FRAGMENT_SCORE <= fragment_score < threshold:
                continue
            area_ratio = _area(fragment) / seed_area
            if area_ratio > RTDETRV4_FRAGMENT_MAX_AREA_RATIO:
                continue
            if containment(seed, fragment) < RTDETRV4_FRAGMENT_CONTAINMENT:
                continue
            envelope = [
                min(envelope[0], fragment[0]),
                min(envelope[1], fragment[1]),
                max(envelope[2], fragment[2]),
                max(envelope[3], fragment[3]),
            ]
        if envelope != seed:
            fused.append((envelope, seed_score))
    return accepted + fused


def choose_person(
    candidates: list[Candidate],
    previous: Box | None,
    scenery: list[Box],
) -> tuple[Box | None, float, int, bool, Box | None]:
    """Pick the tracked person out of one frame's candidate boxes, and count
    the people.

    Returns (box in COCO xywh, score, distinct people, identity unresolved,
    the xyxy box to anchor the next frame on). Pure geometry over boxes, so
    the part of tracking that was wrong is the part that is unit-tested;
    everything around it is model inference.
    """
    candidates = [
        (box, score) for box, score in candidates
        if not any(iou(box, still) >= SCENERY_IOU for still in scenery)
    ]

    def same_person(box: Box) -> bool:
        """The same person again, by overlap or by nesting either way round."""
        return bool(previous) and (
            iou(box, previous) >= TRACK_IOU
            or containment(box, previous) >= MERGE_CONTAINMENT
        )

    usable = [
        (b, s) for b, s in candidates
        if s >= PERSON_SCORE or (s >= TRACK_SCORE and same_person(b))
    ]
    if not usable:
        return None, 0.0, 0, False, None

    # Largest, but only among boxes that are the same person. Continuity is a
    # filter rather than the ranking: ranking by overlap with the previous box
    # is self-reinforcing, so one frame that picked the upper half would keep
    # picking it forever. Containment is what lets a half-body lock recover,
    # since the full box wholly contains the half box and so still qualifies.
    same = [p for p in usable if same_person(p[0])] if previous else []
    box, score = max(same or usable, key=lambda p: _area(p[0]))

    distinct = [box]
    for other, _ in sorted(usable, key=lambda p: -_area(p[0])):
        if all(iou(other, kept) < MERGE_IOU
               and containment(other, kept) < MERGE_CONTAINMENT
               for kept in distinct):
            distinct.append(other)
    x0, y0, x1, y1 = box
    return ([x0, y0, x1 - x0, y1 - y0], score, len(distinct),
            len(distinct) > 1, box)


def row_for(frame_index: int, at_s: float, keypoints, scores,
            box: Box, box_score: float, people: int,
            unresolved: bool) -> dict:
    """A posed row: the 21 body points with scores, plus the box that framed
    them. `frame` is whatever cadence the caller counts in - the producer
    writes decode-cadence indices."""
    scores = [float(s) for s in scores]
    return {
        "frame": frame_index,
        "atS": round(at_s, 4),
        "box": [round(v, 1) for v in box],
        "boxScore": round(box_score, 3),
        "people": people,
        "identityUnresolved": unresolved,
        "lowerBodyConfidence": round(
            sum(scores[i] for i in LOWER_BODY) / len(LOWER_BODY), 4
        ),
        "lowerBodyVisible": sum(
            1 for i in LOWER_BODY if scores[i] >= MIN_KEYPOINT_SCORE
        ),
        "keypoints": {
            KEYPOINT_NAMES[i]: [
                round(float(keypoints[i][0]), 2),
                round(float(keypoints[i][1]), 2),
                round(scores[i], 4),
            ]
            for i in BODY
        },
    }


def missing_row(frame_index: int, at_s: float) -> dict:
    """An explicit gap. A skipped frame would silently shorten the series."""
    return {
        "frame": frame_index,
        "atS": round(at_s, 4),
        "box": None,
        "boxScore": None,
        "people": 0,
        "identityUnresolved": False,
        "lowerBodyConfidence": None,
        "lowerBodyVisible": 0,
        "keypoints": None,
    }
