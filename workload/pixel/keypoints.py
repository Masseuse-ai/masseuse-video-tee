"""Sociopticon 308-keypoint indices, and the body measures we build from them.

Sapiens2's shipped config labels every keypoint `LABEL_0` through `LABEL_307`,
so the names are not in the weights and have to come from somewhere else. They
do NOT come from the Goliath definition in the upstream repo
(`sapiens/pose/configs/_base_/keypoints308.py`), which is what an earlier
version of this file claimed: that file lists wrists at 9 and 10, hips at 11
and 12, and each foot's three points at 17-19 and 20-22, none of which is what
the shipped model emits.

The ordering below is recovered from `flip_pairs` in the model's own config,
which is authoritative because it ships with the weights. Those pairs run
1<->2, 3<->4 ... 13<->14 one apart, then 15<->18, 16<->19, 17<->20 three apart,
then 21<->42 through 41<->62 twenty-one apart. Only one layout produces that:
seven consecutive left/right pairs of two, then two blocks of three for the
feet, then two blocks of twenty-one for the hands. So there are no wrists in the
body block at all, and the foot points are grouped per foot rather than per
landmark. Index 0 is the only unpaired point below 63, which is the nose.

    0        nose
    1-14     eyes, ears, shoulders, elbows, hips, knees, ankles
    15-17    left foot: big toe, small toe, heel
    18-20    right foot: big toe, small toe, heel
    21-41    left hand
    42-62    right hand
    63-307   face

Confirmed against the footage independently: the stored hip pair holds an 87px
separation with a 3% coefficient of variation across 132 seconds, which two
free hands cannot do and a pelvis does by construction.

All four target signals are measurable from this handful:

    feet elevation   ankles and heels against the bed plane
    leg spread       ankle and knee separation, scaled by hip width
    hip roll/rearing pelvis height and tilt against the shoulder line
    toe curl         heel-to-toe vector against the shank

The readings the analysis derives from the hip region are deliberately absent:
no keypoint moves with them, so they cannot be recovered here at any model size
or resolution. They come from the regional motion descriptors (motion.py)
instead.
"""

from __future__ import annotations

from dataclasses import dataclass

NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_HIP, RIGHT_HIP = 9, 10
LEFT_KNEE, RIGHT_KNEE = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 13, 14
LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL = 15, 16, 17
RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL = 18, 19, 20

# The whole body block. Index 63 onward is the 245 face keypoints, which do not
# survive a prone subject shot from across a room with the face in a pillow.
BODY = tuple(range(21))

# The two hands, twenty-one points each. Stored separately from BODY because
# the production producer writes only BODY to poses.jsonl (pose_rows.row_for),
# so a consumer has to be able to tell "this capture has no hand data" from
# "the hands were not visible".
#
# Hands matter for one specific reason: a reach for the controller to turn it
# off is a deliberate act rather than a symptom. Feet are the opposite - the
# person raises and lowers them constantly through a session - which is why
# feet are excluded from the postural anchors.
#
# Defined, and on the 2026-05-30 reference clip found not usable in 2026-08 on
# the Mac RT-DETRv2/fp32 pipeline of the time (retired 2026-09-02): the model
# did not put these points on the hands - through the one reach they scored
# 0.10 to 0.14, and where they scored well, as the left block did at 0.69
# during setup, they were sitting on the lower back. The hand itself is
# visible in that footage, 45x53 pixels with separate fingers, resting on the
# controller for the opening minute, so that was the model declining to find a
# hand rather than a hand too small to find. Whether the production RT-DETRv4-X
# crop changes the answer is open and belongs to the hand-keypoint work on
# production rows; until then the analysis stands in with the elbow.
LEFT_HAND = tuple(range(21, 42))
RIGHT_HAND = tuple(range(42, 63))
HANDS = LEFT_HAND + RIGHT_HAND

# Within each hand block the root is index 0 of that block, on the standard
# 21-point topology of root plus four joints for each of five fingers. Treated
# as a hypothesis rather than a fact until checked against elbow proximity,
# since the block's internal order is not pinned down by flip_pairs.
LEFT_HAND_ROOT = LEFT_HAND[0]
RIGHT_HAND_ROOT = RIGHT_HAND[0]

# Everything the signals touch. Reading confidence over just these avoids the
# 274 face keypoints dragging the average around when the person is face down
# and the face is, correctly, invisible.
LOWER_BODY = (
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_BIG_TOE, RIGHT_BIG_TOE,
    LEFT_SMALL_TOE, RIGHT_SMALL_TOE,
    LEFT_HEEL, RIGHT_HEEL,
)

TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

KEYPOINT_NAMES = {
    NOSE: "nose",
    LEFT_EYE: "left_eye", RIGHT_EYE: "right_eye",
    LEFT_EAR: "left_ear", RIGHT_EAR: "right_ear",
    LEFT_SHOULDER: "left_shoulder", RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow", RIGHT_ELBOW: "right_elbow",
    LEFT_HIP: "left_hip", RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee", RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle", RIGHT_ANKLE: "right_ankle",
    LEFT_BIG_TOE: "left_big_toe", LEFT_SMALL_TOE: "left_small_toe",
    LEFT_HEEL: "left_heel",
    RIGHT_BIG_TOE: "right_big_toe", RIGHT_SMALL_TOE: "right_small_toe",
    RIGHT_HEEL: "right_heel",
}

# Below this a keypoint is a guess. Sapiens emits a coordinate for every
# keypoint whether or not it can see one, so an unfiltered track looks
# continuous and confident while describing a limb that left frame.
MIN_KEYPOINT_SCORE = 0.3


@dataclass(frozen=True)
class BodyMeasures:
    """One frame of the person, in units of their own hip width.

    Every distance is divided by hip width rather than left in pixels. The body
    moves toward and away from a fixed camera, so raw pixels conflate "the feet
    came up" with "the body shifted down the bed"; hip width is the most stable
    thing on a prone body - it barely changes with the movements we are
    measuring, unlike torso length, which foreshortens the moment the body
    rears up.

    `feet_up` and `hips_reared` are each measured against their own landmark's
    rest line (the `plane` given to `measure`), which is what keeps them
    independent.
    `hip_vs_shoulder` is kept alongside because it measures pelvic tilt without
    depending on the fit at all, so a bad rest line shows up as the two
    disagreeing.
    """

    feet_up: float | None
    legs_spread: float | None
    hips_reared: float | None
    toe_curl: float | None
    hip_vs_shoulder: float | None
    hip_width_px: float | None
    confidence: float
    missing: tuple[str, ...]


def _point(keypoints, scores, index: int):
    if scores[index] < MIN_KEYPOINT_SCORE:
        return None
    return float(keypoints[index][0]), float(keypoints[index][1])


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _distance(a, b) -> float | None:
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def measure(keypoints, scores, plane=None) -> BodyMeasures:
    """Reduce one frame's keypoints to the body signals.

    `plane` is a rest-line fit answering `elevation(name, x, y)` per landmark
    (None where it has no line for one). Without one, elevation has no fixed
    reference and both `feet_up` and `hips_reared` are returned as None rather
    than silently falling back to measuring each against the other.
    """
    get = lambda i: _point(keypoints, scores, i)  # noqa: E731

    l_hip, r_hip = get(LEFT_HIP), get(RIGHT_HIP)
    l_knee, r_knee = get(LEFT_KNEE), get(RIGHT_KNEE)
    l_ankle, r_ankle = get(LEFT_ANKLE), get(RIGHT_ANKLE)
    l_heel, r_heel = get(LEFT_HEEL), get(RIGHT_HEEL)
    l_toe, r_toe = get(LEFT_BIG_TOE), get(RIGHT_BIG_TOE)
    l_sh, r_sh = get(LEFT_SHOULDER), get(RIGHT_SHOULDER)

    missing = tuple(
        KEYPOINT_NAMES[i] for i in LOWER_BODY if scores[i] < MIN_KEYPOINT_SCORE
    )
    confidence = float(sum(scores[i] for i in LOWER_BODY) / len(LOWER_BODY))

    hip_width = _distance(l_hip, r_hip)
    if not hip_width:
        # Without a scale nothing below is comparable across frames, and
        # reporting unnormalized pixels would silently mix the two.
        return BodyMeasures(
            None, None, None, None, None, hip_width, confidence, missing
        )

    hip_mid = _midpoint(l_hip, r_hip)
    shoulder_mid = _midpoint(l_sh, r_sh)

    # Heels register a lift earlier than ankles when the knee stays down, so
    # take whichever foot landmark is highest rather than the ankle alone. Each
    # is measured against its own rest line, so one foot habitually riding
    # higher than the other does not read as a permanent lift.
    feet = [
        ("left_ankle", l_ankle), ("right_ankle", r_ankle),
        ("left_heel", l_heel), ("right_heel", r_heel),
    ]
    feet_up = None
    if plane is not None:
        lifts = [
            lift for name, point in feet if point
            if (lift := plane.elevation(name, *point)) is not None
        ]
        if lifts:
            feet_up = max(lifts) / hip_width

    spread = _distance(l_ankle, r_ankle) or _distance(l_knee, r_knee)
    legs_spread = spread / hip_width if spread else None

    # Averaged over the two hips rather than taken at the midpoint, because each
    # hip has its own rest line and the midpoint has none.
    hips_reared = None
    if plane is not None:
        lifts = [
            lift for name, point in (("left_hip", l_hip), ("right_hip", r_hip))
            if point
            if (lift := plane.elevation(name, *point)) is not None
        ]
        if lifts:
            hips_reared = sum(lifts) / len(lifts) / hip_width

    hip_vs_shoulder = None
    if hip_mid and shoulder_mid:
        hip_vs_shoulder = (shoulder_mid[1] - hip_mid[1]) / hip_width

    toe_curl = _plantar_flexion(
        [(l_knee, l_ankle, l_heel, l_toe), (r_knee, r_ankle, r_heel, r_toe)]
    )

    return BodyMeasures(
        feet_up=feet_up,
        legs_spread=legs_spread,
        hips_reared=hips_reared,
        toe_curl=toe_curl,
        hip_vs_shoulder=hip_vs_shoulder,
        hip_width_px=hip_width,
        confidence=confidence,
        missing=missing,
    )


def _plantar_flexion(legs) -> float | None:
    """Angle between shank and foot, averaged over whichever feet are visible.

    Reported in degrees away from a right angle, so a relaxed foot sits near
    zero and a pointed one grows positive.
    """
    import math

    angles = []
    for knee, ankle, heel, toe in legs:
        if not (knee and ankle and heel and toe):
            continue
        shank = (ankle[0] - knee[0], ankle[1] - knee[1])
        foot = (toe[0] - heel[0], toe[1] - heel[1])
        shank_len = math.hypot(*shank)
        foot_len = math.hypot(*foot)
        if shank_len < 1e-6 or foot_len < 1e-6:
            continue
        cosine = (shank[0] * foot[0] + shank[1] * foot[1]) / (shank_len * foot_len)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) - 90.0)
    return sum(angles) / len(angles) if angles else None
