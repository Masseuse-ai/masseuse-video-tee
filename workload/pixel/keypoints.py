"""Sapiens2 308-keypoint indices, and the body measures we build from them.

Sapiens2's shipped config labels every keypoint `LABEL_0` through `LABEL_307`,
so the names are not in the weights. They come from the Sapiens2 repository's
own definition, `sapiens/pose/configs/_base_/keypoints308.py`: its
`dataset_info` (dataset "goliath", the Sociopticon format) lists 344 points and
drops the 36 teeth points to make 308, and the model card says predictions
follow that file. The file opens with a COCO-WholeBody block (wrists at 9 and
10, hips at 11 and 12, feet at 17-22) that is only there to copy sigmas from;
an earlier version of this module read that block as the definition and
concluded the file did not match the model.

Checked three ways on 2026-09-17. The 146 `flip_pairs` in the shipped 0.4B and
1B configs are exactly the swap pairs of that definition. Both models, run on a
camera-facing portrait and on a skier, put `tip_of_nose` on `nose`, each iris
centre on its eye, each ear's 26 points on its ear, each wrist by its own elbow
with its twenty finger points around it, each foot's three points by its own
ankle, the acromions on the shoulders, and every `left_` point on the subject's
left (the image's right). And the body block matched this module's names point
for point.

    0        nose
    1-14     eyes, ears, shoulders, elbows, hips, knees, ankles (left, right)
    15-17    left foot: big toe, small toe, heel
    18-20    right foot: big toe, small toe, heel
    21-41    right hand: tip, distal, middle, base of each finger, thumb to
             little finger, then the wrist (41)
    42-62    left hand, the same order, wrist at 62
    63-69    olecranon, cubital fossa, acromion (left, right), neck
    70-307   face: midline, eyebrows, eyelids, nose, lips, ears, iris, pupil

`ALL_NAMES` is the full order. The pupil channels (290-307) were flat on both
models in that check (peak scores 0.08-0.15, no localized maximum), so they
read as guesses under `MIN_KEYPOINT_SCORE`; the iris ring (272-289) was sharp.

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

# The whole body block. Index 63 onward is the arm, neck and face points, which
# do not survive a prone subject shot from across a room with the face in a
# pillow.
BODY = tuple(range(21))

# The two hands, twenty-one points each: for every finger its tip (`4`), distal
# (`3`) and middle (`2`) joints and its base (`_third_joint`), thumb to little
# finger, then the wrist last. The right hand comes first in the model's order.
# Stored separately from BODY because the production producer writes only BODY
# to poses.jsonl (pose_rows.row_for), so a consumer has to be able to tell
# "this capture has no hand data" from "the hands were not visible".
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
RIGHT_HAND = tuple(range(21, 42))
LEFT_HAND = tuple(range(42, 63))
HANDS = RIGHT_HAND + LEFT_HAND

# The root of each block is its wrist, the block's last index. Flip pairs
# cannot tell the two blocks apart (they are symmetric); the definition's
# sides and the skier check above can, and both put 21-41 on the right arm.
RIGHT_WRIST = RIGHT_HAND[-1]
LEFT_WRIST = LEFT_HAND[-1]
RIGHT_HAND_ROOT = RIGHT_WRIST
LEFT_HAND_ROOT = LEFT_WRIST

# The dense block after the hands: seven arm and neck points, then the face.
ARM_AND_NECK = tuple(range(63, 70))
FACE = tuple(range(70, 308))
# Each: left centre and eight border points, then the right eye's.
IRIS = tuple(range(272, 290))
PUPIL = tuple(range(290, 308))

# Everything the signals touch. Reading confidence over just these avoids the
# 245 arm, neck and face keypoints dragging the average around when the person
# is face down and the face is, correctly, invisible.
LOWER_BODY = (
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_BIG_TOE, RIGHT_BIG_TOE,
    LEFT_SMALL_TOE, RIGHT_SMALL_TOE,
    LEFT_HEEL, RIGHT_HEEL,
)

TORSO = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# Every index by name, in the model's order: the Sapiens2 `keypoints308.py`
# definition with its teeth points removed (the module docstring says how
# the order was checked against the shipped models).
ALL_NAMES = (
    # 0-20: body: nose, eyes, ears, shoulders, elbows, hips, knees, ankles,
    # feet
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel",
    # 21-41: right hand: per finger tip (4), distal (3), middle (2), base
    # (third joint); thumb to little finger; then the wrist
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2",
    "right_forefinger_third_joint", "right_middle_finger4",
    "right_middle_finger3", "right_middle_finger2",
    "right_middle_finger_third_joint", "right_ring_finger4",
    "right_ring_finger3", "right_ring_finger2",
    "right_ring_finger_third_joint", "right_pinky_finger4",
    "right_pinky_finger3", "right_pinky_finger2",
    "right_pinky_finger_third_joint", "right_wrist",
    # 42-62: left hand, the same order
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2",
    "left_forefinger_third_joint", "left_middle_finger4",
    "left_middle_finger3", "left_middle_finger2",
    "left_middle_finger_third_joint", "left_ring_finger4",
    "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2",
    "left_pinky_finger_third_joint", "left_wrist",
    # 63-69: arm and neck: olecranon, cubital fossa, acromion, neck
    "left_olecranon", "right_olecranon", "left_cubital_fossa",
    "right_cubital_fossa", "left_acromion", "right_acromion", "neck",
    # 70-77: face midline: glabella, nose root and bridge, labiomental
    # groove, chin
    "center_of_glabella", "center_of_nose_root", "tip_of_nose_bridge",
    "midpoint_1_of_nose_bridge", "midpoint_2_of_nose_bridge",
    "midpoint_3_of_nose_bridge", "center_of_labiomental_groove",
    "tip_of_chin",
    # 78-95: eyebrows, right then left
    "upper_startpoint_of_r_eyebrow", "lower_startpoint_of_r_eyebrow",
    "end_of_r_eyebrow", "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow", "upper_midpoint_2_of_r_eyebrow",
    "upper_midpoint_3_of_r_eyebrow", "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow", "upper_startpoint_of_l_eyebrow",
    "lower_startpoint_of_l_eyebrow", "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow", "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow", "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow", "lower_midpoint_3_of_l_eyebrow",
    # 96-143: upper eyelids: lash line, eyelid line, crease line; left then
    # right
    "l_inner_end_of_upper_lash_line", "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line", "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line", "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line", "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line", "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line", "l_midpoint_2_of_upper_eyelid_line",
    "l_midpoint_5_of_upper_eyelid_line", "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line", "l_midpoint_1_of_upper_eyelid_line",
    "l_midpoint_3_of_upper_eyelid_line", "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line", "l_midpoint_5_of_upper_crease_line",
    "l_centerpoint_of_upper_crease_line", "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line", "l_midpoint_3_of_upper_crease_line",
    "r_inner_end_of_upper_lash_line", "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line", "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line", "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line", "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line", "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line", "r_midpoint_1_of_upper_eyelid_line",
    "r_midpoint_4_of_upper_eyelid_line", "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line", "r_midpoint_2_of_upper_eyelid_line",
    "r_midpoint_6_of_upper_eyelid_line", "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line", "r_midpoint_4_of_upper_crease_line",
    "r_centerpoint_of_upper_crease_line", "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line", "r_midpoint_6_of_upper_crease_line",
    # 144-177: lower eyelids: lash line, eyelid line; left then right
    "l_inner_end_of_lower_lash_line", "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line", "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line", "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line", "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line", "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line", "l_midpoint_2_of_lower_eyelid_line",
    "l_midpoint_5_of_lower_eyelid_line", "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line", "l_midpoint_1_of_lower_eyelid_line",
    "l_midpoint_3_of_lower_eyelid_line", "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line", "r_centerpoint_of_lower_lash_line",
    "r_midpoint_1_of_lower_lash_line", "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line", "r_midpoint_4_of_lower_lash_line",
    "r_midpoint_5_of_lower_lash_line", "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line", "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line", "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line", "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line", "r_midpoint_6_of_lower_eyelid_line",
    # 178-187: nose: tip, base, outer corners, nostrils
    "tip_of_nose", "bottom_center_of_nose", "r_outer_corner_of_nose",
    "l_outer_corner_of_nose", "inner_corner_of_r_nostril",
    "outer_corner_of_r_nostril", "upper_corner_of_r_nostril",
    "inner_corner_of_l_nostril", "outer_corner_of_l_nostril",
    "upper_corner_of_l_nostril",
    # 188-203: outer lip contour
    "r_outer_corner_of_mouth", "l_outer_corner_of_mouth",
    "center_of_cupid_bow", "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip", "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip", "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip",
    # 204-219: inner lip contour
    "r_inner_corner_of_mouth", "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip", "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip", "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip", "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip", "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip", "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip", "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip", "midpoint_6_of_lower_inner_lip",
    # 220-271: ears, left then right
    "l_top_end_of_inferior_crus", "l_top_end_of_superior_crus",
    "l_start_of_antihelix", "l_end_of_antihelix", "l_midpoint_1_of_antihelix",
    "l_midpoint_1_of_inferior_crus", "l_midpoint_2_of_antihelix",
    "l_midpoint_3_of_antihelix", "l_point_1_of_inner_helix",
    "l_point_2_of_inner_helix", "l_point_3_of_inner_helix",
    "l_point_4_of_inner_helix", "l_point_5_of_inner_helix",
    "l_point_6_of_inner_helix", "l_point_7_of_inner_helix",
    "l_highest_point_of_antitragus", "l_bottom_point_of_tragus",
    "l_protruding_point_of_tragus", "l_top_point_of_tragus",
    "l_start_point_of_crus_of_helix", "l_deepest_point_of_concha",
    "l_tip_of_ear_lobe", "l_midpoint_between_22_15",
    "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix", "l_point_8_of_inner_helix",
    "r_top_end_of_inferior_crus", "r_top_end_of_superior_crus",
    "r_start_of_antihelix", "r_end_of_antihelix", "r_midpoint_1_of_antihelix",
    "r_midpoint_1_of_inferior_crus", "r_midpoint_2_of_antihelix",
    "r_midpoint_3_of_antihelix", "r_point_1_of_inner_helix",
    "r_point_8_of_inner_helix", "r_point_3_of_inner_helix",
    "r_point_4_of_inner_helix", "r_point_5_of_inner_helix",
    "r_point_6_of_inner_helix", "r_point_7_of_inner_helix",
    "r_highest_point_of_antitragus", "r_bottom_point_of_tragus",
    "r_protruding_point_of_tragus", "r_top_point_of_tragus",
    "r_start_point_of_crus_of_helix", "r_deepest_point_of_concha",
    "r_tip_of_ear_lobe", "r_midpoint_between_22_15",
    "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix", "r_point_2_of_inner_helix",
    # 272-289: iris: centre and eight border points, left then right
    "l_center_of_iris", "l_border_of_iris_3", "l_border_of_iris_midpoint_1",
    "l_border_of_iris_12", "l_border_of_iris_midpoint_4",
    "l_border_of_iris_9", "l_border_of_iris_midpoint_3", "l_border_of_iris_6",
    "l_border_of_iris_midpoint_2", "r_center_of_iris", "r_border_of_iris_3",
    "r_border_of_iris_midpoint_1", "r_border_of_iris_12",
    "r_border_of_iris_midpoint_4", "r_border_of_iris_9",
    "r_border_of_iris_midpoint_3", "r_border_of_iris_6",
    "r_border_of_iris_midpoint_2",
    # 290-307: pupil: centre and eight border points, left then right
    "l_center_of_pupil", "l_border_of_pupil_3",
    "l_border_of_pupil_midpoint_1", "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4", "l_border_of_pupil_9",
    "l_border_of_pupil_midpoint_3", "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2", "r_center_of_pupil",
    "r_border_of_pupil_3", "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12", "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9", "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6", "r_border_of_pupil_midpoint_2",
)
assert len(ALL_NAMES) == 308

# The body block by name: what pose_rows.row_for writes and the analysis reads.
KEYPOINT_NAMES = {i: ALL_NAMES[i] for i in BODY}

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
