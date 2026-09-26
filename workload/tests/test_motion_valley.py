"""The midline valley: a synthetic strip with one dark groove, whose planted
width, position and tilt the measurement has to read back, and whose
narrowing it has to read as narrower. Everything else in `motion.py` is a
statistic over pixels; these tests hold the one shape statistic to what
was drawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WORKLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOAD / "pixel"))

import motion  # noqa: E402


def grooved(width_hw: float = 0.16, offset_hw: float = 0.10,
            slope: float = 0.0, depth: float = 0.5,
            base: float = 160.0) -> np.ndarray:
    """A strip with one dark axial groove: a lateral Gaussian valley whose
    centre may sit off the midline and tilt with the rows, over a flat
    surface of brightness `base`. `width_hw` is its full width at half
    depth, so the measurement can be checked against what was planted."""
    rows = np.arange(motion.ROWS, dtype=np.float32)[:, None]
    columns = np.arange(motion.COLUMNS, dtype=np.float32)[None, :]
    centre = (motion.COLUMNS / 2 + offset_hw * motion.SCALE
              + slope * (rows - motion.row_for(0.15)))
    sigma = width_hw * motion.SCALE / (2 * np.sqrt(2 * np.log(2)))
    valley = np.exp(-0.5 * ((columns - centre) / sigma) ** 2)
    return (base - base * depth * valley).astype(np.float32)


def test_the_axis_is_found_where_the_groove_was_planted():
    for offset, slope in ((0.0, 0.0), (0.12, 0.0), (-0.20, 0.25), (0.30, -0.1)):
        axis = motion.fit_midline_axis(grooved(offset_hw=offset, slope=slope))
        assert axis.offset == pytest.approx(offset, abs=0.01), (offset, slope)
        assert axis.slope == pytest.approx(slope, abs=0.02), (offset, slope)
        assert axis.depth == pytest.approx(0.5, abs=0.05)
        assert axis.spread < 0.01
        assert axis.usable


def test_a_flat_strip_has_no_usable_axis():
    flat = np.full((motion.ROWS, motion.COLUMNS), 150.0, np.float32)
    rng = np.random.default_rng(3)
    speckled = flat + rng.normal(0, 2.0, flat.shape).astype(np.float32)
    axis = motion.fit_midline_axis(speckled)
    assert axis.depth < motion.VALLEY_MIN_DEPTH
    assert not axis.usable


def test_the_width_reads_the_planted_groove_and_narrows_with_it():
    wide = grooved(width_hw=0.20, slope=0.2)
    narrow = grooved(width_hw=0.13, slope=0.2)
    axis = motion.fit_midline_axis(wide)
    w_wide, a_wide, d_wide, l_wide = motion.midline_valley(wide, axis)
    w_narrow, a_narrow, _, _ = motion.midline_valley(narrow, axis)
    assert w_wide == pytest.approx(0.20, abs=0.02)
    assert w_narrow == pytest.approx(0.13, abs=0.02)
    assert a_narrow < a_wide
    assert d_wide == pytest.approx(0.5, abs=0.03)
    assert l_wide == pytest.approx(0.5, abs=0.03)


def test_the_width_survives_a_lateral_registration_wobble():
    axis = motion.fit_midline_axis(grooved(width_hw=0.16))
    steady = motion.midline_valley(grooved(width_hw=0.16), axis)[0]
    for shift in (-0.05, -0.02, 0.02, 0.05):
        wobbled = motion.midline_valley(
            grooved(width_hw=0.16, offset_hw=0.10 + shift), axis)[0]
        assert wobbled == pytest.approx(steady, abs=0.01), shift


def test_the_tracker_warms_up_then_measures_and_refits():
    tracker = motion.MidlineTracker(fps=10.0)
    strips = [grooved(width_hw=0.16 + 0.02 * np.sin(i)) for i in range(60)]
    seen = []
    for index, current in enumerate(strips):
        valley, axis = tracker.update(current, current)
        if index < tracker.warmup_frames - 1:
            assert valley is None and axis is None
        else:
            assert valley is not None and axis is not None and axis.usable
            seen.append(valley[0])
    assert len(seen) == 60 - tracker.warmup_frames + 1
    planted = [0.16 + 0.02 * np.sin(i)
               for i in range(tracker.warmup_frames - 1, 60)]
    assert np.corrcoef(seen, planted)[0, 1] > 0.95


def test_the_descriptor_carries_the_valley_and_its_absence():
    current = grooved()
    axis = motion.fit_midline_axis(current)
    valley = motion.midline_valley(current, axis)
    with_valley = motion.describe(current, None, midline_valley=valley,
                                  midline_axis=axis).as_json()
    assert with_valley["midlineValley"] == list(valley)
    assert set(with_valley["midlineAxis"]) == {"offset", "slope", "depth",
                                               "spread"}
    without = motion.describe(current, None).as_json()
    assert without["midlineValley"] is None and without["midlineAxis"] is None
