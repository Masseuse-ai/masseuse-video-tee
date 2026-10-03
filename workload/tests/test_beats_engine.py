"""The second tagger with torch and its real checkpoint.

The release's smoke stage runs this inside the built image
(workload/tee/buildx.sh, SMOKE_TESTS), where the checkpoint is at
BEATS_MODEL_PATH; elsewhere it is skipped without torch or the file. What
is checked: the checkpoint loads into the vendored model with its 527
classes by AudioSet id and a 768-value embedding; the vendored filterbank
gives silence its closed form (every log energy at the float floor, one
value after the fine-tuning's normalisation); tagging is deterministic,
bounded and tells a tone from silence; a window of the wrong length is
refused rather than padded behind the stage's back.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beats import DEFAULT_MODEL, FBANK_MEAN, FBANK_STD, BeatsEngine  # noqa: E402

MODEL = Path(os.environ.get("BEATS_MODEL_PATH", DEFAULT_MODEL))
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="no BEATs checkpoint here")


@pytest.fixture(scope="module")
def engine():
    return BeatsEngine(MODEL, window_s=2.0, device="cpu", threads=None)


def test_the_checkpoint_loads_with_its_classes_and_embedding(engine):
    assert engine.class_count == 527 and len(set(engine.class_ids)) == 527
    assert all(class_id.startswith("/") for class_id in engine.class_ids)
    assert engine.embedding_size == 768
    assert engine.window_samples == 32_000
    assert engine.graphed is False and engine.device == "cpu"
    assert engine.version == f"unilm-beats:{MODEL.name}"


def test_the_filterbank_of_silence_is_its_closed_form(engine):
    fbank = engine.features(np.zeros(engine.window_samples, dtype=np.float32))
    # 25 ms frames every 10 ms over 2 s, edges snipped; 128 mel bins.
    assert tuple(fbank.shape) == (1, 1, 198, 128)
    floor = (math.log(float(np.finfo(np.float32).eps)) - FBANK_MEAN) / (2 * FBANK_STD)
    np.testing.assert_allclose(fbank.numpy(), floor, rtol=1e-6)


def test_tagging_is_deterministic_bounded_and_hears_a_tone(engine):
    t = np.arange(engine.window_samples, dtype=np.float32) / 16_000
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    scores, embedding = engine.tag(tone)
    again_scores, again_embedding = engine.tag(tone)
    assert scores.shape == (527,) and embedding.shape == (768,)
    assert scores.dtype == np.float32 and embedding.dtype == np.float32
    assert np.all((scores >= 0.0) & (scores <= 1.0)) and np.all(np.isfinite(embedding))
    np.testing.assert_array_equal(scores, again_scores)
    np.testing.assert_array_equal(embedding, again_embedding)
    quiet_scores, quiet_embedding = engine.tag(np.zeros(engine.window_samples, dtype=np.float32))
    assert np.max(np.abs(scores - quiet_scores)) > 0.05
    assert np.linalg.norm(embedding - quiet_embedding) > 0.1


def test_a_window_of_the_wrong_length_is_refused(engine):
    with pytest.raises(ValueError):
        engine.tag(np.zeros(16_000, dtype=np.float32))
