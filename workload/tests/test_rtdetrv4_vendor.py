"""Regression checks for the inference-only RT-DETRv4 vendor patches.

torch is a producer-image dependency, not part of the CPU analysis set, so
this file skips where it is absent rather than failing the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixel"))

from vendor.rtdetrv4.engine.rtv4 import HybridEncoder  # noqa: E402


def test_precomputed_position_embedding_follows_model_device_and_dtype():
    encoder = HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        use_encoder_idx=[2],
        eval_spatial_size=[64, 64],
        depth_mult=0.33,
    )

    assert "pos_embed2" in dict(encoder.named_buffers())
    assert "pos_embed2" not in encoder.state_dict()
    assert encoder.pos_embed2.dtype == torch.float32

    encoder.to(dtype=torch.float64)
    assert encoder.pos_embed2.dtype == torch.float64
