"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

"""Inference registrations required by the frozen RT-DETRv4-X config."""

from .common import FrozenBatchNorm2d, freeze_batch_norm2d, get_activation
from .hgnetv2 import HGNetv2

__all__ = [
    "FrozenBatchNorm2d",
    "HGNetv2",
    "freeze_batch_norm2d",
    "get_activation",
]
