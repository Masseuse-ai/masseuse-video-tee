"""Inference-only registration surface for the vendored RT-DETRv4 engine.

Upstream imports its optimizer and COCO training stack here for registration.
The punishment producer only constructs the deployed detector, and importing
the training stack would make faster-coco-eval and dataset tooling production
dependencies. Keep the runtime to the five classes referenced by the frozen
RT-DETRv4 config. The implementation below those classes is otherwise the
upstream tree pinned in ``UPSTREAM_COMMIT``.
"""

from .backbone.hgnetv2 import HGNetv2
from .rtv4 import (
    DFINETransformer,
    HybridEncoder,
    PostProcessor,
    RTv4,
)

__all__ = [
    "DFINETransformer",
    "HGNetv2",
    "HybridEncoder",
    "PostProcessor",
    "RTv4",
]