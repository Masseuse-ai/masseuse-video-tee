"""The pose tracker the producer runs on the GPU: RT-DETRv4-X + Sapiens2-1B.

Inference exists in exactly one place - `workload/pixel/live_pose.GpuPose`,
which wraps the `Tracker` below and streams frames through it on a Blackwell
Cloud Run instance. There is no batch or local run mode here any more: the
Mac RT-DETRv2 + fp32/MPS tracker, its VideoToolbox frame extraction and the
6 fps "gold" tracks it wrote were retired in 2026-09 in favour of production
captures (`poses.jsonl`, one file-mode session per reference clip). Every
CPU consumer reads those captures through `tracks.py`; nothing outside the
producer image needs this module or torch.

What stays here is only what needs torch: loading the vendored RT-DETRv4
deploy graph, loading Sapiens2, and the forward passes. Person selection,
scenery rejection, fragment fusion and the row format are in `pose_rows.py`
so they can be tested and reused without a GPU.

Numerics: Sapiens2 loads in bf16 by default (the production dtype; fp32
weights plus autocast recast the 1B model on every forward and missed the
6 fps budget), the detector keeps fp32 master weights under the producer's
CUDA autocast, and flip TTA is off in production - the file-mode
equivalence gate adjudicates the keypoint delta.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

from pose_rows import (
    PERSON_SCORE, RTDETRV4_FRAGMENT_SCORE, TRACK_SCORE,
    _fuse_rtdetrv4_fragments, choose_person,
)

RTDETRV4_ROOT = Path(__file__).resolve().parent / "vendor" / "rtdetrv4"
RTDETRV4_X_CONFIG = (
    RTDETRV4_ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_x_coco.yml"
)
RTDETRV4_X_CHECKPOINT = Path(
    "/models/detectors/rtdetrv4/rtv4_hgnetv2_x_coco.pth"
)
# PostProcessor.deploy() intentionally returns the raw zero-based COCO labels
# before the training/eval-only category-id remap. COCO person is label zero.
RTDETRV4_PERSON_LABEL = 0
POSE_MODEL = "facebook/sapiens2-pose-1b"
DETECTOR_NAME = "rtdetrv4-x"


class _RtdetrV4Detector:
    """Official RT-DETRv4 deploy graph behind the candidate contract."""

    def __init__(
        self,
        device: str,
        config_path: Path = RTDETRV4_X_CONFIG,
        checkpoint_path: Path = RTDETRV4_X_CHECKPOINT,
    ):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import v2 as transforms

        from vendor.rtdetrv4.engine.core.yaml_config import YAMLConfig

        if not config_path.is_file():
            raise FileNotFoundError(f"RT-DETRv4 config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RT-DETRv4 checkpoint not found: {checkpoint_path}"
            )

        cfg = YAMLConfig(str(config_path))
        # The checkpoint supplies the backbone; never let the config attempt
        # the training-only remote pretrained-weight path.
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        if cfg.yaml_cfg.get("num_classes") != 80:
            raise ValueError("RT-DETRv4 checkpoint must use COCO's 80 labels")

        load_started = time.monotonic()
        if checkpoint_path.suffix == ".safetensors":
            # The EMA-only re-export: a quarter of the official pickle's
            # bytes (no optimizer/criterion state) and a zero-copy load.
            from safetensors.torch import load_file

            state = load_file(str(checkpoint_path))
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            state = (
                checkpoint["ema"]["module"]
                if "ema" in checkpoint
                else checkpoint["model"]
            )
        model = cfg.model
        incompatible = model.load_state_dict(state)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "RT-DETRv4 checkpoint/config mismatch: "
                f"missing={incompatible.missing_keys[:5]} "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )

        self.device = device
        # RT-DETRv4's decoder mixes generated FP32 reference points,
        # LayerNorm, and linear layers. Converting every parameter to BF16
        # produces hard mixed-dtype failures. Keep this 62M-parameter model's
        # master weights FP32 and let the producer's CUDA autocast select BF16
        # kernels; unlike 1B-parameter Sapiens2, its recast cost is small.
        self.dtype = torch.float32
        self.model = model.deploy().eval().to(device=device)
        self.postprocessor = cfg.postprocessor.deploy().eval().to(device)
        # Reported as the bootMs `detectorWeights` phase by the producer, so
        # cold-boot regressions name the model that caused them.
        self.boot_seconds = {
            "detectorWeights": time.monotonic() - load_started,
        }
        self.person_label = RTDETRV4_PERSON_LABEL
        self.transform = transforms.Compose([
            transforms.Resize(
                (640, 640),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

    def candidates(
        self, image: Image.Image, threshold: float
    ) -> list[tuple[list[float], float]]:
        pixels = (
            self.transform(image)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        # The vendored deploy postprocessor expects [width, height], then
        # scales normalized cxcywh predictions into original-image xyxy.
        original_size = torch.tensor(
            [[image.width, image.height]],
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            outputs = self.model(pixels)
            labels, boxes, scores = self.postprocessor(outputs, original_size)
        labels = labels[0]
        boxes = boxes[0].float()
        scores = scores[0].float()
        keep = (
            (labels == self.person_label)
            & (scores >= min(threshold, RTDETRV4_FRAGMENT_SCORE))
        )
        raw = [
            ([float(v) for v in box], float(score))
            for box, score in zip(boxes[keep], scores[keep])
        ]
        return _fuse_rtdetrv4_fragments(raw, threshold)


class Tracker:
    """Detector + pose model with identity carried frame to frame.

    Frames must be fed in order: the last accepted box anchors identity, and
    `scenery` (static false positives, learned by the producer from the
    stream's opening seconds) is discarded before selection.
    """

    def __init__(self, device: str, pose_model: str = POSE_MODEL,
                 flip: bool = False,
                 dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoImageProcessor, AutoModelForPoseEstimation

        self.device = device
        self.dtype = dtype
        self.flip = flip
        checkpoint_path = Path(os.environ.get(
            "POSE_DETECTOR_CHECKPOINT", str(RTDETRV4_X_CHECKPOINT)
        ))
        self.detector_backend = _RtdetrV4Detector(
            device, checkpoint_path=checkpoint_path
        )
        self.detector_name = DETECTOR_NAME
        self.pose_processor = AutoImageProcessor.from_pretrained(pose_model)
        self.pose = (
            AutoModelForPoseEstimation.from_pretrained(pose_model, dtype=dtype)
            .eval()
            .to(device)
        )
        self.flip_pairs = (
            torch.tensor(self.pose.config.flip_pairs, device=device)
            if flip else None
        )
        # The last accepted box, which anchors identity across frames.
        self.previous: list[float] | None = None
        # Static false positives to discard, from pose_rows.find_scenery.
        self.scenery: list[list[float]] = []

    def detect(self, image: Image.Image) -> tuple[list[float] | None, float, int, bool]:
        """The tracked person's box in COCO xywh, plus how many distinct people."""
        # Post-processed at the continuity floor rather than at PERSON_SCORE, so
        # a weak box can still be recovered below by its agreement with the
        # previous frame. Anything not so recovered is held to PERSON_SCORE.
        return self.choose(self.detection_candidates(image))

    def detection_candidates(
        self, image: Image.Image
    ) -> list[tuple[list[float], float]]:
        """All person boxes before continuity/scenery selection."""
        return self.detector_backend.candidates(
            image, min(PERSON_SCORE, TRACK_SCORE)
        )

    def choose(
        self, candidates: list[tuple[list[float], float]]
    ) -> tuple[list[float] | None, float, int, bool]:
        """`pose_rows.choose_person` with this tracker's identity state."""
        box, score, people, unresolved, anchor = choose_person(
            candidates, self.previous, self.scenery
        )
        self.previous = anchor
        return box, score, people, unresolved
