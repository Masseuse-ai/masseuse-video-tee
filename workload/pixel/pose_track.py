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
6 fps budget), the detector keeps fp32 master weights under CUDA autocast,
and flip TTA is off in production - the file-mode equivalence gate
adjudicates the keypoint delta.

Per frame, the tracker touches the host twice. The frame arrives as one CHW
uint8 tensor on the device; the detector's resize and the pose crop happen
there, each forward runs with its post-processing as one captured CUDA
graph (`gpu_graph`; eager when capture is unavailable), and each hands back
a single small tensor - 300 detector rows, 308 keypoint rows - copied once.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from pose_rows import (
    PERSON_SCORE, TRACK_SCORE, candidates_from_packed, choose_person,
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
DETECTOR_INPUT = (640, 640)  # height, width: the config's eval_spatial_size


def autocast_for(device: str, dtype: torch.dtype):
    """Autocast for `device`, with the weight-cast cache off: the cache is
    incompatible with CUDA graph capture, and the same context wraps the
    eager twin so both paths run the identical kernels."""
    kind = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(kind, dtype=dtype, cache_enabled=False)


class _RtdetrV4Detector:
    """Official RT-DETRv4 deploy graph behind the candidate contract."""

    def __init__(
        self,
        device: str,
        config_path: Path = RTDETRV4_X_CONFIG,
        checkpoint_path: Path = RTDETRV4_X_CHECKPOINT,
        autocast_dtype: torch.dtype = torch.bfloat16,
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
        self.autocast_dtype = autocast_dtype
        # Applied to the frame as a CHW uint8 tensor already on the device.
        # torchvision's antialiased uint8 resize reproduces the PIL filter
        # the old PIL path used, so the detector sees the same pixels.
        self.transform = transforms.Compose([
            transforms.Resize(
                DETECTOR_INPUT,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        # The postprocessor's frame size: ones, so boxes come back normalised
        # and the real size is applied on the host (see `forward`).
        self._ones = torch.ones((1, 2), device=device, dtype=self.dtype)
        # The forward and the deploy postprocessor as one replayed graph
        # (`capture`); `forward` is the same computation run eagerly.
        self.graph = None

    def autocast(self):
        return autocast_for(self.device, self.autocast_dtype)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """The deploy graph over one 640x640 input, packed for a single copy
        back: `(num_top_queries, 6)` rows of xyxy normalised to the frame
        (the postprocessor's size scaling is linear, so it is fed ones and
        the frame size is applied on the host), score, label."""
        with torch.inference_mode(), self.autocast():
            outputs = self.model(pixels)
            labels, boxes, scores = self.postprocessor(outputs, self._ones)
        return torch.cat([
            boxes[0].float(), scores[0].float()[:, None],
            labels[0].float()[:, None],
        ], dim=-1)

    def capture(self, telemetry=None) -> None:
        """Capture `forward` over a static input (or fall back to eager)."""
        import gpu_graph

        static = torch.zeros(
            (1, 3, *DETECTOR_INPUT), device=self.device, dtype=self.dtype)
        self.graph = gpu_graph.build(
            self.forward, (static,), name="detector", device=self.device,
            telemetry=telemetry)

    def pixels(self, frame: torch.Tensor) -> torch.Tensor:
        """The detector's input for a CHW uint8 frame on the device."""
        return self.transform(frame).unsqueeze(0).to(dtype=self.dtype)

    def candidates(
        self, frame: torch.Tensor, threshold: float
    ) -> list[tuple[list[float], float]]:
        """Person boxes (image xyxy, score) for a CHW uint8 frame on the
        device, fragments fused as `pose_rows` prescribes."""
        if self.graph is None:
            self.capture()
        packed = self.graph.replay(self.pixels(frame)).cpu().tolist()
        height, width = frame.shape[-2:]
        return candidates_from_packed(
            packed, float(width), float(height), self.person_label, threshold)


class Tracker:
    """Detector + pose model with identity carried frame to frame.

    Frames must be fed in order: the last accepted box anchors identity, and
    `scenery` (static false positives, learned by the producer from the
    stream's opening seconds) is discarded before selection.
    """

    def __init__(self, device: str, pose_model: str = POSE_MODEL,
                 flip: bool = False,
                 dtype: torch.dtype = torch.bfloat16,
                 autocast_dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoImageProcessor, AutoModelForPoseEstimation

        self.device = device
        self.dtype = dtype
        self.autocast_dtype = autocast_dtype
        self.flip = flip
        checkpoint_path = Path(os.environ.get(
            "POSE_DETECTOR_CHECKPOINT", str(RTDETRV4_X_CHECKPOINT)
        ))
        self.detector_backend = _RtdetrV4Detector(
            device, checkpoint_path=checkpoint_path,
            autocast_dtype=autocast_dtype,
        )
        self.detector_name = DETECTOR_NAME
        self.pose_processor = AutoImageProcessor.from_pretrained(pose_model)
        self.crop_size = (self.pose_processor.size["height"],
                          self.pose_processor.size["width"])
        self.pose = (
            AutoModelForPoseEstimation.from_pretrained(pose_model, dtype=dtype)
            .eval()
            .to(device)
        )
        self.flip_pairs = (
            torch.tensor(self.pose.config.flip_pairs, device=device)
            if flip else None
        )
        # The pose forward plus its post-processing as one replayed graph,
        # and the heatmap extent it needs; both set by `capture`.
        self.pose_graph = None
        self.heatmap_extent = None
        # The last accepted box, which anchors identity across frames.
        self.previous: list[float] | None = None
        # Static false positives to discard, from pose_rows.find_scenery.
        self.scenery: list[list[float]] = []

    def autocast(self):
        return autocast_for(self.device, self.autocast_dtype)

    @staticmethod
    def frame_tensor(rgb, device: str) -> torch.Tensor:
        """An HWC uint8 RGB frame (the decoder's numpy array) as the CHW
        uint8 tensor on the device every stage reads: one upload per frame,
        the detector's resize and the pose crop derived from it there."""
        return torch.from_numpy(rgb).to(device).permute(2, 0, 1).contiguous()

    def capture(self, telemetry=None) -> None:
        """Capture both graphs over static inputs; each falls back to eager
        on its own if capture fails. Call once after the weights are loaded."""
        import gpu_graph
        import pose_post

        self.detector_backend.capture(telemetry)
        height, width = self.crop_size
        static_crop = torch.zeros(
            (1, 3, height, width), device=self.device, dtype=self.dtype)
        static_box = torch.tensor(
            [0.0, 0.0, float(width), float(height)], device=self.device)
        # One eager forward tells the heatmap shape; the extent tensor must
        # exist before capture (a tensor built inside is a host copy).
        with torch.inference_mode(), self.autocast():
            heatmaps = self.pose(static_crop).heatmaps
        self.heatmap_extent = pose_post.heatmap_extent(
            heatmaps.shape, self.device)
        self.pose_graph = gpu_graph.build(
            self.pose_forward, (static_crop, static_box), name="pose",
            device=self.device, telemetry=telemetry)

    def pose_forward(self, crop: torch.Tensor,
                     box: torch.Tensor) -> torch.Tensor:
        """One forward over the `(1, 3, H, W)` crop and its post-processing:
        `(K, 3)` image-space x, y, score for the person `box` framed."""
        import pose_post

        with torch.inference_mode(), self.autocast():
            heatmaps = self.pose(crop).heatmaps
        return pose_post.keypoints_in_image(
            heatmaps, box, self.heatmap_extent, self.crop_size)

    def pose_forward_tta(self, crop: torch.Tensor,
                         box: torch.Tensor) -> torch.Tensor:
        """The flip-TTA pair as one batched forward, eagerly.

        The model's own flip handling is `flip_back` applied to the flipped
        forward's heatmaps (modeling_sapiens2, applied when `flip_pairs` is
        passed), so one batch-2 forward followed by the same public
        `flip_back` on the second half, averaged with the first as the
        processor averages `outputs_flipped`, is the identical computation.
        """
        import pose_post
        from transformers.models.sapiens2.modeling_sapiens2 import flip_back

        both = torch.cat([crop, crop.flip(-1)])
        with torch.inference_mode(), self.autocast():
            heatmaps = self.pose(both).heatmaps.float()
        merged = (heatmaps[:1] + flip_back(heatmaps[1:], self.flip_pairs)) / 2
        return pose_post.keypoints_in_image(
            merged, box, self.heatmap_extent, self.crop_size)

    def pose_crop(self, frame: torch.Tensor, box: list[float]) -> torch.Tensor:
        """The processor's crop of the CHW uint8 device frame around `box`
        (COCO xywh), rescaled and normalised on the device, in the model
        dtype."""
        inputs = self.pose_processor(
            [frame], boxes=[[box]], return_tensors="pt", device=self.device)
        return inputs["pixel_values"].to(dtype=self.dtype)

    def pose_keypoints(self, frame: torch.Tensor, box: list[float]):
        """`(K, 3)` x, y, score in image pixels for the person in `box`, as
        a numpy array: one copy back per frame."""
        if self.pose_graph is None:
            self.capture()
        crop = self.pose_crop(frame, box)
        box_tensor = torch.tensor(box, dtype=torch.float32, device=self.device)
        if self.flip:
            result = self.pose_forward_tta(crop, box_tensor)
        else:
            result = self.pose_graph.replay(crop, box_tensor)
        return result.cpu().numpy()

    def detect(self, frame: torch.Tensor) -> tuple[list[float] | None, float, int, bool]:
        """The tracked person's box in COCO xywh, plus how many distinct people."""
        # Post-processed at the continuity floor rather than at PERSON_SCORE, so
        # a weak box can still be recovered below by its agreement with the
        # previous frame. Anything not so recovered is held to PERSON_SCORE.
        return self.choose(self.detection_candidates(frame))

    def detection_candidates(
        self, frame: torch.Tensor
    ) -> list[tuple[list[float], float]]:
        """All person boxes before continuity/scenery selection, for a CHW
        uint8 frame on the device."""
        return self.detector_backend.candidates(
            frame, min(PERSON_SCORE, TRACK_SCORE)
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
