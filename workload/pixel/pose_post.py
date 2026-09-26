"""Sapiens2 heatmaps to image-space keypoints, as one tensor.

The same computation as `Sapiens2ImageProcessor.post_process_pose_estimation`
- the heatmap argmax, the UDP-DARK sub-pixel refinement and the remap from
crop to image coordinates - using the processor's own functions so the
numbers are the pipeline's, arranged so it can sit inside a CUDA graph:

- the box arrives as a tensor (a static graph input) rather than being
  built from a Python list per call, which is a host-to-device copy capture
  rejects;
- the heatmap extent is a tensor made once before capture, for the same
  reason (`heatmap_extent`);
- the result is a single `(num_keypoints, 3)` tensor of x, y, score, so
  one device-to-host copy per frame brings everything back. The producer
  used to hand CUDA tensors to the row builder, which read them one
  element at a time: some 350 tiny synchronous copies per pose frame.

`keypoints_in_image_batch` is the same over B crops at once (`(B, K, 3)`),
for a graph captured over a batch; the production path is its batch of one.
"""

from __future__ import annotations

import torch

DARK_KERNEL = 11  # the processor's default UDP-DARK blur


def heatmap_extent(heatmap_shape, device) -> torch.Tensor:
    """`[width - 1, height - 1]` of the model's heatmaps, the divisor that
    maps heatmap pixels onto the crop window (the processor's `heatmap_size`)."""
    height, width = heatmap_shape[-2], heatmap_shape[-1]
    return torch.tensor([width - 1, height - 1], dtype=torch.float32,
                        device=device)


def keypoints_in_image_batch(heatmaps: torch.Tensor, boxes_xywh: torch.Tensor,
                             extent: torch.Tensor, crop_size: tuple[int, int],
                             kernel_size: int = DARK_KERNEL) -> torch.Tensor:
    """`(B, K, H, W)` heatmaps for B people, each framed by its row of
    `boxes_xywh` (`(B, 4)`, COCO x, y, width, height in image pixels), as
    `(B, K, 3)` image-space x, y, score.

    The processor's functions are batched over their first dimension, so
    this is their computation with B crops in flight at once: what a graph
    captured over a batch of crops replays. `crop_size` is the processor's
    `(height, width)`; `extent` comes from `heatmap_extent`.
    """
    from transformers.models.sapiens2.image_processing_sapiens2 import (
        box_xywh_to_cxcywh, boxes_to_crop_params, get_keypoint_predictions,
        post_dark_unbiased_data_processing,
    )

    # float32 as the processor does: the refinement's log/Hessian arithmetic
    # is not meant for bf16.
    heatmaps = heatmaps.float()
    keypoints, scores = get_keypoint_predictions(heatmaps)
    keypoints = post_dark_unbiased_data_processing(
        keypoints=keypoints, heatmaps=heatmaps, blur_kernel_size=kernel_size)
    centers, scales = boxes_to_crop_params(
        box_xywh_to_cxcywh(boxes_xywh.float()), output_size=crop_size)
    keypoints = (keypoints / extent * scales[:, None, :]
                 + centers[:, None, :] - 0.5 * scales[:, None, :])
    return torch.cat([keypoints, scores[..., None]], dim=-1)


def keypoints_in_image(heatmaps: torch.Tensor, box_xywh: torch.Tensor,
                       extent: torch.Tensor, crop_size: tuple[int, int],
                       kernel_size: int = DARK_KERNEL) -> torch.Tensor:
    """`(1, K, H, W)` heatmaps for one person framed by `box_xywh` (COCO
    x, y, width, height in image pixels), as `(K, 3)` image-space x, y, score:
    the batch of one."""
    return keypoints_in_image_batch(
        heatmaps, box_xywh[None], extent, crop_size, kernel_size)[0]
