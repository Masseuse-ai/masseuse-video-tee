"""`pose_post.keypoints_in_image` is the processor's post-processing, packed.

torch and transformers are producer-image dependencies, not part of the CPU
analysis set, so this file skips where they are absent (the CI runner) and
runs inside the built image (the release smoke test), where the versions
are the ones the enclave ships.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sapiens2 = pytest.importorskip(
    "transformers.models.sapiens2.image_processing_sapiens2")

import pose_post  # noqa: E402

KEYPOINTS, HEIGHT, WIDTH = 308, 256, 192
CROP = (1024, 768)


def _heatmaps(seed: int) -> torch.Tensor:
    """Random heatmaps with one clear peak per keypoint, a few dead ones."""
    generator = torch.Generator().manual_seed(seed)
    heat = torch.rand(1, KEYPOINTS, HEIGHT, WIDTH, generator=generator) * 0.4
    peaks = torch.randint(0, HEIGHT * WIDTH, (KEYPOINTS,), generator=generator)
    heat.view(1, KEYPOINTS, -1)[0, torch.arange(KEYPOINTS), peaks] = 1.0
    heat[0, :5] = 0.0  # keypoints the model gave nothing for
    return heat


@pytest.mark.parametrize("box", [
    [200.0, 90.0, 240.0, 180.0],   # wider than the crop's aspect
    [300.0, 20.0, 60.0, 300.0],    # taller
    [0.0, 0.0, 640.0, 360.0],      # the whole frame
])
def test_keypoints_in_image_matches_the_processor(box):
    processor = sapiens2.Sapiens2ImageProcessor()
    assert (processor.size["height"], processor.size["width"]) == CROP
    heat = _heatmaps(seed=int(box[0]))
    reference = processor.post_process_pose_estimation(
        SimpleNamespace(heatmaps=heat), boxes=[[box]])[0][0]
    extent = pose_post.heatmap_extent(heat.shape, "cpu")
    assert extent.tolist() == [WIDTH - 1, HEIGHT - 1]
    packed = pose_post.keypoints_in_image(
        heat, torch.tensor(box), extent, CROP)
    assert packed.shape == (KEYPOINTS, 3) and packed.dtype == torch.float32
    assert torch.equal(packed[:, :2], reference["keypoints"])
    assert torch.equal(packed[:, 2], reference["scores"])


def test_a_batch_of_crops_matches_the_processor_person_by_person():
    """`keypoints_in_image_batch` over three people is the processor's
    post-processing of the three, and each row is what the batch of one
    gives for that person alone (the production path is unchanged).

    The one exception is a keypoint whose heatmap is all zero: its argmax
    is (-1, -1) and the processor's refinement then reads the neighbouring
    heatmap in memory, which in a batch is the previous person's. Such a
    keypoint has score 0 and no consumer reads its position, so the rows
    are compared where the score is positive.
    """
    processor = sapiens2.Sapiens2ImageProcessor()
    boxes = [[200.0, 90.0, 240.0, 180.0], [300.0, 20.0, 60.0, 300.0],
             [0.0, 0.0, 640.0, 360.0]]
    heat = torch.cat([_heatmaps(seed=11 + i) for i in range(len(boxes))])
    assert heat.shape[0] == 3
    reference = processor.post_process_pose_estimation(
        SimpleNamespace(heatmaps=heat), boxes=[boxes])[0]
    extent = pose_post.heatmap_extent(heat.shape, "cpu")
    packed = pose_post.keypoints_in_image_batch(
        heat, torch.tensor(boxes), extent, CROP)
    assert packed.shape == (3, KEYPOINTS, 3) and packed.dtype == torch.float32
    for i, box in enumerate(boxes):
        assert torch.equal(packed[i, :, :2], reference[i]["keypoints"])
        assert torch.equal(packed[i, :, 2], reference[i]["scores"])
        alone = pose_post.keypoints_in_image(
            heat[i:i + 1], torch.tensor(box), extent, CROP)
        assert torch.equal(alone[:, 2], packed[i, :, 2])
        live = packed[i, :, 2] > 0
        assert int(live.sum()) == KEYPOINTS - 5
        assert torch.equal(alone[live], packed[i][live])


def test_bf16_heatmaps_are_refined_in_float32_like_the_processor():
    processor = sapiens2.Sapiens2ImageProcessor()
    box = [100.0, 50.0, 200.0, 250.0]
    heat = _heatmaps(seed=7).to(torch.bfloat16)
    reference = processor.post_process_pose_estimation(
        SimpleNamespace(heatmaps=heat.float()), boxes=[[box]])[0][0]
    packed = pose_post.keypoints_in_image(
        heat, torch.tensor(box), pose_post.heatmap_extent(heat.shape, "cpu"),
        CROP)
    assert torch.equal(packed[:, :2], reference["keypoints"])
    assert torch.equal(packed[:, 2], reference["scores"])


def test_the_box_tensor_is_read_not_rebuilt():
    """The box is a graph input: the same heatmaps with a different box in
    the same tensor must move the keypoints, as a replay would."""
    heat = _heatmaps(seed=3)
    extent = pose_post.heatmap_extent(heat.shape, "cpu")
    box = torch.tensor([10.0, 10.0, 100.0, 200.0])
    first = pose_post.keypoints_in_image(heat, box, extent, CROP).clone()
    box.copy_(torch.tensor([310.0, 10.0, 100.0, 200.0]))
    second = pose_post.keypoints_in_image(heat, box, extent, CROP)
    live = first[:, 2] > 0
    assert torch.allclose(second[live, 0] - first[live, 0],
                          torch.full((int(live.sum()),), 300.0), atol=1e-3)
    assert torch.equal(second[:, 1], first[:, 1])
    assert torch.equal(second[:, 2], first[:, 2])
