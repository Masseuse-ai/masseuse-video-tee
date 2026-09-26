# RT-DETRv4 vendor

Source: <https://github.com/RT-DETRs/RT-DETRv4>

The exact source revision is recorded in `UPSTREAM_COMMIT`; `LICENSE` is the
upstream Apache-2.0 license. `engine/` and `configs/` are vendored because no
Transformers implementation or immutable package release exists for v4.

FemLed carries three inference-only patches:

1. `engine/__init__.py`, `engine/backbone/__init__.py`, and
   `engine/rtv4/__init__.py` register only the five classes used by the frozen
   X config. Upstream eagerly imports its optimizer, COCO evaluator, profiler,
   TensorBoard, and VFM teacher on every inference process.
2. `engine/core/_config.py` imports TensorBoard lazily, and
   `engine/core/yaml_config.py` imports the training-only VFM teacher lazily.
3. `HybridEncoder` registers precomputed positional embeddings as
   non-persistent buffers. Upstream stores them as plain CPU tensor attributes,
   which fails when the deployed graph is moved to CUDA.

The checkpoint is not committed. Its immutable identities are:

```
rtv4_hgnetv2_x_coco.pth (official download)
sha256 2925ec2e53d48e6141db8601c8656566219e696cd63bf8d6955190cae6bfd3f9

rtv4_hgnetv2_x_coco_ema.safetensors (FemLed re-export: ckpt["ema"]["module"]
only, tensors bit-identical; a quarter of the pickle's bytes for cold boot)
sha256 6912734704579139b15ce836ce6161d997e4bf622a014999b998fbed76c1fca4
```
