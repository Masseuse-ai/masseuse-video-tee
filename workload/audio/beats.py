"""BEATs, the second AudioSet tagger: per-window class scores and a summary.

BEATs (Microsoft's `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2`, from
microsoft/unilm, MIT) is an AudioSet tagger like CED (ced.py): given a
window of 16 kHz mono audio it scores each of the 527 AudioSet classes.
The same forward pass yields a summary of the window: the mean, over the
window's time-frequency patches, of the encoder's last layer (768 values).
That is all that leaves this module: samples in, two float vectors out.

The model code is upstream's (`vendor/unilm_beats`, its filterbank from
`vendor/kaldi_fbank`, torchaudio's); the checkpoint is baked into the image
at a pinned revision and SHA-256 (workload/tee/Dockerfile.tee, the `beats`
stage). The classes are identified by their AudioSet ontology ids, read
from the checkpoint in its output order (`class_ids`); the image carries no
table of class names for this model.

The filterbank always runs on the CPU (a few milliseconds a window). The
network runs on the CPU by default, with a bounded number of threads, or on
CUDA as one captured graph (pixel/gpu_graph.py) when asked for; every
window has the same length, so the graph's shapes are fixed. torch is
imported on construction only: the stage and its tests import this module
without it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "/opt/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
# The filterbank's normalisation, as upstream fine-tuned the checkpoint with
# it (BEATs.preprocess divides by twice the standard deviation).
FBANK_MEAN = 15.41663
FBANK_STD = 6.55582
# The CPU path's intra-op threads: the slot's other CPU work (the decoders,
# CED, the analysis process, the record) keeps the rest of its cores. torch's
# thread count is process-wide; nothing else in the producer runs torch on
# the CPU.
DEFAULT_THREADS = 4


class _Direct:
    """The CPU path: the network called as it is (gpu_graph.Eager's
    interface, without importing the pixel package)."""

    graphed = False

    def __init__(self, fn):
        self.fn = fn

    def replay(self, *inputs):
        return self.fn(*inputs)


class BeatsEngine:
    """One loaded BEATs checkpoint, scoring windows of a fixed length."""

    def __init__(self, model_path: str | Path | None = None, *,
                 window_s: float = 2.0, device: str = "cpu",
                 threads: int | None = DEFAULT_THREADS, telemetry=None) -> None:
        import torch  # noqa: PLC0415 - heavy, and only the enclave needs it

        audio_dir = str(Path(__file__).resolve().parent)
        if audio_dir not in sys.path:
            sys.path.insert(0, audio_dir)
        from vendor.unilm_beats.BEATs import BEATs, BEATsConfig  # noqa: PLC0415

        self.torch = torch
        self.model_path = Path(
            model_path or os.environ.get("BEATS_MODEL_PATH", DEFAULT_MODEL))
        self.device = str(device)
        self.threads = threads if self.device == "cpu" else None
        if self.threads:
            torch.set_num_threads(int(self.threads))
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model = BEATs(BEATsConfig(state["cfg"]))
        model.load_state_dict(state["model"], strict=True)
        if getattr(model, "predictor", None) is None:
            raise RuntimeError(f"{self.model_path.name} has no class predictor")
        model.eval().to(self.device)
        self.model = model
        label_dict = state["label_dict"]
        self.class_ids = tuple(str(label_dict[index]) for index in range(len(label_dict)))
        self.class_count = len(self.class_ids)
        self.embedding_size = int(model.cfg.encoder_embed_dim)
        self.window_samples = int(round(window_s * SAMPLE_RATE))
        fbank = self.features(np.zeros(self.window_samples, dtype=np.float32))
        self._network = self._build_network(fbank, telemetry)
        self.graphed = bool(getattr(self._network, "graphed", False))

    @property
    def version(self) -> str:
        return f"unilm-beats:{self.model_path.name}"

    def _build_network(self, fbank, telemetry):
        """The forward from the normalised filterbank to (scores, summary):
        upstream's `extract_features` with the predictor's mean taken over
        the same patches the summary is the mean of."""
        torch = self.torch
        model = self.model

        def network(features):
            with torch.inference_mode():
                x = model.patch_embedding(features)
                x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
                x = model.layer_norm(x)
                if model.post_extract_proj is not None:
                    x = model.post_extract_proj(x)
                x, _ = model.encoder(x, padding_mask=None)
                summary = x.mean(dim=1)
                scores = torch.sigmoid(model.predictor(x).mean(dim=1))
                return scores, summary

        if not self.device.startswith("cuda"):
            return _Direct(network)
        from gpu_graph import build  # noqa: PLC0415 - pixel/, on the producer's path

        static = fbank.to(self.device)
        return build(network, (static,), name="beats", device=self.device,
                     telemetry=telemetry)

    def features(self, window: np.ndarray):
        """The normalised filterbank of one window, on the CPU, as upstream's
        `preprocess` computes it: (1, 1, frames, 128)."""
        torch = self.torch
        wav = np.ascontiguousarray(window, dtype=np.float32)
        if wav.shape != (self.window_samples,):
            raise ValueError(f"window of {wav.shape} samples, expected ({self.window_samples},)")
        with torch.inference_mode():
            source = torch.from_numpy(wav)[None, :]
            fbank = self.model.preprocess(source, fbank_mean=FBANK_MEAN, fbank_std=FBANK_STD)
        return fbank.unsqueeze(1)

    def tag(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One window (float32 mono in [-1, 1], `window_samples` long) in;
        the 527 class scores (sigmoid, in `class_ids` order) and the
        768-value summary out, both float32."""
        fbank = self.features(window)
        if self.device.startswith("cuda"):
            fbank = fbank.to(self.device)
        scores, summary = self._network.replay(fbank)
        return (scores[0].float().cpu().numpy().copy(),
                summary[0].float().cpu().numpy().copy())
