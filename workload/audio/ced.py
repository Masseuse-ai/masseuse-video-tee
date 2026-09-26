"""ctypes binding for ced.cpp's exception-free C ABI v1.

CED (Consistent Ensemble Distillation, Xiaomi's `mispeech/ced-small`) is an
AudioSet tagger: given a few seconds of 16 kHz mono audio it scores each of
the 527 AudioSet classes. The enclave runs the GGUF conversion of the small
model through ced.cpp (`workload/tee/Dockerfile.tee` builds the library and
pins the weights by SHA-256). `classify` keeps the scores for the labels
below: the non-speech vocalization classes the analysis is about, plus
`Speech` so that a voice in the room (the massage guidance playing out of
the phone, a person talking) can be told apart from them. `classify_all`
returns the whole table, one score per class in the model's own order, so
that the analysis can also read the room (music, a fan, rain, a machine's
hum) and tell it from the person. Nothing else about the audio leaves this
module: the input is samples, the output is floats, one per label.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

# AudioSet display names, as the model's label table spells them.
TARGET_LABELS = (
    "Screaming",
    "Crying, sobbing",
    "Whimper",
    "Wail, moan",
    "Sigh",
    "Groan",
    "Grunt",
    "Breathing",
    "Gasp",
    "Pant",
    "Speech",
)

DEFAULT_LIBRARY = "/opt/ced/libced.so"
DEFAULT_MODEL = "/opt/ced/ced-small-f16.gguf"


class CedTag(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_int),
        ("score", ctypes.c_float),
        ("label", ctypes.c_char_p),
    ]


class CedEngine:
    """One warm, single-threaded ced.cpp context."""

    def __init__(
        self,
        library_path: str | Path | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.library_path = Path(
            library_path or os.environ.get("CED_LIBRARY_PATH", DEFAULT_LIBRARY))
        self.model_path = Path(
            model_path or os.environ.get("CED_MODEL_PATH", DEFAULT_MODEL))
        self.library = ctypes.CDLL(str(self.library_path))
        self._bind()
        if self.library.ced_capi_abi_version() != 1:
            raise RuntimeError("unsupported ced.cpp C API version")
        self.context = self.library.ced_capi_load(str(self.model_path).encode())
        if not self.context:
            error = self.library.ced_capi_last_error(None).decode(errors="replace")
            raise RuntimeError(f"could not load CED model: {error}")
        self.class_count = self.library.ced_capi_num_classes(self.context)
        self.labels = tuple(
            self.library.ced_capi_label(self.context, index).decode("utf-8")
            for index in range(self.class_count)
        )
        label_to_index = {label: index for index, label in enumerate(self.labels)}
        missing = [label for label in TARGET_LABELS if label not in label_to_index]
        if missing:
            self.close()
            raise RuntimeError(f"CED model is missing labels: {', '.join(missing)}")
        self.target_indices = {
            label: label_to_index[label] for label in TARGET_LABELS
        }
        self._output = (CedTag * self.class_count)()

    def _bind(self) -> None:
        library = self.library
        library.ced_capi_abi_version.restype = ctypes.c_int
        library.ced_capi_load.argtypes = [ctypes.c_char_p]
        library.ced_capi_load.restype = ctypes.c_void_p
        library.ced_capi_free.argtypes = [ctypes.c_void_p]
        library.ced_capi_last_error.argtypes = [ctypes.c_void_p]
        library.ced_capi_last_error.restype = ctypes.c_char_p
        library.ced_capi_num_classes.argtypes = [ctypes.c_void_p]
        library.ced_capi_num_classes.restype = ctypes.c_int
        library.ced_capi_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.ced_capi_label.restype = ctypes.c_char_p
        library.ced_capi_classify_pcm.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CedTag),
            ctypes.c_int,
        ]
        library.ced_capi_classify_pcm.restype = ctypes.c_int

    @property
    def version(self) -> str:
        return f"ced.cpp-abi1:{self.model_path.name}"

    def classify_all(self, wav: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        """Every class's score over `wav` (float32 mono in [-1, 1]), as a
        float32 vector in the order of `labels`."""
        contiguous = np.ascontiguousarray(wav, dtype=np.float32)
        count = self.library.ced_capi_classify_pcm(
            self.context,
            contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(contiguous),
            sample_rate,
            self._output,
            self.class_count,
        )
        if count != self.class_count:
            error = self.library.ced_capi_last_error(self.context).decode(
                errors="replace"
            )
            raise RuntimeError(f"CED classification failed: {error}")
        vector = np.zeros(self.class_count, dtype=np.float32)
        for tag in self._output[:count]:
            vector[tag.index] = tag.score
        return vector

    def target_scores(self, vector: np.ndarray) -> dict[str, float]:
        """The TARGET_LABELS entries of a `classify_all` vector."""
        return {label: float(vector[index]) for label, index in self.target_indices.items()}

    def classify(self, wav: np.ndarray, sample_rate: int = 16_000) -> dict[str, float]:
        """Scores for TARGET_LABELS over `wav` (float32 mono in [-1, 1])."""
        return self.target_scores(self.classify_all(wav, sample_rate))

    def close(self) -> None:
        if getattr(self, "context", None):
            self.library.ced_capi_free(self.context)
            self.context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
