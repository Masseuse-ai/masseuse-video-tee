"""Frame-level pitch (F0) and level of 16 kHz mono audio.

Normalised autocorrelation over 64 ms frames every 16 ms: each frame gets a
level in dBFS and, when it is periodic enough and in the human voice range,
a pitch in Hz. No speech model is involved and nothing here recognises
words; the output is a contour of numbers per frame.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class PitchFrame(NamedTuple):
    """One analysis frame, timestamped at its center.

    `pitch_hz` is None for unvoiced frames — silence, frames below the
    periodicity floor, and frames whose estimate lands outside [fmin, fmax].
    """

    time_s: float
    pitch_hz: float | None
    confidence: float
    dbfs: float


def rms_dbfs(wav: np.ndarray) -> float:
    if wav.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    return max(-90.0, 20.0 * math.log10(max(rms, 10 ** (-90.0 / 20.0))))


def pitch_frames(
    wav: np.ndarray,
    sample_rate: int = 16_000,
    *,
    fmin: float = 70.0,
    fmax: float = 700.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_dbfs: float = -55.0,
    min_periodicity: float = 0.35,
) -> list[PitchFrame]:
    """Per-frame F0 contour from normalized autocorrelation.

    Every frame position is emitted, voiced or not, so callers can measure
    voiced coverage and pitch trajectory over the same grid.
    """
    signal = np.ascontiguousarray(wav, dtype=np.float32)
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_length - 2, int(sample_rate / fmin))
    window = np.hanning(frame_length).astype(np.float32)
    center_offset_s = frame_length / 2 / sample_rate
    frames: list[PitchFrame] = []

    for start in range(0, len(signal) - frame_length + 1, hop_length):
        time_s = start / sample_rate + center_offset_s
        frame = signal[start:start + frame_length].astype(np.float64)
        frame -= np.mean(frame)
        frame *= window
        dbfs = rms_dbfs(frame)
        if dbfs < min_dbfs:
            frames.append(PitchFrame(time_s, None, 0.0, dbfs))
            continue

        spectrum = np.fft.rfft(frame, n=frame_length * 2)
        autocorrelation = np.fft.irfft(
            spectrum * np.conjugate(spectrum)
        )[:frame_length]
        normalized = np.zeros(max_lag - min_lag + 1, dtype=np.float64)
        frame_sq = np.square(frame)
        for offset, lag in enumerate(range(min_lag, max_lag + 1)):
            denominator = math.sqrt(
                float(np.sum(frame_sq[:-lag]) * np.sum(frame_sq[lag:]))
            )
            if denominator > 1e-12:
                normalized[offset] = autocorrelation[lag] / denominator

        peak_offset = int(np.argmax(normalized))
        confidence = float(normalized[peak_offset])
        if confidence < min_periodicity:
            frames.append(PitchFrame(time_s, None, confidence, dbfs))
            continue
        lag = float(min_lag + peak_offset)
        if 0 < peak_offset < len(normalized) - 1:
            left, center, right = normalized[
                peak_offset - 1:peak_offset + 2
            ]
            denominator = left - 2 * center + right
            if abs(denominator) > 1e-12:
                lag += 0.5 * (left - right) / denominator
        pitch = sample_rate / lag
        in_range = fmin <= pitch <= fmax
        frames.append(
            PitchFrame(
                time_s,
                float(pitch) if in_range else None,
                confidence,
                dbfs,
            )
        )
    return frames


def estimate_pitch(
    wav: np.ndarray,
    sample_rate: int = 16_000,
    *,
    fmin: float = 70.0,
    fmax: float = 700.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_dbfs: float = -55.0,
    min_periodicity: float = 0.35,
) -> dict:
    """Estimate median F0 from normalized autocorrelation frames.

    Unvoiced sounds (a breath, a pant) legitimately return no pitch while
    remaining visible to the classifier. `voicedFramePct` reports pitch
    coverage independently.
    """
    signal = np.ascontiguousarray(wav, dtype=np.float32)
    frames = pitch_frames(
        signal,
        sample_rate,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        min_dbfs=min_dbfs,
        min_periodicity=min_periodicity,
    )
    voiced = [frame for frame in frames if frame.pitch_hz is not None]

    return {
        "pitchHz": (
            round(float(np.median([frame.pitch_hz for frame in voiced])), 1)
            if voiced
            else None
        ),
        "pitchConfidence": (
            round(float(np.median([frame.confidence for frame in voiced])), 3)
            if voiced
            else 0.0
        ),
        "voicedFramePct": (
            round(len(voiced) / max(1, len(frames)) * 100.0, 1)
        ),
        "loudnessDbfs": round(rms_dbfs(signal), 1),
    }
