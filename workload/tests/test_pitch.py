"""The pitch contour's normalised autocorrelation: the vectorised form is
the lag loop it replaced, to the float noise of the summation order, on
voiced, unvoiced and silent frames; and it is fast enough for the audio
stage to keep up with real time (2026-09-16 it was not)."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio"))

import pitch  # noqa: E402

SAMPLE_RATE = 16_000


def _reference(frame: np.ndarray, autocorrelation: np.ndarray,
               min_lag: int, max_lag: int) -> np.ndarray:
    """The loop as it stood before 2026-09-17, kept here as the oracle."""
    normalized = np.zeros(max_lag - min_lag + 1, dtype=np.float64)
    frame_sq = np.square(frame)
    for offset, lag in enumerate(range(min_lag, max_lag + 1)):
        denominator = math.sqrt(
            float(np.sum(frame_sq[:-lag]) * np.sum(frame_sq[lag:]))
        )
        if denominator > 1e-12:
            normalized[offset] = autocorrelation[lag] / denominator
    return normalized


def _windowed(signal: np.ndarray, frame_length: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    frame = signal[:frame_length].astype(np.float64)
    frame -= np.mean(frame)
    frame *= np.hanning(frame_length)
    spectrum = np.fft.rfft(frame, n=frame_length * 2)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum))[:frame_length]
    return frame, autocorrelation


def _frames() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(1024) / SAMPLE_RATE
    voiced = 0.3 * np.sin(2 * np.pi * 142.0 * t) + 0.1 * np.sin(2 * np.pi * 284.0 * t)
    return {
        "voiced": voiced.astype(np.float32),
        "voiced_low": (0.05 * np.sin(2 * np.pi * 85.0 * t)).astype(np.float32),
        "voiced_high": (0.2 * np.sin(2 * np.pi * 620.0 * t)).astype(np.float32),
        "noise": rng.normal(0, 0.1, 1024).astype(np.float32),
        "breath": (rng.normal(0, 0.02, 1024) * np.hanning(1024)).astype(np.float32),
        "silence": np.zeros(1024, dtype=np.float32),
        "one_sample": np.concatenate(([0.5], np.zeros(1023))).astype(np.float32),
        "tail_only": np.concatenate((np.zeros(900), rng.normal(0, 0.2, 124))).astype(np.float32),
    }


def test_the_vectorised_normalisation_is_the_lag_loop():
    min_lag = max(1, int(SAMPLE_RATE / 700.0))
    max_lag = min(1024 - 2, int(SAMPLE_RATE / 70.0))
    for name, signal in _frames().items():
        frame, autocorrelation = _windowed(signal)
        fast = pitch.normalized_autocorrelation(frame, autocorrelation, min_lag, max_lag)
        slow = _reference(frame, autocorrelation, min_lag, max_lag)
        assert fast.shape == slow.shape == (max_lag - min_lag + 1,)
        assert np.allclose(fast, slow, atol=1e-9, rtol=0.0), name
        assert int(np.argmax(fast)) == int(np.argmax(slow)), name


def test_the_contour_reads_the_same_pitch_as_before_on_a_voiced_tone():
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 142.0 * t)).astype(np.float32)
    frames = pitch.pitch_frames(tone, SAMPLE_RATE)
    voiced = [f for f in frames if f.pitch_hz is not None]
    assert len(voiced) > 0.9 * len(frames)
    assert abs(float(np.median([f.pitch_hz for f in voiced])) - 142.0) < 1.5
    summary = pitch.estimate_pitch(tone, SAMPLE_RATE)
    assert abs(summary["pitchHz"] - 142.0) < 1.5 and summary["voicedFramePct"] > 90.0
    quiet = pitch.pitch_frames(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert all(f.pitch_hz is None and f.dbfs == -90.0 for f in quiet)


def test_a_two_second_window_is_measured_well_inside_a_half_second_hop():
    # The audio stage runs estimate_pitch over its 2 s window every 0.5 s
    # hop, on the same thread as the classifier. Noise is the costly case:
    # every frame is above the level floor and none is periodic.
    rng = np.random.default_rng(3)
    window = rng.normal(0, 0.1, SAMPLE_RATE * 2).astype(np.float32)
    pitch.pitch_frames(window, SAMPLE_RATE)  # warm
    started = time.perf_counter()
    for _ in range(3):
        pitch.pitch_frames(window, SAMPLE_RATE)
    per_window = (time.perf_counter() - started) / 3
    assert per_window < 0.15, f"{per_window * 1000:.0f} ms for a 2 s window"
