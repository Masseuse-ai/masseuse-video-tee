"""Per-segment acoustic measurements of a short span of audio.

These summarise a segment the analysis process asked about (a sound it
proposed from the frame levels it was streamed, see `audio_stage.py`) into
a handful of numbers: an F0 summary, a loudness summary and two spectral
shape statistics. They are computed here, in the process that holds the
samples, and only the numbers cross the socket.
"""

from __future__ import annotations

import math

import numpy as np

from pitch import pitch_frames, rms_dbfs

SAMPLE_RATE = 16_000

# A median over one or two voiced frames is noise, not an F0. Unvoiced
# breathing is a legitimate pant, so the honest answer for these is no pitch
# rather than a number that would skew every aggregate.
MIN_VOICED_FRACTION = 0.2
MIN_VOICED_FRAMES = 3

# ced.cpp's mel graph refuses segments much shorter than a second, and the
# streaming classification never shows it one anyway - it classifies 2 s
# windows. Short segments are therefore centred in at least this much
# surrounding context before classification.
CED_MIN_CONTEXT_S = 1.0


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def pitch_stats(segment: np.ndarray) -> dict:
    """F0 contour summary. Unvoiced segments legitimately report no pitch."""
    frames = pitch_frames(segment, SAMPLE_RATE)
    voiced = [frame for frame in frames if frame.pitch_hz is not None]
    voiced_fraction = len(voiced) / len(frames) if frames else 0.0
    reliable = (
        len(voiced) >= MIN_VOICED_FRAMES and voiced_fraction >= MIN_VOICED_FRACTION
    )
    pitches = [frame.pitch_hz for frame in voiced] if reliable else []
    slope = None
    if reliable:
        times = np.array([frame.time_s for frame in voiced], dtype=np.float64)
        values = np.array(pitches, dtype=np.float64)
        if float(np.ptp(times)) > 1e-6:
            slope = round(float(np.polyfit(times, values, 1)[0]), 1)
    return {
        "medianHz": round(float(np.median(pitches)), 1) if pitches else None,
        "p10Hz": percentile(pitches, 10),
        "p90Hz": percentile(pitches, 90),
        "slopeHzPerS": slope,
        "voicedFraction": round(voiced_fraction, 3),
        "voicedFrames": len(voiced),
        "pitchReliable": reliable,
        "confidence": (
            round(float(np.median([f.confidence for f in voiced])), 3)
            if voiced
            else 0.0
        ),
        "frames": len(frames),
    }


def loudness_stats(segment: np.ndarray) -> dict:
    frames = pitch_frames(segment, SAMPLE_RATE)
    levels = [frame.dbfs for frame in frames if math.isfinite(frame.dbfs)]
    return {
        "peakDbfs": round(max(levels), 1) if levels else round(rms_dbfs(segment), 1),
        "meanDbfs": round(float(np.mean(levels)), 1) if levels else None,
        "rmsDbfs": round(rms_dbfs(segment), 1),
    }


def spectral_stats(segment: np.ndarray) -> dict:
    """Centroid and 85% rolloff: breathy versus voiced, in two numbers."""
    if segment.size < 256:
        return {"centroidHz": None, "rolloff85Hz": None}
    window = np.hanning(segment.size).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(segment.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(segment.size, 1.0 / SAMPLE_RATE)
    total = float(spectrum.sum())
    if total <= 1e-12:
        return {"centroidHz": None, "rolloff85Hz": None}
    centroid = float((freqs * spectrum).sum() / total)
    cumulative = np.cumsum(spectrum)
    index = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff = float(freqs[min(index, len(freqs) - 1)])
    return {
        "centroidHz": round(centroid, 1),
        "rolloff85Hz": round(rolloff, 1),
    }
