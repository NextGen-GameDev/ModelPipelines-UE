"""Helpers for the STT goldens (WP9S.2, contract C71): sidecar IO, synthetic clips, word frames and an independent
numpy re-implementation of the Parakeet front end (the C73 recipe) used as a self-check."""
from __future__ import annotations

import hashlib
import math

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
WIN = 400
HOP = 160
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
MIN_VALID_FRAMES = 2          # fewer valid mel frames than this -> "too short" (no features, no decode)
WORD_MARK = "▁"
BLANK = 8192


def f32_bytes(a):
    """Raw little-endian float32, row-major, no header."""
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rms_dbfs(x):
    """20*log10(sqrt(mean(x^2))) in float64; all-zero (or empty) -> -200.0."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -200.0
    ms = float(np.mean(x * x))
    if ms == 0.0:
        return -200.0
    return 20.0 * math.log10(math.sqrt(ms))


def valid_frames(num_samples):
    return num_samples // HOP


def synthetic_clips(first_clip_audio):
    """-> [(id, float32 samples)] in contract order. exact_16000 = samples [16000:32000) of the first visible clip."""
    n = np.arange(16000, dtype=np.float64)
    out = [
        ("silence_0p5s", np.zeros(8000, dtype=np.float32)),
        ("sine440_1s", (0.5 * np.sin(2 * np.pi * 440.0 * n / SAMPLE_RATE)).astype(np.float32)),
        ("noise_seed1234_2s", np.random.default_rng(1234).uniform(-0.1, 0.1, 32000).astype(np.float32)),
        ("short_320", (0.5 * np.sin(2 * np.pi * 300.0 * n[:320] / SAMPLE_RATE)).astype(np.float32)),
        ("short_319", (0.5 * np.sin(2 * np.pi * 300.0 * n[:319] / SAMPLE_RATE)).astype(np.float32)),
        ("exact_16000", np.ascontiguousarray(first_clip_audio[16000:32000], dtype=np.float32)),
    ]
    assert len(out[5][1]) == 16000
    return out


def word_frames(steps, pieces_of):
    """steps = [(k, d)] incl. blanks; pieces_of(k) -> token piece string. A word starts at a piece beginning with U+2581
    (or at the first token). frames = encoder frame index t of the emitting step; end = last token's t + its duration.
    -> [[word, first_frame, last_frame], ...]"""
    words = []
    cur = None
    t = 0
    for k, d in steps:
        if k != BLANK:
            p = pieces_of(k)
            if cur is None or p.startswith(WORD_MARK):
                cur = [p.lstrip(WORD_MARK), t, t + d]
                words.append(cur)
            else:
                cur[0] += p
                cur[2] = t + d
        t += d
    return words


# ---- independent numpy front end (C73 steps 1-5) -------------------------------------------------------------------

def np_logmel(audio, mel_filters):
    """audio float32 [n] -> unnormalised log-mel [n//160+1, 128] (float32).
    pre-emphasis -> zero-pad 256 each side -> frames of 512 hop 160 -> symmetric Hann(400) centred in 512 -> |rfft|^2
    -> mel_filters[128,257] @ power -> log(. + 2^-24)."""
    x = np.asarray(audio, dtype=np.float32)
    x = np.concatenate([x[:1], x[1:] - np.float32(PREEMPH) * x[:-1]]).astype(np.float32)
    n = len(x)
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2)).astype(np.float64)
    nframes = n // HOP + 1
    k = np.arange(WIN, dtype=np.float64)
    win = np.zeros(N_FFT)
    off = (N_FFT - WIN) // 2
    win[off:off + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * k / (WIN - 1))
    idx = np.arange(nframes)[:, None] * HOP + np.arange(N_FFT)[None, :]
    spec = np.fft.rfft(xp[idx] * win[None, :], axis=1)
    power = spec.real ** 2 + spec.imag ** 2                        # [T, 257]
    mel = power @ np.asarray(mel_filters, dtype=np.float64).T      # [T, 128]
    return np.log(mel + LOG_GUARD).astype(np.float32)


def np_features(logmel, num_samples):
    """Per-bin mean / std (ddof=1) over the valid frames only, (x - mean) / (std + 1e-5); returns the valid frames."""
    L = num_samples // HOP
    v = logmel[:L].astype(np.float64)
    mean = v.sum(axis=0) / L
    var = ((v - mean) ** 2).sum(axis=0) / (L - 1)
    return ((v - mean) / (np.sqrt(var) + NORM_EPS)).astype(np.float32)
