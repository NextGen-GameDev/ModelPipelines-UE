"""Long-audio windowing and stitched decode: the exact reference of contract C77 (WP9S.5c1).

Applies to the 16 kHz mono signal after mix-down and resampling. Python and C++ implement exactly this; the integer
sample positions must match.

    N samples, R = 16000, H = 160, MIN = 25 s, MAX = 35 s, TAIL = 5 s, HALF = 10 frames
    P[0] = 0; P[k+1] = P[k] + (double)x[k]^2           (float64, strictly sequential)
    S(i) = P[H*(i+HALF)] - P[H*(i-HALF)]
    start = 0; while N - start > MAX*R:
        lo = start + MIN*R; hi = start + min(MAX*R, (N - start) - TAIL*R)
        cut = H * argmin S(i) over integers i with lo <= H*i <= hi (ties -> smallest i); emit [start, cut); start = cut
    emit [start, N)
"""
from __future__ import annotations

import numpy as np

R = 16000
H = 160
MIN_S = 25
MAX_S = 35
TAIL_S = 5
HALF = 10
GATE_DB = -60.0
FRAME_S = 0.08


def prefix_sums(x):
    """float64 prefix sums of x^2, length N+1, P[0] = 0 (np.cumsum is strictly sequential)."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    p = np.zeros(len(x) + 1, dtype=np.float64)
    np.cumsum(x * x, out=p[1:])
    return p


def find_cut(p, start, n):
    """Cut position (sample) for the window starting at `start` of an n-sample signal with prefix sums p."""
    lo = start + MIN_S * R
    hi = start + min(MAX_S * R, (n - start) - TAIL_S * R)
    i0 = -(-lo // H)          # ceil
    i1 = hi // H              # floor
    assert i1 >= i0, (lo, hi)
    i = np.arange(i0, i1 + 1, dtype=np.int64)
    s = p[H * (i + HALF)] - p[H * (i - HALF)]
    return int(H * i[int(np.argmin(s))])     # argmin returns the first minimum -> smallest i


def plan_windows(x):
    """-> [(start, end)] sample ranges tiling [0, N)."""
    n = len(x)
    out = []
    start = 0
    if n > MAX_S * R:
        p = prefix_sums(x)
        while n - start > MAX_S * R:
            cut = find_cut(p, start, n)
            out.append((start, cut))
            start = cut
    out.append((start, n))
    return out


def plan_fixed(n, seconds=30):
    """Cuts at every `seconds` (no pause search) for the comparison run."""
    step = seconds * R
    out = [(s, min(s + step, n)) for s in range(0, n, step)]
    return out or [(0, n)]


def decode_windows(x, windows, decode_fn):
    """decode_fn(samples) -> dict(rms_dbfs, gated, token_ids, text, words=[[w, t0, t1]] window-local seconds).
    Returns (window dicts, text, global words)."""
    wins = []
    texts = []
    words = []
    for s, e in windows:
        d = dict(decode_fn(np.ascontiguousarray(x[s:e])))
        wins.append({"start": int(s), "end": int(e), **d})
        if d["text"].strip():
            texts.append(d["text"].strip())
        off = s / R
        for w, a, b in d["words"]:
            words.append([w, a + off, b + off])
    return wins, " ".join(texts), words


def words_from_frames(word_frames):
    return [[w, a * FRAME_S, b * FRAME_S] for w, a, b in word_frames]
