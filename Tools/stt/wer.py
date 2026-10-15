"""Word error rate with the Open ASR Leaderboard normaliser (Whisper EnglishTextNormalizer). WP9S.1b.

Corpus WER = (S + D + I) summed over the set / N reference words summed over the set (not the mean of clip WERs).
"""
from __future__ import annotations

import json
import os


def edit_counts(ref_words, hyp_words):
    """Word-level Levenshtein -> (substitutions, deletions, insertions). Backtrace prefers substitution, then
    deletion, then insertion (the total S+D+I is the minimum regardless)."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref_words[i - 1]
        for j in range(1, m + 1):
            c = d[i - 1][j - 1] + (ri != hyp_words[j - 1])
            d[i][j] = min(c, d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, s, de, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]):
            s += ref_words[i - 1] != hyp_words[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, de, ins


def corpus_wer(refs, hyps):
    """refs, hyps: parallel lists of already-normalised strings -> dict(wer (fraction), S, D, I, N)."""
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        s, de, ins = edit_counts(rw, hw)
        S, D, I, N = S + s, D + de, I + ins, N + len(rw)
    return {"wer": (S + D + I) / N if N else 0.0, "S": S, "D": D, "I": I, "N": N}


def default_spelling_path():
    import _paths
    return os.path.join(_paths.dev_dir(), "testdata", "stt", "normalizer", "english.json")


def make_normalizer(spelling_path=None):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    with open(spelling_path or default_spelling_path(), encoding="utf-8") as f:
        return EnglishTextNormalizer(json.load(f))


def normalized_corpus_wer(refs, hyps, normalizer=None):
    nz = normalizer or make_normalizer()
    return corpus_wer([nz(r) for r in refs], [nz(h) for h in hyps])
