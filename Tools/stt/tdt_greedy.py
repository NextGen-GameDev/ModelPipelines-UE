"""Greedy TDT decoding over onnxruntime: the reference loop for the C++ pipeline (contract C70).

    F = encoder frames; t = 0; token = BLANK; state = zeros; steps = []
    while t < F and len(steps) < 10 * F - 1:
        tok_logits, dur_logits, h, c = decoder_joint(enc[t], token, state)
        k = argmax(tok_logits); d = DURATIONS[argmax(dur_logits)]      # first max on ties
        if k == BLANK and d == 0: d = 1                                # only blanks are forced forward
        steps.append((k, d))
        if k != BLANK: token = k; state = (h, c)                       # predictor advances on non-blank only
        t += d
"""
from __future__ import annotations

import numpy as np

BLANK = 8192
VOCAB = 8193
DURATIONS = (0, 1, 2, 3, 4)
MAX_SYMBOLS_PER_STEP = 10   # HF max_length = 10 * F counts the start token, so steps < 10 * F - 1


def greedy_tdt(step_fn, enc, durations=DURATIONS, blank=BLANK, vocab=VOCAB, max_symbols=MAX_SYMBOLS_PER_STEP,
               trace=None, max_steps=None):
    """step_fn(frame[1,D] f32, token[1] i64, h, c) -> (token_logits[1,V], duration_logits[1,ND], h', c').
    enc: [F, D] (or [1, F, D]). Returns (steps, ids): steps = [(k, d)] for every step (blanks included), ids = non-blank
    ids. `trace` (optional list) receives (t, top1, top2) logit margins per step. `max_steps` overrides the default cap
    10*F-1 (HF sizes its cap from the encoder length of ALL mel frames, which can be one frame more than the valid F)."""
    enc = np.asarray(enc, dtype=np.float32)
    if enc.ndim == 3:
        enc = enc[0]
    F = enc.shape[0]
    t = 0
    token = np.array([blank], dtype=np.int64)
    h = np.zeros((2, 1, enc.shape[1]), dtype=np.float32)
    c = np.zeros_like(h)
    steps = []
    cap = max_symbols * F - 1 if max_steps is None else max_steps
    while t < F and len(steps) < cap:
        tl, dl, h2, c2 = step_fn(enc[t:t + 1], token, h, c)
        tl = np.asarray(tl)[0, :vocab]
        k = int(np.argmax(tl))
        d = int(durations[int(np.argmax(np.asarray(dl)[0]))])
        if k == blank and d == 0:
            d = 1
        if trace is not None:
            top2 = np.partition(tl, -2)[-2:]
            trace.append((t, float(top2.max()), float(top2.min())))
        steps.append((k, d))
        if k != blank:
            token = np.array([k], dtype=np.int64)
            h, c = h2, c2
        t += d
    ids = [k for k, _ in steps if k != blank]
    return steps, ids


def sequences_and_durations(steps, blank=BLANK):
    """HF generate() layout: sequences = [blank] + k's, durations = [0] + d's."""
    return [blank] + [k for k, _ in steps], [0] + [d for _, d in steps]


class OrtTdt:
    """Two onnxruntime CPU sessions (encoder, decoder_joint) from a model folder."""

    def __init__(self, folder, threads=0):
        import os
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(os.path.join(folder, "encoder.onnx"), so, providers=prov)
        self.dj = ort.InferenceSession(os.path.join(folder, "decoder_joint.onnx"), so, providers=prov)
        self.step_ns = 0
        self.step_count = 0

    def encode(self, input_features):
        """input_features [1,T,128] f32 -> [F,640]."""
        out = self.encoder.run(["encoder_projected"], {"input_features": np.ascontiguousarray(input_features, np.float32)})[0]
        return out[0]

    def step(self, frame, token, h, c):
        import time
        t0 = time.perf_counter_ns()
        r = self.dj.run(["token_logits", "duration_logits", "state_h_out", "state_c_out"],
                        {"encoder_frame": frame, "token": token, "state_h": h, "state_c": c})
        self.step_ns += time.perf_counter_ns() - t0
        self.step_count += 1
        return r

    def decode(self, input_features, trace=None):
        enc = self.encode(input_features)
        return greedy_tdt(self.step, enc, trace=trace)
