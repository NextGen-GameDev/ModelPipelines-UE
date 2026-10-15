"""WP9S.1a tests: greedy TDT loop, clip selection, and (when the model/test data exist) ONNX vs Hugging Face."""
import json
import os
import sys

import numpy as np
import pytest

# Runs only in the pipelines-stt env (transformers >= 5 with ParakeetForTDT, soundfile); the base env skips this module.
pytest.importorskip("soundfile", reason="needs the pipelines-stt env")
if not hasattr(pytest.importorskip("transformers"), "ParakeetForTDT"):
    pytest.skip("needs the pipelines-stt env (transformers with ParakeetForTDT)", allow_module_level=True)

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
from stt import clips as cl  # noqa: E402
from stt import tdt_greedy as tg  # noqa: E402

DEV = _paths.dev_dir()
FOLDER = os.path.join(DEV, "models", "stt", "phonon-2-onnx", "fp32")
CLIPS = os.path.join(DEV, "goldens", "stt", "clips_visible.json")
CONTAINER = os.path.join(DEV, "models", "stt", "phonon-2", "unpacked", "model.fermion")
BASE = os.path.join(DEV, "models", "stt", "parakeet-tdt-0.6b-v3-base")
HAVE_MODEL = os.path.isfile(os.path.join(FOLDER, "SOURCE.json")) and os.path.isfile(CLIPS) and os.path.isfile(CONTAINER)
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="phonon-2 onnx export / clips / container not present")


# ---- pure loop with a scripted step function (hand-computed) -----------------------------------------------------

def scripted(script):
    """step_fn replaying [(k, dur_index)]; records the (frame value, token) it was called with."""
    calls = []
    it = iter(script)

    def fn(frame, token, h, c):
        k, di = next(it)
        calls.append((float(frame[0, 0]), int(token[0]), float(h.sum())))
        tl = np.zeros((1, 8193), np.float32)
        tl[0, k] = 1.0
        dl = np.zeros((1, 5), np.float32)
        dl[0, di] = 1.0
        return tl, dl, h + 1.0, c + 1.0
    return fn, calls


def test_loop_forces_blank_forward_and_advances_state_on_tokens_only():
    enc = np.arange(6, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)   # frame t has value t, D=4
    # t=0: token 5, d=0 (stays on frame 0, state advances); t=0: blank d=0 -> forced d=1; t=1: token 7 d=2; t=3: blank d=3
    fn, calls = scripted([(5, 0), (8192, 0), (7, 2), (8192, 3)])
    steps, ids = tg.greedy_tdt(fn, enc)
    assert steps == [(5, 0), (8192, 1), (7, 2), (8192, 3)]
    assert ids == [5, 7]
    assert [c[0] for c in calls] == [0.0, 0.0, 1.0, 3.0]            # frame index follows the durations
    assert [c[1] for c in calls] == [8192, 5, 5, 7]                 # token = last non-blank
    # state sum starts at 0 (h shape [2,1,4] -> +1.0 per element = +8 per advance); blank step does not advance
    assert [c[2] for c in calls] == [0.0, 8.0, 8.0, 16.0]
    assert tg.sequences_and_durations(steps) == ([8192, 5, 8192, 7, 8192], [0, 0, 1, 2, 3])


def test_loop_stops_when_frames_exhausted():
    enc = np.zeros((2, 4), np.float32)
    fn, _ = scripted([(8192, 4)])
    steps, ids = tg.greedy_tdt(fn, enc)
    assert steps == [(8192, 4)] and ids == []


def test_loop_step_cap_is_10F_minus_1():
    enc = np.zeros((2, 4), np.float32)                                # F=2 -> cap 19 steps
    fn, calls = scripted([(5, 0)] * 100)                              # token with d=0 never advances
    steps, _ = tg.greedy_tdt(fn, enc)
    assert len(steps) == 19 and len(tg.sequences_and_durations(steps)[0]) == 20   # 10*F tokens incl. the start token


def test_argmax_tie_takes_first_max():
    a = np.array([0.5, 1.0, 1.0, 0.2], np.float32)
    assert int(np.argmax(a)) == 1
    tl = np.zeros((1, 8193), np.float32)
    tl[0, 10] = tl[0, 20] = 3.0
    dl = np.array([[0, 2, 2, 0, 0]], np.float32)
    steps, _ = tg.greedy_tdt(lambda f, t, h, c: (tl, dl, h, c), np.zeros((1, 4), np.float32))
    assert steps == [(10, 1)]                                         # first token max (10 not 20); duration index 1 not 2


def test_argmax_tie_duration_index():
    tl = np.zeros((1, 8193), np.float32)
    tl[0, 8192] = 1.0
    dl = np.array([[0, 0, 2, 2, 0]], np.float32)                      # tie between index 2 and 3 -> duration 2
    steps, _ = tg.greedy_tdt(lambda f, t, h, c: (tl, dl, h, c), np.zeros((1, 4), np.float32))
    assert steps == [(8192, 2)]


# ---- clip selection on a synthetic directory ---------------------------------------------------------------------

def _mk(root, spk, ch, utts):
    d = root / spk / ch
    d.mkdir(parents=True)
    with open(d / ("%s-%s.trans.txt" % (spk, ch)), "w", encoding="utf-8") as f:
        for u in utts:
            (d / (u + ".flac")).write_bytes(b"")
            f.write("%s TEXT %s\n" % (u, u))


def test_select_visible_closest_to_8s_ties_to_smaller_id(tmp_path):
    root = tmp_path / "test-clean"
    sr = 16000
    _mk(root, "1", "10", ["1-10-0000", "1-10-0001", "1-10-0002"])
    _mk(root, "1", "11", ["1-11-0000"])
    _mk(root, "2", "20", ["2-20-0000", "2-20-0001"])
    _mk(root, "3", "30", ["3-30-0000"])
    dur = {  # samples
        "1-10-0000": 5 * sr, "1-10-0001": int(8.4 * sr), "1-10-0002": 3 * sr, "1-11-0000": int(7.7 * sr),   # 7.7 is 0.3 away, 8.4 is 0.4
        "2-20-0000": 7 * sr, "2-20-0001": 9 * sr,                                                          # tie (1 s each) -> smaller id
        "3-30-0000": 12 * sr}
    rows = cl.scan_split(str(root), durations=dur)
    assert len(rows) == 7
    sel = cl.select_visible(rows)
    assert [r["id"] for r in sel] == ["1-11-0000", "2-20-0000", "3-30-0000"]
    out = cl.to_golden_rows(sel, str(tmp_path))
    assert out[0] == {"id": "1-11-0000", "path": "test-clean/1/11/1-11-0000.flac", "seconds": 7.7,
                      "num_samples": int(7.7 * sr), "ref": "TEXT 1-11-0000"}


# ---- ONNX vs Hugging Face (offline, skipped when the artifacts are absent) ----------------------------------------

@needs_model
def test_c70_signatures():
    from stt import phonon2_export as ex
    ex.assert_c70(FOLDER)
    src = json.load(open(os.path.join(FOLDER, "SOURCE.json"), encoding="utf-8"))
    assert src["revision"] == ex.REVISION and src["license"] == "cc-by-4.0" and src["opset"] == 17
    assert "encoder.onnx.data" in src["files"]


@pytest.fixture(scope="module")
def stack():
    import warnings
    warnings.filterwarnings("ignore")
    import check_phonon2_onnx as c
    hf = c.HfRef(CONTAINER, BASE)
    return c, hf, tg.OrtTdt(FOLDER)


@needs_model
def test_loop_equals_hf_on_three_visible_clips(stack):
    c, hf, ort = stack
    for row in cl.load_clips(CLIPS)[:3]:
        r = c.run_clip(hf, ort, cl.read_audio(cl.clip_abs_path(row, DEV)))
        assert r["seq_h"] == r["seq_o"] and r["dur_h"] == r["dur_o"] and r["text_h"] == r["text_o"], row["id"]
        assert len(r["seq_o"]) > 10


@needs_model
def test_silence_matches_hf(stack):
    c, hf, ort = stack
    r = c.run_clip(hf, ort, np.zeros(8000, np.float32))
    assert r["seq_h"] == r["seq_o"] and r["dur_h"] == r["dur_o"] and r["text_h"] == r["text_o"]
