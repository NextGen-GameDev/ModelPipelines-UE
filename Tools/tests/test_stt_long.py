"""WP9S.5c1 tests: C77 windowing (pure numpy) and, when present, self-checks of long_goldens.json."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
from stt import long_audio as la  # noqa: E402

R = la.R
GDIR = os.path.join(_paths.dev_dir(), "goldens", "stt", "long")
GJSON = os.path.join(GDIR, "long_goldens.json")


def noise(seconds, seed=1, amp=0.1):
    return np.random.default_rng(seed).uniform(-amp, amp, int(seconds * R)).astype(np.float32)


def test_short_is_one_window():
    for sec in (0.5, 10, 35):
        x = noise(sec)
        assert la.plan_windows(x) == [(0, len(x))]


def test_just_over_max_cuts():
    x = noise(35.5)
    w = la.plan_windows(x)
    assert len(w) == 2 and w[0][1] == w[1][0] and w[1][1] == len(x)
    assert 25 * R <= w[0][1] <= 35 * R and w[0][1] <= len(x) - 5 * R + 0   # hi = min(35, 30.5)
    assert w[0][1] % la.H == 0


@pytest.mark.parametrize("sec", [35.001, 40, 61.3, 95, 200])
def test_tiling_and_lengths(sec):
    x = noise(sec, seed=int(sec))
    w = la.plan_windows(x)
    assert w[0][0] == 0 and w[-1][1] == len(x)
    for (a, b), (c, d) in zip(w, w[1:]):
        assert b == c
    for a, b in w[:-1]:
        assert 25 * R <= b - a <= 35 * R
    assert 5 * R <= w[-1][1] - w[-1][0] <= 35 * R


def test_all_zero_cuts_at_25s_each():
    w = la.plan_windows(np.zeros(100 * R, np.float32))
    assert w == [(0, 25 * R), (25 * R, 50 * R), (50 * R, 75 * R), (75 * R, 100 * R)]


def test_gap_gets_cut_inside():
    x = noise(70, seed=5)
    x[int(27.0 * R):int(30.0 * R)] = 0.0
    c = la.plan_windows(x)[0][1]
    assert 27 * R <= c <= 30 * R
    # earliest tie: first centre whose 200 ms window is fully inside the zeros
    assert c == la.H * (-(-(27 * R + 1600) // la.H))


def test_argmin_earliest_tie_hand_computed():
    # energy 1.0 everywhere except two equal quiet 200 ms spans (1.0 elsewhere); the earlier must win
    x = np.ones(60 * R, np.float32)
    x[int(28.0 * R) - 1600:int(28.0 * R) + 1600] = 0.0   # quiet centred at 28.0 s: centres 28.0 .. 28.0 exactly
    x[int(32.0 * R) - 1600:int(32.0 * R) + 1600] = 0.0
    assert la.plan_windows(x)[0][1] == int(28.0 * R)


def test_prefix_sums_hand_computed():
    p = la.prefix_sums(np.array([1.0, 2.0, -3.0], np.float32))
    assert p.dtype == np.float64 and list(p) == [0.0, 1.0, 5.0, 14.0]


def test_find_cut_uses_tail_limit():
    x = noise(36, seed=3)
    c = la.plan_windows(x)[0][1]
    assert 25 * R <= c <= 31 * R      # hi = min(35, 36 - 5) = 31 s


def test_600s_deterministic():
    x = noise(600, seed=7)
    a, b = la.plan_windows(x), la.plan_windows(x.copy())
    assert a == b and a[-1][1] == len(x)
    assert all(25 * R <= e - s <= 35 * R for s, e in a[:-1])


def test_fixed_plan():
    assert la.plan_fixed(70 * R) == [(0, 30 * R), (30 * R, 60 * R), (60 * R, 70 * R)]


def test_decode_windows_stitch():
    x = np.zeros(10 * R, np.float32)
    wins = [(0, 4 * R), (4 * R, 6 * R), (6 * R, 10 * R)]
    calls = iter([("hello", [["hello", 0.5, 1.0]]), (" ", []), ("world", [["world", 0.0, 0.2]])])

    def fn2(s):
        t, w = next(calls)
        return {"rms_dbfs": 0.0, "gated": False, "token_ids": [], "text": t, "words": w}
    dw, text, words = la.decode_windows(x, wins, fn2)
    assert text == "hello world"
    assert words == [["hello", 0.5, 1.0], ["world", 6.0, 6.2]]
    assert [w["start"] for w in dw] == [0, 4 * R, 6 * R]


# ---- goldens self-checks ----------------------------------------------------------------------------------------------

needs = pytest.mark.skipif(not os.path.isfile(GJSON), reason="long_goldens.json not present")


@needs
def test_goldens_sidecars_and_windows():
    with open(GJSON, encoding="utf-8") as f:
        d = json.load(f)
    assert d["schema"] == "stt-long-goldens/1"
    for r in d["recordings"]:
        path = os.path.join(GDIR, *r["audio"]["file"].split("/"))
        b = open(path, "rb").read()
        assert len(b) == 4 * r["audio"]["num_samples"]
        assert hashlib.sha256(b).hexdigest() == r["audio"]["sha256"]
        x = np.frombuffer(b, dtype="<f4")
        assert [(w["start"], w["end"]) for w in r["windows"]] == la.plan_windows(x)
    ids = {r["id"]: r for r in d["recordings"]}
    assert ids["silence_60s"]["text"] == "" and all(w["gated"] for w in ids["silence_60s"]["windows"])
    assert 33 * R <= ids["gap_70s"]["windows"][0]["end"] <= 36 * R


@needs
def test_goldens_accuracy_bar():
    with open(GJSON, encoding="utf-8") as f:
        w = json.load(f)["wer"]
    assert w["long"] <= w["per_utterance"] + 0.005
