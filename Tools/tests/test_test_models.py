import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import onnx
import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)

import make_test_models as mtm  # noqa: E402

MK_MODELS = os.path.join(TOOLS, "make_test_models.py")
MK_GOLDENS = os.path.join(TOOLS, "make_test_goldens.py")
EDGE = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-3, -1e6, 1e6, 8.5e37]


def run(script, *args, dev=None):
    env = dict(os.environ)
    if dev is not None:
        env["MODELPIPELINES_DEV_DIR"] = str(dev)
    return subprocess.run([sys.executable, script, *map(str, args)], env=env,
                          capture_output=True, text=True)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("built")
    mdir = d / "models"
    r = run(MK_MODELS, "--out", mdir)
    assert r.returncode == 0, r.stderr
    gpath = d / "g.json"
    r = run(MK_GOLDENS, "--models", mdir, "--out", gpath)
    assert r.returncode == 0, r.stderr
    with open(gpath, encoding="utf-8") as f:
        return mdir, json.load(f), gpath


def sig(vis):
    out = []
    for v in vis:
        t = v.type.tensor_type
        dims = [d.dim_param if d.dim_param else d.dim_value for d in t.shape.dim]
        out.append((v.name, t.elem_type, dims))
    return out


F, I64, I32, F16 = (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64,
                    onnx.TensorProto.INT32, onnx.TensorProto.FLOAT16)
EXPECT = {
    "affine_dynamic.onnx": ([("x", F, ["batch", 4])], [("y", F, ["batch", 4])]),
    "affine_static.onnx": ([("x", F, [2, 4])], [("y", F, [2, 4])]),
    "int_ops.onnx": ([("ids", I64, ["batch", "seq"]), ("mask", I32, ["batch", "seq"])],
                     [("ids_plus_one", I64, ["batch", "seq"]), ("mask_half", F, ["batch", "seq"])]),
    "half_io.onnx": ([("h", F16, [2])], [("h_out", F16, [2])]),
    "tiny_stt_encoder.onnx": ([("input_features", F, [1, "frames", 8])],
                              [("encoder_projected", F, [1, "frames", 4])]),
    "tiny_stt_decoder_joint.onnx": (
        [("encoder_frame", F, [1, 4]), ("token", I64, [1]),
         ("state_h", F, [2, 1, 4]), ("state_c", F, [2, 1, 4])],
        [("token_logits", F, [1, 6]), ("duration_logits", F, [1, 3]),
         ("state_h_out", F, [2, 1, 4]), ("state_c_out", F, [2, 1, 4])]),
    "tiny_stt_bad_decoder.onnx": (
        [("encoder_frame", F, [1, 4]), ("token", F, [1]),
         ("state_h", F, [2, 1, 4]), ("state_c", F, [2, 1, 4])],
        [("token_logits", F, [1, 6]), ("duration_logits", F, [1, 3]),
         ("state_h_out", F, [2, 1, 4]), ("state_c_out", F, [2, 1, 4])]),
}
OLD_SHA = {
    "affine_dynamic.onnx": "9b0bdee58397fc137abbbb08ec711ee2e72c42839325a268affecd50984746d9",
    "affine_static.onnx": "760a26d3eeeda61e2e90ccdd4756f71d5b3ccc87e0989223518020c1e4e77c92",
    "int_ops.onnx": "1f5e05ec86f9b3304c262dfc09fd0182286b6c272fdc8410ecb418b71f4cafdd",
    "half_io.onnx": "025653452645dd619b100ae20eaffeba2234875e91f6d6448c35665f4e40ae84",
}


@pytest.mark.parametrize("fn", list(EXPECT))
def test_model_structure(built, fn):
    m = onnx.load(os.path.join(built[0], fn))
    onnx.checker.check_model(m)
    assert m.ir_version == 8
    assert [(o.domain, o.version) for o in m.opset_import] == [("", 17)]
    assert m.producer_name == "ModelPipelines-Tools"
    assert (sig(m.graph.input), sig(m.graph.output)) == EXPECT[fn]


def test_models_deterministic(built, tmp_path):
    r = run(MK_MODELS, "--out", tmp_path)
    assert r.returncode == 0
    for fn in EXPECT:
        assert sha(tmp_path / fn) == sha(os.path.join(built[0], fn))
        assert fn in r.stdout


def by_id(doc, fn):
    m = next(m for m in doc["models"] if m["file"] == fn)
    return m, {c["id"]: c for c in m["cases"]}


def arr(t, dtype):
    return np.array(t["data"], dtype=dtype).reshape(t["shape"])


def test_schema_and_lengths(built):
    doc = built[1]
    assert doc["schema"] == "pipelines-test-model-goldens/1"
    g = doc["generator"]
    assert g["seed"] == 1234 and g["batches"] == [1, 3, 8, 256]
    assert [m["file"] for m in doc["models"]] == list(EXPECT)
    for m in doc["models"]:
        for c in m["cases"]:
            for t in c["inputs"] + c["outputs"]:
                assert len(t["data"]) == math.prod(t["shape"])


def test_model_level_shapes(built):
    m, _ = by_id(built[1], "int_ops.onnx")
    assert m["inputs"] == [{"name": "ids", "type": "int64", "shape": [-1, -1]},
                           {"name": "mask", "type": "int32", "shape": [-1, -1]}]
    assert m["outputs"][1] == {"name": "mask_half", "type": "float32", "shape": [-1, -1]}
    m, _ = by_id(built[1], "affine_dynamic.onnx")
    assert m["inputs"][0]["shape"] == [-1, 4]
    m, cases = by_id(built[1], "half_io.onnx")
    assert cases == {} and m["inputs"][0]["type"] == "float16"


def test_affine_values(built):
    doc = built[1]
    for fn in ("affine_dynamic.onnx", "affine_static.onnx"):
        _, cases = by_id(doc, fn)
        for c in cases.values():
            x = arr(c["inputs"][0], np.float32)
            y = arr(c["outputs"][0], np.float32)
            exp = x * np.float32(2) + np.float32(1)
            assert x.tobytes() == x.astype(np.float32).tobytes()
            assert y.tobytes() == exp.tobytes()
    _, cases = by_id(doc, "affine_dynamic.onnx")
    assert cases["batch3"]["outputs"][0]["data"][:4] == [1.0, 1.0, 3.0, -1.0]


def test_float_roundtrip(built):
    # the text in the file must parse back to exactly the float32 value
    with open(built[2], encoding="utf-8") as f:
        doc = json.load(f)
    for c in by_id(doc, "affine_dynamic.onnx")[1].values():
        for t in c["inputs"] + c["outputs"]:
            for v in t["data"]:
                assert float(np.float32(v)) == v


def test_int_ops_values(built):
    _, cases = by_id(built[1], "int_ops.onnx")
    assert set(cases) == {"ids1x5", "ids2x3"}
    for c in cases.values():
        ids = arr(c["inputs"][0], np.int64)
        mask = arr(c["inputs"][1], np.int32)
        assert np.array_equal(arr(c["outputs"][0], np.int64), ids + 1)
        assert arr(c["outputs"][1], np.float32).tobytes() == (mask.astype(np.float32) * np.float32(0.5)).tobytes()
        assert set(mask.flatten().tolist()) <= {0, 1}
        assert all(abs(v) <= 2 ** 52 for v in c["inputs"][0]["data"])
    c = cases["ids1x5"]
    assert c["inputs"][0]["shape"] == [1, 5]
    assert c["inputs"][0]["data"][:2] == [1099511627776, -1099511627776]
    assert cases["ids2x3"]["inputs"][0]["shape"] == [2, 3]


def test_case_ids_shapes_edges(built):
    _, cases = by_id(built[1], "affine_dynamic.onnx")
    assert list(cases) == ["batch1", "batch3", "batch8", "batch256"]
    for b in (1, 3, 8, 256):
        assert cases[f"batch{b}"]["inputs"][0]["shape"] == [b, 4]
    assert cases["batch3"]["inputs"][0]["data"][:9] == [float(np.float32(v)) for v in EDGE]
    assert math.copysign(1, cases["batch3"]["inputs"][0]["data"][1]) == -1
    _, st = by_id(built[1], "affine_static.onnx")
    assert list(st) == ["static2x4"] and st["static2x4"]["inputs"][0]["shape"] == [2, 4]


def test_sha256_fields(built):
    for m in built[1]["models"]:
        assert m["sha256"] == sha(os.path.join(built[0], m["file"]))


def test_seed_repeatable_and_different(built, tmp_path):
    mdir = built[0]
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    assert run(MK_GOLDENS, "--models", mdir, "--out", a).returncode == 0
    assert run(MK_GOLDENS, "--models", mdir, "--out", b).returncode == 0
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() == open(built[2], "rb").read()
    r = run(MK_GOLDENS, "--models", mdir, "--out", c, "--seed", 99, "--batches", "2,5")
    assert r.returncode == 0, r.stderr
    doc = json.loads(c.read_text(encoding="utf-8"))
    _, cases = by_id(doc, "affine_dynamic.onnx")
    assert list(cases) == ["batch2", "batch5"]
    assert doc["generator"]["seed"] == 99 and doc["generator"]["batches"] == [2, 5]
    _, base = by_id(built[1], "int_ops.onnx")
    _, other = by_id(doc, "int_ops.onnx")
    assert base["ids1x5"]["inputs"][1]["data"] != other["ids1x5"]["inputs"][1]["data"] or \
        base["ids2x3"]["inputs"][0]["data"] != other["ids2x3"]["inputs"][0]["data"]


def test_missing_models_fails(tmp_path):
    r = run(MK_GOLDENS, "--models", tmp_path / "none", "--out", tmp_path / "o.json")
    assert r.returncode != 0
    assert "missing models" in (r.stderr + r.stdout)
    assert not (tmp_path / "o.json").exists()


def test_dev_dir_env(tmp_path):
    assert run(MK_MODELS, dev=tmp_path).returncode == 0
    assert (tmp_path / "models" / "test" / "half_io.onnx").is_file()
    r = run(MK_GOLDENS, dev=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "goldens" / "test_models" / "test_models.json").is_file()
    assert not any(n.endswith(".onnx") for n in os.listdir(TOOLS))


def mm(a, b):
    # numpy's BLAS matmul crashes natively in this env; use a plain broadcast sum (float64 accumulate)
    return (a.astype(np.float64)[..., :, None] * b.astype(np.float64)).sum(-2)


@pytest.mark.parametrize("fn", list(OLD_SHA))
def test_existing_models_bytes_unchanged(built, fn):
    assert sha(os.path.join(built[0], fn)) == OLD_SHA[fn]


def test_tiny_stt_runs_in_ort(built):
    import onnxruntime as ort
    mdir, doc, _ = built
    for fn in ("tiny_stt_encoder.onnx", "tiny_stt_decoder_joint.onnx", "tiny_stt_bad_decoder.onnx"):
        ort.InferenceSession(os.path.join(mdir, fn), providers=["CPUExecutionProvider"])
    _, enc = by_id(doc, "tiny_stt_encoder.onnx")
    assert list(enc) == ["frames1", "frames3", "frames7"]
    for T in (1, 3, 7):
        c = enc[f"frames{T}"]
        assert c["inputs"][0]["shape"] == [1, T, 8] and c["outputs"][0]["shape"] == [1, T, 4]
        x = arr(c["inputs"][0], np.float32)
        w = mtm.W_ENC
        assert np.allclose(arr(c["outputs"][0], np.float32), mm(x, w), rtol=1e-5, atol=1e-5)
    m, bad = by_id(doc, "tiny_stt_bad_decoder.onnx")
    assert bad == {} and m["inputs"][1]["type"] == "float32"


def test_tiny_stt_decoder_math(built):
    m, dec = by_id(built[1], "tiny_stt_decoder_joint.onnx")
    assert list(dec) == ["blank_zero_state", "token0", "token4", "nonzero_state"]
    assert m["inputs"][1] == {"name": "token", "type": "int64", "shape": [1]}
    for c in dec.values():
        fr = arr(c["inputs"][0], np.float32)
        tok = int(c["inputs"][1]["data"][0])
        sh, sc = arr(c["inputs"][2], np.float32), arr(c["inputs"][3], np.float32)
        o = {t["name"]: arr(t, np.float32) for t in c["outputs"]}
        assert np.allclose(o["token_logits"], mm(fr, mtm.W1) + mtm.E[tok][None], atol=1e-6)
        assert np.allclose(o["duration_logits"], mm(fr, mtm.W3), atol=1e-6)
        assert np.allclose(o["state_h_out"], sh + 1, atol=1e-6)
        assert np.allclose(o["state_c_out"], sc * 0.5, atol=1e-6)
    assert dec["blank_zero_state"]["inputs"][1]["data"] == [5]
    assert any(v != 0 for v in dec["nonzero_state"]["inputs"][2]["data"])


def test_tiny_stt_greedy(built):
    import onnxruntime as ort
    mdir, doc, _ = built
    g = doc["tiny_stt_greedy"]
    assert g["num_mel_bins"] == 8 and g["blank_token_id"] == 5 and g["durations"] == [0, 1, 2]
    assert len(g["frames"]) == 6 and all(len(f) == 4 for f in g["frames"])
    steps = g["steps"]
    # recompute with the C70 loop from the ONNX file, independent of the script's helper
    sess = ort.InferenceSession(os.path.join(mdir, "tiny_stt_decoder_joint.onnx"),
                                providers=["CPUExecutionProvider"])
    F = len(g["frames"])
    t, token, steps2 = 0, 5, []
    sh = sc = np.zeros((2, 1, 4), np.float32)
    forced = False
    while t < F and len(steps2) < 10 * F - 1:
        tl, dl, h2, c2 = sess.run(None, {"encoder_frame": np.array([g["frames"][t]], np.float32),
                                         "token": np.array([token], np.int64),
                                         "state_h": sh, "state_c": sc})
        k, d = int(np.argmax(tl)), [0, 1, 2][int(np.argmax(dl))]
        if k == 5 and d == 0:
            d, forced = 1, True
        steps2.append([k, d])
        if k != 5:
            token, sh, sc = k, h2, c2
        t += d
    assert steps == steps2
    assert g["ids"] == [k for k, d in steps if k != 5]
    assert any(k != 5 and d == 0 for k, d in steps)
    assert forced
    assert any(d == 2 for k, d in steps)
    assert len(steps) < 10 * F - 1
