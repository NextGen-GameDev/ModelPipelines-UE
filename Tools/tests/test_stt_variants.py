"""WP9S.1b tests: WER helper (hand-computed), fp16 weight conversion on a tiny graph, and built variant folders."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("soundfile", reason="needs the pipelines-stt env")
if not hasattr(pytest.importorskip("transformers"), "ParakeetForTDT"):
    pytest.skip("needs the pipelines-stt env (transformers with ParakeetForTDT)", allow_module_level=True)

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
from stt import phonon2_export as ex  # noqa: E402
from stt import phonon2_variants as pv  # noqa: E402
from stt import wer  # noqa: E402

DEV = _paths.dev_dir()
ROOT = os.path.join(DEV, "models", "stt", "phonon-2-onnx")
SPELLING = wer.default_spelling_path()


# ---- WER --------------------------------------------------------------------------------------------------------

def test_wer_identical_is_zero():
    r = wer.corpus_wer(["a b c d"], ["a b c d"])
    assert r["wer"] == 0.0 and r["N"] == 4


def test_wer_one_substitution_in_four_words():
    r = wer.corpus_wer(["the cat sat down"], ["the cat sit down"])
    assert (r["S"], r["D"], r["I"], r["N"]) == (1, 0, 0, 4)
    assert r["wer"] == 0.25


def test_wer_insertion_and_deletion():
    ins = wer.corpus_wer(["a b"], ["a x b"])
    assert (ins["S"], ins["D"], ins["I"]) == (0, 0, 1) and ins["wer"] == 0.5
    dele = wer.corpus_wer(["a b c d"], ["a b d"])
    assert (dele["S"], dele["D"], dele["I"]) == (0, 1, 0) and dele["wer"] == 0.25


def test_wer_empty_hypothesis_is_all_deletions():
    r = wer.corpus_wer(["one two three"], [""])
    assert (r["S"], r["D"], r["I"], r["N"]) == (0, 3, 0, 3) and r["wer"] == 1.0


def test_wer_is_corpus_level_not_mean_of_clips():
    # clip 1: 1 error / 1 word (100 %), clip 2: 0 errors / 9 words -> corpus 1/10, mean of clips would be 50 %
    r = wer.corpus_wer(["a", "b c d e f g h i j"], ["x", "b c d e f g h i j"])
    assert r["wer"] == pytest.approx(0.1)


@pytest.mark.skipif(not os.path.isfile(SPELLING), reason="normaliser spelling map not present")
def test_normalizer_matches_whisper_english_normalizer():
    import json
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    ref = EnglishTextNormalizer(json.load(open(SPELLING, encoding="utf-8")))
    nz = wer.make_normalizer(SPELLING)
    for s in ("Mr. Smith has a colour TV.", "He paid twenty five dollars", "Mr. Brown's COLOURS, twenty-five of them"):
        assert nz(s) == ref(s)
    assert nz("colour") == ref("colour")
    # the three cases the brief names must not be identity maps through the normaliser
    assert nz("Mr. Smith") != "Mr. Smith"


# ---- fp16 weight conversion on a tiny graph ---------------------------------------------------------------------

def test_fp16_weight_conversion_tiny_graph():
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    w_big = rng.standard_normal((64, 32)).astype(np.float32)          # 2048 elements: converted
    w_small = np.arange(4, dtype=np.float32)                          # 4 elements: untouched
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w_big"], ["y0"]), helper.make_node("Add", ["y0", "b_small"], ["y"])],
        "t", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        [numpy_helper.from_array(w_big, "w_big"), numpy_helper.from_array(np.resize(w_small, 32), "b_small")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    x = rng.standard_normal((1, 64)).astype(np.float32)
    want = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    conv, over = pv.convert_fp16_weights(m, min_elems=1024)
    assert conv == 1 and over == 0
    onnx.checker.check_model(m)
    kinds = {i.name: i.data_type for i in m.graph.initializer}
    assert kinds["w_big_fp16"] == TensorProto.FLOAT16 and kinds["b_small"] == TensorProto.FLOAT
    assert [n.op_type for n in m.graph.node] == ["Cast", "MatMul", "Add"]
    got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    # weights rounded to fp16: relative error per weight <= 2^-11, so the dot product is close but not identical
    assert np.abs(got - want).max() < 5e-2 and not np.array_equal(got, want)


# ---- built variants ---------------------------------------------------------------------------------------------

BUILT = [v for v in pv.VARIANTS if os.path.isfile(os.path.join(ROOT, v, "SOURCE.json"))]
FP32_DJ = os.path.join(ROOT, "fp32", "decoder_joint.onnx")


@pytest.mark.skipif(not BUILT, reason="no built variant folders")
@pytest.mark.parametrize("variant", BUILT)
def test_built_variant_meets_s1_and_c70(variant):
    folder = os.path.join(ROOT, variant)
    s1 = pv.check_s1(folder)
    assert s1["s1"], s1
    assert not os.path.exists(os.path.join(folder, "encoder.onnx.data"))
    ex.assert_c70(folder)
    import json
    src = json.load(open(os.path.join(folder, "SOURCE.json"), encoding="utf-8"))
    assert src["variant"] == variant and src["license"] == "cc-by-4.0"
    assert src["derived_from"]["variant"] == "fp32"
    assert src["files"]["decoder_joint.onnx"]["sha256"] == ex.sha256_file(FP32_DJ)


@pytest.mark.skipif(not BUILT, reason="no built variant folders")
@pytest.mark.parametrize("variant", BUILT)
def test_built_variant_decoder_joint_byte_identical(variant):
    assert ex.sha256_file(os.path.join(ROOT, variant, "decoder_joint.onnx")) == ex.sha256_file(FP32_DJ)
