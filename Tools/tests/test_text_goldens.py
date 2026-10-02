import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
import make_text_goldens as g  # noqa: E402

DEV = _paths.dev_dir()
GOLDEN = os.path.join(DEV, "goldens", "text", "text_goldens.json")
SENTENCES = os.path.join(TOOLS, "data", "text_sentences.json")
EXPECT = os.path.join(DEV, "goldens", "import", "import_expectations.json")
PY = sys.executable


# ---- pure helpers with hand-computed values ---------------------------------------------------------------------

def test_softmax_two_classes():
    # logits (0, ln 3) -> exp = (1, 3) -> (0.25, 0.75)
    s = g.softmax64([0.0, math.log(3.0)])
    assert s[0] == pytest.approx(0.25, abs=1e-12) and s[1] == pytest.approx(0.75, abs=1e-12)


def test_softmax_is_shift_invariant_and_stable():
    s = g.softmax64([1000.0, 1000.0])
    assert s.tolist() == [0.5, 0.5]


def test_sigmoid():
    s = g.sigmoid64([0.0, math.log(3.0)])
    assert s[0] == 0.5 and s[1] == pytest.approx(0.75, abs=1e-12)


def test_top_and_margin_ties_to_lower_index():
    assert g.top_and_margin([0.5, 0.5]) == (0, 0.0)
    idx, m = g.top_and_margin([0.1, 0.6, 0.3])
    assert idx == 1 and m == pytest.approx(0.3)


def test_pool_mean_masked():
    h = np.array([[1, 2], [3, 4], [100, 100]], dtype=np.float32)
    v = g.pool_np(h, [1, 1, 0], "Mean", False)
    assert v.tolist() == [2.0, 3.0] and v.dtype == np.float32


def test_pool_cls_normalized():
    h = np.array([[3, 4], [9, 9]], dtype=np.float32)
    assert g.pool_np(h, [1, 1], "CLS", True).tolist() == pytest.approx([0.6, 0.8])


def test_cosine():
    assert g.cosine64([1, 0], [0, 1]) == 0.0
    assert g.cosine64([1, 1], [2, 2]) == pytest.approx(1.0, abs=1e-15)
    assert g.cosine64([1, 0], [1, 1]) == pytest.approx(math.sqrt(0.5))


def test_similarity_top_ties_and_order():
    embs = [[1, 0], [0, 1], [1, 0], [1, 1]]
    top = g.similarity_top(embs, 0, 3)
    assert [t["index"] for t in top] == [0, 2, 3]  # tie 0 and 2 -> lower index first
    assert top[2]["score"] == pytest.approx(math.sqrt(0.5))


# ---- generated goldens ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc():
    if not os.path.isfile(GOLDEN):
        pytest.skip("run Tools/make_text_goldens.py first (%s missing)" % GOLDEN)
    with open(GOLDEN, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def expectations():
    if not os.path.isfile(EXPECT):
        pytest.skip("import expectations missing")
    with open(EXPECT, "r", encoding="utf-8") as f:
        return json.load(f)


def model(doc, slug):
    return next(m for m in doc["models"] if m["slug"] == slug)


def tokenizer_for(doc, m):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(doc["models_root"], m["folder"], "tokenizer.json"))


def test_visible_sentence_file_shape():
    with open(SENTENCES, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert d["schema"] == "pipelines-text-sentences/1"
    assert [s["id"] for s in d["sentences"]] == ["t%02d" % i for i in range(50)]


def test_schema_models_and_sentences(doc):
    assert doc["schema"] == "pipelines-text-goldens/1"
    assert [m["slug"] for m in doc["models"]] == ["all-MiniLM-L6-v2", "distilbert-sst2", "tiny_text_embedding",
                                                  "tiny_text_classification"]
    assert [m["reference"] for m in doc["models"]] == ["torch", "torch", "onnxruntime", "onnxruntime"]
    assert [m["folder"] for m in doc["models"]] == ["reference/all-MiniLM-L6-v2", "reference/distilbert-sst2",
                                                    "import_test/tiny_text_embedding",
                                                    "import_test/tiny_text_classification"]
    assert "/" in doc["models_root"] and "\\" not in doc["models_root"]
    for s in doc["sentences"]:
        assert s["text_utf8_hex"] == s["text"].encode("utf-8").hex()
    for m in doc["models"]:
        assert [c["sentence_id"] for c in m["cases"]] == [s["id"] for s in doc["sentences"]]
    for k in ("python", "torch", "transformers", "tokenizers", "onnxruntime", "numpy", "variant"):
        assert doc["generator"][k]


def test_visible_sentence_mix(doc):
    if doc["generator"]["variant"] != "visible":
        pytest.skip("content checks apply to the visible sentence file only")
    texts = [s["text"] for s in doc["sentences"]]
    assert [s["id"] for s in doc["sentences"]] == ["t%02d" % i for i in range(50)]
    with open(SENTENCES, "r", encoding="utf-8") as f:
        assert texts == [s["text"] for s in json.load(f)["sentences"]]
    assert sum(1 for t in texts if not t.isascii()) >= 10
    assert sum(1 for t in texts if t == "") == 1
    ws = [t for t in texts if t != "" and t.strip() == ""]
    assert len(ws) == 1 and " " in ws[0] and "\t" in ws[0]
    assert sum(1 for t in texts if t and not any(ch.isalpha() for ch in t)) >= 2
    mini, sst = model(doc, "all-MiniLM-L6-v2"), model(doc, "distilbert-sst2")
    assert sum(1 for c in mini["cases"] if len(c["input_ids"]) == mini["package"]["max_tokens"]) >= 2
    assert sum(1 for c in sst["cases"] if len(c["input_ids"]) == sst["package"]["max_tokens"]) >= 1
    tok = tokenizer_for(doc, sst)
    tok.no_truncation()
    assert sum(1 for t in texts if len(tok.encode(t).ids) > 512) >= 1


def test_input_ids_match_tokenizers(doc):
    for m in doc["models"]:
        tok = tokenizer_for(doc, m)
        tok.no_padding()
        tok.enable_truncation(max_length=m["package"]["max_tokens"])
        for s, c in zip(doc["sentences"], m["cases"]):
            assert tok.encode(s["text"], add_special_tokens=m["package"]["add_special_tokens"]).ids == c["input_ids"], \
                (m["slug"], s["id"])
            assert len(c["input_ids"]) <= m["package"]["max_tokens"]


def test_embedding_cases(doc):
    mini = model(doc, "all-MiniLM-L6-v2")
    tiny = model(doc, "tiny_text_embedding")
    for m, dim in ((mini, 384), (tiny, 4)):
        for c in m["cases"]:
            assert len(c["embedding"]) == dim
            assert abs(np.linalg.norm(np.asarray(c["embedding"], dtype=np.float64)) - 1.0) <= 1e-5
    for c in mini["cases"]:
        assert c["ort_cosine"] >= 0.99999 and c["ort_max_abs_diff"] < 1e-4
    for c in tiny["cases"]:
        assert c["ort_cosine"] == 1.0 and c["ort_max_abs_diff"] == 0.0
    assert "similarity" not in tiny


def test_classification_cases(doc):
    sst = model(doc, "distilbert-sst2")
    assert sst["package"]["score_activation"] == "Softmax"
    for c in sst["cases"]:
        assert abs(sum(c["scores"]) - 1.0) <= 1e-9
        assert c["label"] == sst["package"]["labels"][c["index"]]
        assert c["index"] == int(np.argmax(c["scores"]))
        assert c["top_margin"] >= 0.01
        assert c["ort_max_abs_diff"] <= 1e-4
    tiny = model(doc, "tiny_text_classification")
    assert tiny["package"]["score_activation"] == "Sigmoid"
    sums = []
    for c in tiny["cases"]:
        assert len(c["scores"]) == 3 and all(0.0 < s < 1.0 for s in c["scores"])
        expected = [1.0 / (1.0 + math.exp(-x)) for x in c["logits"]]
        assert c["scores"] == pytest.approx(expected, abs=1e-12)
        assert c["label"] == tiny["package"]["labels"][c["index"]]
        assert c["ort_max_abs_diff"] == 0.0
        sums.append(sum(c["scores"]))
    assert max(abs(s - 1.0) for s in sums) > 1e-6  # sigmoid scores do not sum to 1


def test_similarity_recomputed(doc):
    mini = model(doc, "all-MiniLM-L6-v2")
    sim = mini["similarity"]
    assert sim["k"] == 5 and len(sim["queries"]) == 5
    embs = [np.asarray(c["embedding"], dtype=np.float64) for c in mini["cases"]]
    ids = [s["id"] for s in doc["sentences"]]
    for q in sim["queries"]:
        qi = ids.index(q["sentence_id"])
        assert qi < 5
        scored = sorted(((float(np.dot(embs[qi], e) / (np.linalg.norm(embs[qi]) * np.linalg.norm(e))), i)
                         for i, e in enumerate(embs)), key=lambda t: (-t[0], t[1]))
        assert [t["index"] for t in q["top"]] == [i for _, i in scored[:5]]
        assert q["top"][0]["index"] == qi  # the query itself is its own best match


def test_torch_recomputation_three_sentences(doc):
    pytest.importorskip("torch")
    cache = os.path.join(DEV, "cache", "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"
    for slug, repo in (("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
                       ("distilbert-sst2", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")):
        m = model(doc, slug)
        net = g.load_torch(repo, m["source"]["revision"], m["task"], cache)
        twt = slug == "all-MiniLM-L6-v2"
        for idx in (0, 20, 41):  # a short query, a sentiment line, a truncated text (visible set)
            idx = min(idx, len(m["cases"]) - 1)
            c = m["cases"][idx]
            if m["task"] == "TextEmbedding":
                now = g.torch_embedding(net, c["input_ids"], m["package"], twt)
                assert np.max(np.abs(now.astype(np.float64) - np.asarray(c["embedding"]))) <= 1e-6
            else:
                now = g.torch_logits(net, c["input_ids"], twt)
                assert np.max(np.abs(now.astype(np.float64) - np.asarray(c["logits"]))) <= 1e-6


def test_package_blocks_equal_import_expectations(doc, expectations):
    for m in doc["models"]:
        assert m["package"] == g.expected_package(expectations, m["folder"])
        assert m["package"] == g.derive_package(os.path.join(doc["models_root"], m["folder"]), m["task"])


def test_alternate_sentence_file_works(tmp_path):
    if not os.path.isdir(os.path.join(DEV, "models", "reference", "all-MiniLM-L6-v2")):
        pytest.skip("reference models missing")
    src = tmp_path / "s.json"
    texts = ["The door is locked.", "Wonderful news!", "", "Привет, мир",
             "I can't stand this awful noise."]
    src.write_text(json.dumps({"schema": "pipelines-text-sentences/1",
                               "sentences": [{"id": "h%02d" % i, "text": t} for i, t in enumerate(texts)]}),
                   encoding="utf-8")
    outs = []
    for n in (1, 2):
        out = tmp_path / ("o%d.json" % n)
        r = subprocess.run([PY, os.path.join(TOOLS, "make_text_goldens.py"), "--sentences", str(src), "--out",
                            str(out), "--variant", "hidden"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]  # deterministic
    d = json.loads(outs[0].decode("utf-8"))
    assert d["generator"]["variant"] == "hidden" and len(d["sentences"]) == 5
    assert [c["sentence_id"] for c in d["models"][0]["cases"]] == ["h00", "h01", "h02", "h03", "h04"]
    assert len(d["models"][0]["similarity"]["queries"]) == 5
