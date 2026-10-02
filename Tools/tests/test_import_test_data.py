import json
import os
import sys

import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
import make_import_test_data as m  # noqa: E402

GOLDEN = os.path.join(_paths.dev_dir(), "goldens", "import", "import_expectations.json")
F32 = 0.003921568859368563  # float32(1/255)


# ---- rule helpers on hand-written dicts ------------------------------------------------------------------------

def test_convnext_crop_pct():
    r = m.image_preprocess({"size": 224, "crop_pct": 0.875})
    assert r["resize_mode"] == "ShortestEdgeCenterCrop"
    assert r["resize_shortest_edge"] == 256  # int(224 / 0.875)
    assert (r["image_width"], r["image_height"]) == (224, 224)


def test_shortest_edge_dict_with_crop_pct():
    r = m.image_preprocess({"size": {"shortest_edge": 48}, "crop_pct": 0.9})
    assert r["resize_shortest_edge"] == 53  # int(53.33)
    assert r["image_width"] == 48


def test_size_384_is_stretch():
    r = m.image_preprocess({"size": 384, "crop_pct": 0.875})
    assert r["resize_mode"] == "Stretch"
    assert (r["image_width"], r["image_height"], r["resize_shortest_edge"]) == (384, 384, 0)


def test_height_width_stretch_w_by_h():
    r = m.image_preprocess({"size": {"height": 48, "width": 40}})
    assert (r["image_width"], r["image_height"], r["resize_mode"]) == (40, 48, "Stretch")


def test_crop_size_path():
    r = m.image_preprocess({"size": {"shortest_edge": 256}, "do_center_crop": True,
                            "crop_size": {"height": 200, "width": 224}})
    assert r["resize_mode"] == "ShortestEdgeCenterCrop"
    assert r["resize_shortest_edge"] == 256
    assert (r["image_width"], r["image_height"]) == (224, 200)


def test_plain_integer_size_stretch():
    r = m.image_preprocess({"size": 100})
    assert (r["resize_mode"], r["image_width"], r["image_height"]) == ("Stretch", 100, 100)


def test_rescale_and_normalize_flags():
    r = m.image_preprocess({"size": 10, "do_rescale": False, "do_normalize": False,
                            "image_mean": [0.1, 0.1, 0.1], "image_std": [0.2, 0.2, 0.2]})
    assert r["rescale_factor"] == 1.0
    assert r["image_mean"] == [0.0, 0.0, 0.0] and r["image_std"] == [1.0, 1.0, 1.0]


def test_defaults_without_preprocessor_file():
    r = m.image_preprocess(None)
    assert r["rescale_factor"] == F32
    assert r["image_mean"] == [0.485, 0.456, 0.406] and r["image_std"] == [0.229, 0.224, 0.225]
    assert (r["image_width"], r["image_height"], r["resize_mode"], r["resize_shortest_edge"]) == (224, 224, "Stretch", 0)


def test_max_tokens_sentinel_falls_through():
    cfg = {"max_position_embeddings": 128}
    assert m.text_max_tokens(None, {"model_max_length": 1e30}, cfg) == 128
    assert m.text_max_tokens(None, {"model_max_length": 1000000000000000019884624838656}, None) == 512
    assert m.text_max_tokens(None, {"model_max_length": 65537}, None) == 512
    assert m.text_max_tokens(None, {"model_max_length": 65536}, None) == 65536
    assert m.text_max_tokens(None, {"model_max_length": 0}, cfg) == 128


def test_max_tokens_priority_and_min():
    assert m.text_max_tokens({"max_seq_length": 64}, {"model_max_length": 512}, {"max_position_embeddings": 128}) == 64
    assert m.text_max_tokens({"max_seq_length": 300}, None, {"max_position_embeddings": 128}) == 128
    assert m.text_max_tokens(None, {"model_max_length": 256}, {"max_position_embeddings": 512}) == 256
    assert m.text_max_tokens(None, {"model_max_length": 256}, None) == 256


def test_license_normalization():
    assert m.normalize_license("apache-2.0") == "Apache-2.0"
    assert m.normalize_license("MIT") == "MIT"
    assert m.normalize_license("'bsd-3-clause'") == "BSD-3-Clause"
    assert m.normalize_license('"bsd-2-clause"') == "BSD-2-Clause"
    assert m.normalize_license("gpl-3.0") == "gpl-3.0"
    assert m.normalize_license(None) == ""


def test_readme_front_matter():
    assert m.readme_license("---\nlicense: mit\ntags:\n- test\n---\n# Tiny\n") == "mit"
    assert m.readme_license('---\nlicense: "apache-2.0"\n---\n') == '"apache-2.0"'
    assert m.normalize_license(m.readme_license('---\nlicense: "apache-2.0"\n---\n')) == "Apache-2.0"
    assert m.readme_license("# no front matter\nlicense: mit\n") is None
    assert m.readme_license("---\ntags:\n- a\n---\nlicense: mit\n") is None


def test_id2label():
    assert m.parse_labels({"0": "a", "1": "b", "2": "c"}) == (["a", "b", "c"], None)
    assert m.parse_labels({"1": "b", "0": "a"}) == (["a", "b"], None)
    assert m.parse_labels({"0": "a", "2": "b", "3": "c"}) == (None, "InvalidLabels")
    assert m.parse_labels({"1": "a"}) == (None, "InvalidLabels")
    assert m.parse_labels({"x": "a"}) == (None, "InvalidLabels")
    assert m.parse_labels(None) == ([], None)


def test_score_activation():
    assert m.score_activation("TextClassification", {"problem_type": "multi_label_classification"}) == "Sigmoid"
    assert m.score_activation("TextClassification", {}) == "Softmax"
    assert m.score_activation("ObjectDetection", {"model_type": "rt_detr"}) == "Sigmoid"
    assert m.score_activation("ObjectDetection", {"model_type": "yolos"}) == "Softmax"
    assert m.score_activation("ObjectDetection", {"model_type": "other"}) == "Sigmoid"
    assert m.score_activation("TextEmbedding", {}) == "Softmax"


def test_detect_task_order():
    cfg = {"architectures": ["DistilBertForSequenceClassification"]}
    assert m.detect_task("ObjectDetection", {"task": "textembedding"}, True, cfg, True) == ("ObjectDetection", "options")
    assert m.detect_task("None", {"task": "textembedding"}, True, cfg, True) == ("TextEmbedding", "modelpipelines.json")
    assert m.detect_task("None", {}, True, cfg, True) == ("TextEmbedding", "modules.json")
    assert m.detect_task("None", None, False, cfg, False) == ("TextClassification", "architectures")
    assert m.detect_task("None", None, False, {"architectures": ["BertModel"]}, True) == ("TextEmbedding", "architectures")
    assert m.detect_task("None", None, False, {"architectures": ["BertModel"]}, False) == (None, None)
    assert m.detect_task("None", None, False, {"model_type": "resnet"}, False) == (None, None)


def test_pooling_from_modules():
    mods = [{"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Normalize", "path": "2_Normalize"}]
    assert m.pooling_and_normalize(mods, lambda p: {"pooling_mode_cls_token": True} if p == "1_Pooling/config.json"
                                   else None) == ("CLS", True)
    assert m.pooling_and_normalize(mods[:1], lambda p: {"pooling_mode_mean_tokens": True}) == ("Mean", False)
    assert m.pooling_and_normalize([], lambda p: None) == ("Mean", False)


# ---- generated data (skipped if not generated) --------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc():
    if not os.path.isfile(GOLDEN):
        pytest.skip("run make_import_test_data.py first")
    with open(GOLDEN, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_keys(doc):
    assert doc["schema"] == "pipelines-import-expectations/1"
    assert set(doc["generator"]) == {"script", "variant", "transformers", "onnx", "onnxruntime"}
    assert os.path.isabs(doc["models_root"])
    ids = [c["id"] for c in doc["cases"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 15
    base = {"id", "folder", "reference", "task_override", "expect", "error_codes"}
    full = base | {"warning_codes", "task", "task_source", "model_file", "labels", "preprocess", "license",
                   "preferred_device", "onnx"}
    pre = {"max_tokens", "add_special_tokens", "pooling", "normalize_embeddings", "image_width", "image_height",
           "resize_mode", "resize_shortest_edge", "rescale_factor", "image_mean", "image_std", "score_activation"}
    for c in doc["cases"]:
        assert c["expect"] in ("valid", "validation_error", "parse_error")
        assert set(c) == (base if c["expect"] == "parse_error" else full), c["id"]
        if c["expect"] != "parse_error":
            assert set(c["preprocess"]) == pre
            assert set(c["license"]) == {"spdx_id", "source_url", "attribution"}
            assert set(c["onnx"]) == {"inputs", "outputs"}


def _root(doc, c):
    return doc["models_root"] if c["reference"] else doc.get("import_root", doc["models_root"])


def test_every_case_folder_exists(doc):
    for c in doc["cases"]:
        assert os.path.isdir(os.path.join(_root(doc, c), c["folder"])), c["id"]


def test_valid_tiny_onnx_blocks_match_files(doc):
    n = 0
    for c in doc["cases"]:
        if c["expect"] != "valid" or c["reference"]:
            continue
        path = os.path.join(_root(doc, c), c["folder"], c["model_file"])
        assert m.onnx_block(path) == c["onnx"], c["id"]
        n += 1
    assert n == 7


def test_tiny_models_check_and_run(doc):
    import onnx
    import onnxruntime as ort
    import numpy as np
    for c in doc["cases"]:
        if c["reference"] or c["expect"] == "parse_error" and c["id"] == "missing_model":
            continue
        folder = os.path.join(_root(doc, c), c["folder"])
        mp = os.path.join(folder, "model.onnx")
        if not os.path.isfile(mp):
            mp = os.path.join(folder, "onnx", "model.onnx")
        if c["id"] == "unknown_op":
            with pytest.raises(Exception):
                ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
            continue
        onnx.checker.check_model(onnx.load(mp))
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        feeds = {}
        for i in sess.get_inputs():
            shape = [2 if not isinstance(s, int) else s for s in i.shape]
            if i.type == "tensor(int64)":
                feeds[i.name] = np.ones(shape, dtype=np.int64)
            else:
                feeds[i.name] = np.random.RandomState(1).rand(*shape).astype(np.float32)
        outs = sess.run(None, feeds)
        assert all(o.shape[0] == 2 for o in outs), c["id"]


@pytest.mark.parametrize("slug", ["all-MiniLM-L6-v2", "distilbert-sst2"])
def test_reference_text_folders_have_loadable_tokenizer_json(slug):
    path = os.path.join(_paths.dev_dir(), "models", "reference", slug, "tokenizer.json")
    if not os.path.isfile(os.path.join(os.path.dirname(path), "model.onnx")):
        pytest.skip("run fetch_reference_models.py first")
    assert os.path.isfile(path), "missing tokenizer.json in " + slug
    from tokenizers import Tokenizer
    assert Tokenizer.from_file(path).get_vocab_size() > 1000


def test_license_unknown_kept_as_written():
    assert m.normalize_license("other") == "other"
    assert m.normalize_license('"cc-by-nc-4.0"') == "cc-by-nc-4.0"
    assert m.normalize_license("'Apache-2.0'") == "Apache-2.0"
    assert m.normalize_license("") == ""


def test_license_per_field_fallback():
    readme = "---\nlicense: mit\n---\n"
    lic = m.resolve_license({"license": {"spdx_id": "  ", "source_url": "u", "attribution": "a"}}, readme)
    assert lic == {"spdx_id": "MIT", "source_url": "u", "attribution": "a"}
    lic = m.resolve_license({"license": {"source_url": "u"}}, readme)
    assert lic == {"spdx_id": "MIT", "source_url": "u", "attribution": ""}
    lic = m.resolve_license({"license": {"spdx_id": "BSD-3-Clause"}}, readme)
    assert lic["spdx_id"] == "BSD-3-Clause" and lic["source_url"] == ""
    assert m.resolve_license(None, readme) == {"spdx_id": "MIT", "source_url": "", "attribution": ""}
    assert m.resolve_license({"license": {"spdx_id": ""}}, None)["spdx_id"] == ""


def test_preferred_device_case_insensitive():
    assert m.resolve_device({"preferred_device": "gpu"}) == "GPU"
    assert m.resolve_device({"preferred_device": "Cpu"}) == "CPU"
    assert m.resolve_device({"preferred_device": "AUTO"}) == "Auto"
    assert m.resolve_device({}) == "Auto"
    assert m.resolve_device(None) == "Auto"
    with pytest.raises(ValueError):
        m.resolve_device({"preferred_device": "tpu"})


def test_large_size_without_crop_pct_follows_other_rules():
    r = m.image_preprocess({"size": {"shortest_edge": 512}, "do_center_crop": True, "crop_size": 448})
    assert r["resize_mode"] == "ShortestEdgeCenterCrop" and r["resize_shortest_edge"] == 512
    assert (r["image_width"], r["image_height"]) == (448, 448)
    r = m.image_preprocess({"size": 400})
    assert r["resize_mode"] == "Stretch" and (r["image_width"], r["image_height"]) == (400, 400)
    r = m.image_preprocess({"size": 384, "crop_pct": 0.9})
    assert r["resize_mode"] == "Stretch" and r["resize_shortest_edge"] == 0


# ---- WP3.6b worst-case cases ----------------------------------------------------------------------------------------

def _gather_rows(model_path, init_name):
    import onnx
    mdl = onnx.load(model_path)
    for i in mdl.graph.initializer:
        if i.name == init_name:
            return list(i.dims)[0]
    raise AssertionError("no initializer " + init_name)


def _check_worst_case_folders(models_root, goldens_path, variant):
    with open(goldens_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    by_id = {c["id"]: c for c in doc["cases"]}
    emb = by_id["tiny_text_embedding"]
    tok_name = "bert-base-uncased" if variant == "visible" else "t5-small"
    tok = os.path.join(_paths.dev_dir(), "models", "tokenizers", tok_name, "tokenizer.json")
    vocab = m.tokenizer_vocab(tok)
    rows = {"small_vocab": 30000 if variant == "visible" else vocab - 1000,
            "short_positions": None}
    pos_rows = 32 if variant == "visible" else 16
    max_tokens = 64 if variant == "visible" else 48
    for case_id in ("small_vocab", "short_positions"):
        c = by_id[case_id]
        assert c["expect"] == "validation_error" and c["error_codes"] == ["SmokeRunFailed"]
        assert c["warning_codes"] == [] and c["task"] == "TextEmbedding"
        assert c["task_source"] == "modules.json"
        assert c["onnx"] == emb["onnx"]
        assert c["license"] == {"spdx_id": "Apache-2.0", "source_url": "https://example.invalid/" + case_id,
                                "attribution": "ModelPipelines test data"}
        folder = os.path.join(models_root, "import_test", case_id)
        assert os.path.isdir(folder)
        mp = os.path.join(folder, "model.onnx")
        assert _gather_rows(mp, "table") == (rows[case_id] or vocab)
        hello_ok, worst_ok, _ = m.worst_case_run(mp, tok, c["preprocess"]["max_tokens"])
        assert hello_ok and not worst_ok, case_id
    assert by_id["small_vocab"]["preprocess"]["max_tokens"] == (64 if variant == "visible" else 32)
    sp = by_id["short_positions"]
    assert sp["preprocess"]["max_tokens"] == max_tokens
    assert _gather_rows(os.path.join(models_root, "import_test", "short_positions", "model.onnx"),
                        "pos_table") == pos_rows
    with open(os.path.join(models_root, "import_test", "short_positions", "config.json"), encoding="utf-8") as f:
        assert "max_position_embeddings" not in json.load(f)


def test_worst_case_cases_visible(doc):
    models_root = doc.get("import_root", doc["models_root"])
    _check_worst_case_folders(models_root, GOLDEN, "visible")


def test_worst_case_cases_hidden_variant(tmp_path):
    import subprocess
    out = tmp_path / "models"
    gold = tmp_path / "import_expectations.json"
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_import_test_data.py"), "--variant", "hidden",
                        "--out-models", str(out), "--out-goldens", str(gold)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if "NEEDS:" in (r.stderr + r.stdout):
        pytest.skip("hidden tokenizers missing")
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    _check_worst_case_folders(str(out), str(gold), "hidden")


@pytest.mark.parametrize("slug", ["all-MiniLM-L6-v2", "distilbert-sst2"])
def test_reference_text_models_pass_worst_case(doc, slug):
    folder = os.path.join(_paths.dev_dir(), "models", "reference", slug)
    c = next((c for c in doc["cases"] if c["id"] == slug), None)
    if c is None:
        pytest.skip("run fetch_reference_models.py first")
    hello_ok, worst_ok, err = m.worst_case_run(os.path.join(folder, c["model_file"]),
                                               os.path.join(folder, "tokenizer.json"), c["preprocess"]["max_tokens"])
    assert hello_ok and worst_ok, err
