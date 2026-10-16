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
    assert len(ids) == len(set(ids)) and len(ids) >= 20
    base = {"id", "folder", "reference", "task_override", "expect", "error_codes"}
    full = base | {"warning_codes", "task", "task_source", "model_file", "labels", "preprocess", "license",
                   "preferred_device", "onnx", "decoder_model_file", "speech"}
    pre = {"max_tokens", "add_special_tokens", "pooling", "normalize_embeddings", "image_width", "image_height",
           "resize_mode", "resize_shortest_edge", "resize_width", "resize_height",
           "rescale_factor", "image_mean", "image_std", "score_activation",
           "resample", "nms_iou_threshold", "max_detections"}
    for c in doc["cases"]:
        assert c["expect"] in ("valid", "validation_error", "parse_error")
        assert set(c) - {"error_message_contains", "parse_only"} == (
            base if c["expect"] == "parse_error" else full), c["id"]
        if c["expect"] != "parse_error":
            assert set(c["preprocess"]) == pre
            assert set(c["license"]) == {"spdx_id", "source_url", "attribution"}
            assert set(c["onnx"]) == {"inputs", "outputs"}
            assert set(c["speech"]) == set(m.SPEECH_DEFAULTS), c["id"]


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
    assert n == 17 + 3  # 12 + the 5 WP4.4a cases + the 3 valid speech cases


def test_tiny_models_check_and_run(doc):
    import onnx
    import onnxruntime as ort
    import numpy as np
    for c in doc["cases"]:
        if c["reference"] or c["expect"] == "parse_error" and c["id"] == "missing_model":
            continue
        if c["id"].startswith("stt_"):
            continue  # speech folders: see test_stt_cases (encoder/decoder have a fixed batch of 1)
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


# ---- WP4.0a / C25 resize-then-crop ------------------------------------------------------------------------------------

def _fields(r):
    return (r["resize_mode"], r["resize_width"], r["resize_height"], r["resize_shortest_edge"],
            r["image_width"], r["image_height"])


def test_c25_deit_int_size_with_crop():
    r = m.image_preprocess({"image_processor_type": "DeiTImageProcessor", "size": 256, "do_center_crop": True,
                            "crop_size": 224})
    assert _fields(r) == ("StretchCenterCrop", 256, 256, 0, 224, 224)


def test_c25_legacy_feature_extractor_name():
    assert m.processor_class_name({"feature_extractor_type": "DeiTFeatureExtractor"}) == "DeiTImageProcessor"
    assert m.processor_class_name({"image_processor_type": "ViTImageProcessorFast"}) == "ViTImageProcessor"
    assert m.processor_class_name({"image_processor_type": "A", "feature_extractor_type": "BFeatureExtractor"}) == "A"
    assert m.processor_class_name({}) == ""
    r = m.image_preprocess({"feature_extractor_type": "DeiTFeatureExtractor", "size": 256, "do_center_crop": True,
                            "crop_size": 224})
    assert _fields(r) == ("StretchCenterCrop", 256, 256, 0, 224, 224)


def test_c25_height_width_with_crop():
    r = m.image_preprocess({"size": {"height": 240, "width": 260}, "do_center_crop": True,
                            "crop_size": {"height": 224, "width": 200}})
    assert _fields(r) == ("StretchCenterCrop", 260, 240, 0, 200, 224)


def test_c25_height_width_without_crop_is_stretch():
    r = m.image_preprocess({"size": {"height": 224, "width": 224}, "do_center_crop": False})
    assert _fields(r) == ("Stretch", 0, 0, 0, 224, 224)
    r = m.image_preprocess({"size": {"height": 30, "width": 20}, "do_center_crop": True})  # no crop_size
    assert _fields(r) == ("Stretch", 0, 0, 0, 20, 30)


def test_c25_shortest_edge_class_with_crop():
    r = m.image_preprocess({"image_processor_type": "CLIPImageProcessor", "size": 256, "do_center_crop": True,
                            "crop_size": 224})
    assert _fields(r) == ("ShortestEdgeCenterCrop", 0, 0, 256, 224, 224)


def test_c25_empty_class_integer_keeps_shortest_edge_rule():
    r = m.image_preprocess({"size": 256, "do_center_crop": True, "crop_size": 224})
    assert _fields(r) == ("ShortestEdgeCenterCrop", 0, 0, 256, 224, 224)


def test_c25_square_integer_without_crop():
    r = m.image_preprocess({"image_processor_type": "DeiTImageProcessor", "size": 32, "do_center_crop": False})
    assert _fields(r) == ("Stretch", 0, 0, 0, 32, 32)


def test_c25_crop_pct_beats_class_rule():
    r = m.image_preprocess({"image_processor_type": "DeiTImageProcessor", "size": 224, "crop_pct": 0.875})
    assert r["resize_mode"] == "ShortestEdgeCenterCrop" and r["resize_shortest_edge"] == 256


def test_c25_other_modes_have_zero_resize_size():
    for pre in ({"size": 224, "crop_pct": 0.875}, {"size": 100}, None,
                {"size": {"shortest_edge": 8}, "do_center_crop": True, "crop_size": 4}):
        r = m.image_preprocess(pre)
        assert r["resize_width"] == 0 and r["resize_height"] == 0


def test_shortest_edge_int_classes_match_installed_transformers():
    assert len(m.SHORTEST_EDGE_INT_CLASSES) == 44
    assert (m.scan_shortest_edge_classes() | m.scan_shortest_edge_classes_empirical()) == set(m.SHORTEST_EDGE_INT_CLASSES)


EXPECTED_C25 = {
    "visible": {"deit_style": ("StretchCenterCrop", 40, 40, 0, 32, 32),
                "deit_hw_crop": ("StretchCenterCrop", 48, 40, 0, 36, 32),
                "deit_legacy_fe": ("StretchCenterCrop", 40, 40, 0, 32, 32),
                "clip_style_crop": ("ShortestEdgeCenterCrop", 0, 0, 40, 32, 32),
                "square_int_nocrop": ("Stretch", 0, 0, 0, 32, 32)},
    "hidden": {"deit_style": ("StretchCenterCrop", 52, 52, 0, 44, 44),
               "deit_hw_crop": ("StretchCenterCrop", 60, 50, 0, 48, 40),
               "deit_legacy_fe": ("StretchCenterCrop", 52, 52, 0, 44, 44),
               "clip_style_crop": ("ShortestEdgeCenterCrop", 0, 0, 52, 44, 44),
               "square_int_nocrop": ("Stretch", 0, 0, 0, 44, 44)},
}


def _check_c25_cases(doc, variant):
    by_id = {c["id"]: c for c in doc["cases"]}
    root = doc.get("import_root", doc["models_root"])
    for case_id, want in EXPECTED_C25[variant].items():
        c = by_id[case_id]
        assert c["expect"] == "valid" and c["error_codes"] == [] and c["warning_codes"] == [], case_id
        assert c["task"] == "ImageClassification" and len(c["labels"]) == 3
        assert _fields(c["preprocess"]) == want, case_id
        shape = c["onnx"]["inputs"][0]["shape"]
        assert c["onnx"]["inputs"][0]["name"] == "pixel_values"
        assert shape == [-1, 3, want[5], want[4]], case_id
        assert c["license"]["spdx_id"] == "Apache-2.0"
        assert os.path.isfile(os.path.join(root, c["folder"], "model.onnx"))


def test_c25_cases_visible(doc):
    _check_c25_cases(doc, "visible")


def test_c25_cases_hidden_variant(tmp_path):
    import subprocess
    out = tmp_path / "models"
    gold = tmp_path / "import_expectations.json"
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_import_test_data.py"), "--variant", "hidden",
                        "--out-models", str(out), "--out-goldens", str(gold)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if "NEEDS:" in (r.stderr + r.stdout):
        pytest.skip("hidden tokenizers missing")
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    with open(gold, "r", encoding="utf-8") as f:
        _check_c25_cases(json.load(f), "hidden")


# ---- WP4.4a / C32 resample, LeViT rule, crop without size -------------------------------------------------------------

def test_resample_defaults_match_installed_transformers():
    found, failed = m.scan_resample_defaults()
    print("modules that failed to import: %s" % (failed or "none"))
    assert len(found) > 50  # the scan really ran
    want = {k: v for k, v in found.items() if v != 2}
    assert m.RESAMPLE_DEFAULTS == want
    assert len(m.RESAMPLE_DEFAULTS) == 37


def _resample(pre):
    return m.image_preprocess(pre)["resample"]


def test_resample_explicit_values():
    for v, name in enumerate(["Nearest", "Lanczos", "Bilinear", "Bicubic", "Box", "Hamming"]):
        assert _resample({"resample": v}) == name
    assert _resample({"resample": 3.0}) == "Bicubic"
    assert m.image_warnings({"resample": 3.0}) == []


def test_resample_invalid_values_are_bilinear_with_warning():
    for bad in ("bicubic", "3", 3.5, 7, -1, None, True, [3]):
        pre = {"image_processor_type": "DeiTImageProcessor", "resample": bad}  # DeiT default would be Bicubic
        assert _resample(pre) == "Bilinear", bad
        assert m.image_warnings(pre) == ["UnsupportedResize"], bad


def test_resample_class_defaults():
    assert _resample({"image_processor_type": "DeiTImageProcessor"}) == "Bicubic"
    assert _resample({"image_processor_type": "DeiTImageProcessorFast"}) == "Bicubic"
    assert _resample({"feature_extractor_type": "DeiTFeatureExtractor"}) == "Bicubic"
    assert _resample({"image_processor_type": "ViTImageProcessor"}) == "Bilinear"
    assert _resample({"image_processor_type": "EfficientNetImageProcessor"}) == "Nearest"
    assert _resample({"image_processor_type": "SmolVLMImageProcessor"}) == "Lanczos"
    assert _resample({}) == "Bilinear"
    assert _resample(None) == "Bilinear"


def test_new_preprocess_keys_present_everywhere():
    for pre in (None, {}, {"size": 224, "crop_pct": 0.875}):
        r = m.image_preprocess(pre)
        assert r["nms_iou_threshold"] == 0.7 and r["max_detections"] == 100


def test_levit_rule():
    pre = {"image_processor_type": "LevitImageProcessor", "size": {"shortest_edge": 224}, "do_center_crop": True,
           "crop_size": 224}
    assert _fields(m.image_preprocess(pre)) == ("ShortestEdgeCenterCrop", 0, 0, 256, 224, 224)
    pre = {"image_processor_type": "LevitImageProcessor", "size": 40, "do_center_crop": True, "crop_size": 32}
    assert _fields(m.image_preprocess(pre)) == ("ShortestEdgeCenterCrop", 0, 0, 45, 32, 32)
    pre = {"image_processor_type": "LevitImageProcessor", "size": {"shortest_edge": 224}}  # no crop: unchanged
    assert _fields(m.image_preprocess(pre)) == ("Stretch", 0, 0, 0, 224, 224)
    pre = {"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 224}, "do_center_crop": True,
           "crop_size": 224}
    assert m.image_preprocess(pre)["resize_shortest_edge"] == 224


def test_crop_without_size_warning():
    pre = {"size": {"height": 32, "width": 32}, "do_center_crop": True}
    assert m.image_warnings(pre) == ["UnsupportedResize"]
    assert m.image_warnings({"size": {"height": 32, "width": 32}, "do_center_crop": False}) == []
    assert m.image_warnings({"size": {"height": 32, "width": 32}, "do_center_crop": True,
                             "crop_size": 32}) == []
    assert m.image_warnings({"size": {"shortest_edge": 32}, "do_center_crop": True}) == []  # rule 4 warns elsewhere
    assert m.image_warnings(None) == []


# case -> (fields, resample, warning codes)
EXPECTED_C32 = {
    "visible": {"levit_style": (("ShortestEdgeCenterCrop", 0, 0, 32, 28, 28), "Bicubic", []),
                "crop_no_size": (("Stretch", 0, 0, 0, 32, 32), "Bilinear", ["UnsupportedResize"]),
                "resample_box": (("Stretch", 0, 0, 0, 32, 32), "Box", []),
                "resample_bad": (("Stretch", 0, 0, 0, 32, 32), "Bilinear", ["UnsupportedResize"]),
                "effnet_default": (("Stretch", 0, 0, 0, 32, 32), "Nearest", [])},
    "hidden": {"levit_style": (("ShortestEdgeCenterCrop", 0, 0, 48, 44, 44), "Bicubic", []),
               "crop_no_size": (("Stretch", 0, 0, 0, 48, 48), "Bilinear", ["UnsupportedResize"]),
               "resample_box": (("Stretch", 0, 0, 0, 48, 48), "Box", []),
               "resample_bad": (("Stretch", 0, 0, 0, 48, 48), "Bilinear", ["UnsupportedResize"]),
               "effnet_default": (("Stretch", 0, 0, 0, 48, 48), "Nearest", [])},
}


def _check_c32_cases(doc, variant):
    by_id = {c["id"]: c for c in doc["cases"]}
    root = doc.get("import_root", doc["models_root"])
    for case_id, (fields, resample, warns) in EXPECTED_C32[variant].items():
        c = by_id[case_id]
        assert c["expect"] == "valid" and c["error_codes"] == [], case_id
        assert c["warning_codes"] == warns, case_id
        assert c["task"] == "ImageClassification" and len(c["labels"]) == 3
        assert _fields(c["preprocess"]) == fields, case_id
        assert c["preprocess"]["resample"] == resample, case_id
        assert c["onnx"]["inputs"][0]["shape"] == [-1, 3, fields[5], fields[4]], case_id
        assert c["license"]["spdx_id"] == "Apache-2.0"
        assert os.path.isfile(os.path.join(root, c["folder"], "model.onnx"))


def _check_new_keys(doc):
    for c in doc["cases"]:
        if "preprocess" in c:
            p = c["preprocess"]
            assert p["nms_iou_threshold"] == 0.7 and p["max_detections"] == 100, c["id"]
            assert p["resample"] in m.RESAMPLE_NAMES.values(), c["id"]
    by_id = {c["id"]: c for c in doc["cases"]}
    if "resnet-50" in by_id:
        assert by_id["resnet-50"]["preprocess"]["resample"] == "Bicubic"
    if "rtdetr_r18vd" in by_id:
        assert by_id["rtdetr_r18vd"]["preprocess"]["resample"] == "Bilinear"


def test_c32_cases_visible(doc):
    _check_c32_cases(doc, "visible")
    _check_new_keys(doc)


def test_c32_cases_hidden_variant(tmp_path):
    import subprocess
    out = tmp_path / "models"
    gold = tmp_path / "import_expectations.json"
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_import_test_data.py"), "--variant", "hidden",
                        "--out-models", str(out), "--out-goldens", str(gold)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if "NEEDS:" in (r.stderr + r.stdout):
        pytest.skip("hidden tokenizers missing")
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    with open(gold, "r", encoding="utf-8") as f:
        d = json.load(f)
    _check_c32_cases(d, "hidden")
    _check_new_keys(d)


# ---- WP9S.6t / C78 / C78b speech import cases and the Phonon sidecar ---------------------------------------------------

import write_phonon2_sidecar as sc  # noqa: E402

CONTRACT_SIDECAR = {"task": "SpeechToText", "license": {
    "spdx_id": "CC-BY-4.0", "source_url": "https://huggingface.co/FermionResearch/Phonon-2",
    "attribution": "Phonon-2 by Fermion Research (CC-BY-4.0), derived from NVIDIA parakeet-tdt-0.6b-v3 (CC-BY-4.0); "
                   "converted to ONNX."}}


def test_sidecar_bytes_exact():
    raw = sc.sidecar_bytes().decode("utf-8")
    assert json.loads(raw) == CONTRACT_SIDECAR
    assert list(json.loads(raw)) == ["task", "license"]
    assert list(json.loads(raw)["license"]) == ["spdx_id", "source_url", "attribution"]
    assert raw.startswith('{\n "task": "SpeechToText",\n "license": {\n  "spdx_id"') and raw.endswith("}\n")


def test_write_sidecar_overwrites(tmp_path):
    (tmp_path / "modelpipelines.json").write_text("old", encoding="utf-8")
    path = sc.write_sidecar(str(tmp_path))
    assert json.load(open(path, encoding="utf-8")) == CONTRACT_SIDECAR


@pytest.mark.parametrize("variant", ["fp32", "fp16w", "int8", "fp16"])
def test_phonon_folder_sidecars(variant):
    folder = os.path.join(_paths.dev_dir(), "models", "stt", "phonon-2-onnx", variant)
    path = os.path.join(folder, "modelpipelines.json")
    if not os.path.isdir(folder):
        pytest.skip("Phonon folder %s not built" % variant)
    assert os.path.isfile(path), "run write_phonon2_sidecar.py"
    with open(path, "rb") as f:
        assert f.read() == sc.sidecar_bytes()
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == CONTRACT_SIDECAR


def test_speech_settings_defaults_and_mapping():
    d = m.speech_settings({"architectures": ["ParakeetForTDT"]}, None)
    assert d == {"sample_rate": 16000, "num_mel_bins": 128, "fft_size": 512, "window_length": 400, "hop_length": 160,
                 "preemphasis": 0.97, "blank_token_id": 8192, "durations": [0, 1, 2, 3, 4],
                 "max_symbols_per_step": 10, "subsampling_factor": 8, "silence_threshold_db": -60.0}
    cfg = {"blank_token_id": 5, "durations": [0, 2], "max_symbols_per_step": 3,
           "encoder_config": {"subsampling_factor": 4}}
    pc = {"feature_extractor": {"sampling_rate": 8000, "feature_size": 80, "n_fft": 256, "win_length": 200,
                                "hop_length": 80, "preemphasis": 1}}
    d = m.speech_settings(cfg, pc)
    assert (d["sample_rate"], d["num_mel_bins"], d["fft_size"], d["window_length"], d["hop_length"]) == (
        8000, 80, 256, 200, 80)
    assert (d["preemphasis"], d["blank_token_id"], d["durations"], d["max_symbols_per_step"],
            d["subsampling_factor"]) == (1.0, 5, [0, 2], 3, 4)
    assert m.SPEECH_DEFAULTS["durations"] == [0, 1, 2, 3, 4]  # defaults are not mutated


@pytest.mark.parametrize("cfg,pc,name", [
    ({}, {"feature_extractor": {"feature_size": "8"}}, "processor_config.json: feature_size"),
    ({}, {"feature_extractor": {"preemphasis": True}}, "processor_config.json: preemphasis"),
    ({"blank_token_id": 5.0}, {}, "config.json: blank_token_id"),
    ({"durations": [0, "1"]}, {}, "config.json: durations"),
    ({"encoder_config": {"subsampling_factor": "8"}}, {}, "config.json: subsampling_factor")])
def test_speech_settings_wrong_type(cfg, pc, name):
    with pytest.raises(m.SpeechJsonError, match=name):
        m.speech_settings(cfg, pc)


def test_detect_task_speech():
    assert m.detect_task("None", None, False, {"architectures": ["ParakeetForTDT"]}, True) == (
        "SpeechToText", "architectures")
    assert m.detect_task("None", {"task": "SpeechToText"}, False, {}, True) == ("SpeechToText", "modelpipelines.json")
    assert m.detect_task("None", None, False, {"architectures": ["ParakeetForCTC"]}, True) == (None, None)
    assert m.detect_task("None", None, False, {"architectures": ["parakeetfortdt"]}, True) == (None, None)


def test_speech_files(tmp_path):
    with pytest.raises(ValueError, match="encoder.onnx"):
        m.speech_files(str(tmp_path))
    (tmp_path / "model.onnx").write_bytes(b"x")  # never used for this task
    (tmp_path / "encoder.onnx").write_bytes(b"x")
    with pytest.raises(ValueError, match="decoder_joint.onnx"):
        m.speech_files(str(tmp_path))
    (tmp_path / "decoder_joint.onnx").write_bytes(b"")  # empty counts as missing
    with pytest.raises(ValueError, match="decoder_joint.onnx"):
        m.speech_files(str(tmp_path))
    (tmp_path / "decoder_joint.onnx").write_bytes(b"x")
    assert m.speech_files(str(tmp_path)) == ("encoder.onnx", "decoder_joint.onnx")


def test_stt_cases(doc):
    by_id = {c["id"]: c for c in doc["cases"]}
    root = doc.get("import_root", doc["models_root"])
    tiny = {"sample_rate": 16000, "num_mel_bins": 8, "fft_size": 512, "window_length": 400, "hop_length": 160,
            "preemphasis": 0.97, "blank_token_id": 5, "durations": [0, 1, 2], "max_symbols_per_step": 10,
            "subsampling_factor": 8, "silence_threshold_db": -60.0}
    for case_id, source, spdx in (("stt_tiny_arch", "architectures", "MIT"),
                                  ("stt_with_model_onnx", "architectures", "MIT"),
                                  ("stt_tiny_sidecar_task", "modelpipelines.json", "CC-BY-4.0")):
        c = by_id[case_id]
        assert c["expect"] == "valid" and c["error_codes"] == [] and c["task"] == "SpeechToText", case_id
        assert c["task_source"] == source and c["license"]["spdx_id"] == spdx
        assert (c["model_file"], c["decoder_model_file"]) == ("encoder.onnx", "decoder_joint.onnx")
        assert c["speech"] == tiny, case_id
        assert os.path.isfile(os.path.join(root, c["folder"], "tokenizer.json"))
    assert os.path.isfile(os.path.join(root, by_id["stt_with_model_onnx"]["folder"], "model.onnx"))
    assert not os.path.isfile(os.path.join(root, by_id["stt_tiny_arch"]["folder"], "model.onnx"))
    with open(os.path.join(root, "import_test", "stt_tiny_sidecar_task", "config.json"), encoding="utf-8") as f:
        assert "architectures" not in json.load(f)
    c = by_id["stt_no_decoder"]
    assert (c["expect"], c["error_codes"], c["error_message_contains"]) == (
        "parse_error", ["MissingModelFile"], ["decoder_joint.onnx"])
    assert not os.path.exists(os.path.join(root, c["folder"], "decoder_joint.onnx"))
    c = by_id["stt_bad_speech_json"]
    assert (c["expect"], c["error_codes"], c["error_message_contains"]) == (
        "parse_error", ["InvalidJson"], ["processor_config.json: feature_size"])
    c = by_id["stt_defaults"]
    assert c["parse_only"] is True and c["task"] == "SpeechToText"
    assert c["speech"] == dict(m.SPEECH_DEFAULTS)
    assert c["onnx"] == m.onnx_block(os.path.join(root, c["folder"], "encoder.onnx"))
    for c in doc["cases"]:  # non-speech records carry "" and the defaults
        if c["expect"] != "parse_error" and c["task"] != "SpeechToText":
            assert c["decoder_model_file"] == "" and c["speech"] == dict(m.SPEECH_DEFAULTS), c["id"]
