"""Generate importer test folders and the importer expectations JSON (WP2.1).

Builds tiny ONNX models with the onnx helper API (no torch), writes one HF-style folder per case under
<out-models>/import_test/<case>, and writes <out-goldens> describing what the importer must produce.
Reference cases (folders from fetch_reference_models.py) are added when those folders exist.

Expectations are derived from the written files with the importer rules below (docs/contracts.md C14), never from
Unreal output. Reference cases are cross-checked against transformers.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

PRODUCER = "ModelPipelines-Tools"
OPSET = 17
IR_VERSION = 8
F32_255 = float(np.float32(1.0 / 255.0))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TASKS = ["TextEmbedding", "TextClassification", "ImageClassification", "ObjectDetection"]
SIGMOID_DETECTION = {"rt_detr", "rt_detr_v2", "d_fine", "deformable_detr", "conditional_detr", "dab_detr"}
SOFTMAX_DETECTION = {"detr", "yolos", "table-transformer"}
REFERENCE_SLUGS = ["all-MiniLM-L6-v2", "distilbert-sst2", "resnet-50", "rtdetr_r18vd"]


# ----------------------------------------------------------------------------------------------------------------
# Importer rules (pure helpers, unit-tested)
# ----------------------------------------------------------------------------------------------------------------

def normalize_license(value):
    """README front-matter license id -> SPDX id. Known ids are mapped; unknown values are kept as written
    (quotes stripped); '' only when missing."""
    if not value:
        return ""
    v = str(value).strip().strip("'\"").strip()
    return {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-2-clause": "BSD-2-Clause",
            "bsd-3-clause": "BSD-3-Clause"}.get(v.lower(), v)


def resolve_license(sidecar, readme_text):
    """Per-field license: spdx_id from the sidecar if non-empty after trimming, else README front matter;
    source_url and attribution only from the sidecar."""
    lic = {"spdx_id": "", "source_url": "", "attribution": ""}
    sc = sidecar.get("license") if isinstance(sidecar, dict) else None
    if isinstance(sc, dict):
        for k in lic:
            v = sc.get(k)
            if isinstance(v, str):
                lic[k] = v
    if not lic["spdx_id"].strip():
        lic["spdx_id"] = ""
        if readme_text is not None:
            lic["spdx_id"] = normalize_license(readme_license(readme_text))
    return lic


DEVICES = {"auto": "Auto", "cpu": "CPU", "gpu": "GPU"}


def resolve_device(sidecar):
    v = sidecar.get("preferred_device") if isinstance(sidecar, dict) else None
    if not isinstance(v, str):
        return "Auto"
    if v.strip().lower() not in DEVICES:
        raise ValueError("unknown preferred_device: " + v)
    return DEVICES[v.strip().lower()]


def readme_license(text):
    """Value of `license:` in the YAML front matter of README text, or None."""
    if not text.startswith("---"):
        return None
    lines = text.split("\n")[1:]
    for line in lines:
        s = line.rstrip("\r")
        if s.strip() == "---":
            return None
        if s.startswith("license:"):
            return s.split(":", 1)[1].strip()
    return None


def parse_labels(id2label):
    """Return (labels, error_code). labels[i] = id2label[str(i)]; keys must be exactly 0..n-1."""
    if not id2label:
        return [], None
    try:
        keys = sorted(int(k) for k in id2label.keys())
    except (ValueError, TypeError):
        return None, "InvalidLabels"
    if keys != list(range(len(keys))):
        return None, "InvalidLabels"
    by_key = {int(k): v for k, v in id2label.items()}
    return [by_key[i] for i in range(len(keys))], None


def text_max_tokens(sbert, tokenizer_config, config):
    """sentence_bert max_seq_length -> tokenizer model_max_length (0<v<=65536) -> max_position_embeddings -> 512;
    then min with max_position_embeddings when present."""
    max_pos = (config or {}).get("max_position_embeddings")
    value = None
    if sbert and isinstance(sbert.get("max_seq_length"), (int, float)) and sbert["max_seq_length"] > 0:
        value = int(sbert["max_seq_length"])
    if value is None and tokenizer_config:
        v = tokenizer_config.get("model_max_length")
        if isinstance(v, (int, float)) and 0 < v <= 65536:
            value = int(v)
    if value is None and isinstance(max_pos, (int, float)):
        value = int(max_pos)
    if value is None:
        value = 512
    if isinstance(max_pos, (int, float)):
        value = min(value, int(max_pos))
    return value


def image_preprocess(pre):
    """Image fields from a preprocessor_config dict (None = file missing)."""
    out = {"image_width": 224, "image_height": 224, "resize_mode": "Stretch", "resize_shortest_edge": 0,
           "rescale_factor": F32_255, "image_mean": list(IMAGENET_MEAN), "image_std": list(IMAGENET_STD)}
    if pre is None:
        return out
    size = pre.get("size")
    shortest = None
    if isinstance(size, dict) and "height" in size and "width" in size:
        out["image_width"], out["image_height"] = int(size["width"]), int(size["height"])
    else:
        if isinstance(size, dict) and "shortest_edge" in size:
            shortest = int(size["shortest_edge"])
        elif isinstance(size, int) and not isinstance(size, bool):
            shortest = size
        crop = pre.get("crop_size")
        if shortest is not None and "crop_pct" in pre:
            if shortest < 384:
                out["resize_mode"] = "ShortestEdgeCenterCrop"
                out["resize_shortest_edge"] = int(shortest / pre["crop_pct"])
            out["image_width"] = out["image_height"] = shortest
        elif pre.get("do_center_crop") and crop is not None and shortest is not None:
            out["resize_mode"] = "ShortestEdgeCenterCrop"
            out["resize_shortest_edge"] = shortest
            if isinstance(crop, dict):
                out["image_width"], out["image_height"] = int(crop["width"]), int(crop["height"])
            else:
                out["image_width"] = out["image_height"] = int(crop)
        elif shortest is not None:
            out["image_width"] = out["image_height"] = shortest
    if pre.get("do_rescale", True):
        out["rescale_factor"] = float(np.float32(pre.get("rescale_factor", 1.0 / 255.0)))
    else:
        out["rescale_factor"] = 1.0
    if pre.get("do_normalize", True):
        out["image_mean"] = list(pre.get("image_mean", IMAGENET_MEAN))
        out["image_std"] = list(pre.get("image_std", IMAGENET_STD))
    else:
        out["image_mean"] = [0.0, 0.0, 0.0]
        out["image_std"] = [1.0, 1.0, 1.0]
    return out


def score_activation(task, config):
    config = config or {}
    if task == "TextClassification" or task == "ImageClassification":
        return "Sigmoid" if config.get("problem_type") == "multi_label_classification" else "Softmax"
    if task == "ObjectDetection":
        mt = config.get("model_type")
        if mt in SOFTMAX_DETECTION:
            return "Softmax"
        return "Sigmoid"
    return "Softmax"


def detect_task(task_override, sidecar, has_modules, config, has_tokenizer):
    """Return (task, task_source) or (None, None) when undetected."""
    if task_override and task_override != "None":
        return task_override, "options"
    if sidecar and isinstance(sidecar.get("task"), str):
        for t in TASKS:
            if t.lower() == sidecar["task"].lower():
                return t, "modelpipelines.json"
    if has_modules:
        return "TextEmbedding", "modules.json"
    archs = (config or {}).get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        a = archs[0]
        if a.endswith("ForSequenceClassification"):
            return "TextClassification", "architectures"
        if a.endswith("ForImageClassification"):
            return "ImageClassification", "architectures"
        if a.endswith("ForObjectDetection"):
            return "ObjectDetection", "architectures"
        if a.endswith("Model") and has_tokenizer:
            return "TextEmbedding", "architectures"
    return None, None


def pooling_and_normalize(modules, read_json):
    """(pooling, normalize_embeddings) from a parsed modules.json; read_json(relpath) -> dict or None."""
    pooling, normalize = "Mean", False
    for m in modules or []:
        t = m.get("type", "")
        if t == "sentence_transformers.models.Normalize":
            normalize = True
        elif t == "sentence_transformers.models.Pooling":
            cfg = read_json((m.get("path", "") + "/config.json").lstrip("/"))
            if cfg:
                if cfg.get("pooling_mode_cls_token"):
                    pooling = "CLS"
                elif cfg.get("pooling_mode_mean_tokens"):
                    pooling = "Mean"
    return pooling, normalize


# ----------------------------------------------------------------------------------------------------------------
# ONNX model builders
# ----------------------------------------------------------------------------------------------------------------

def _onnx():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    return onnx, TensorProto, helper, numpy_helper


def _finish(graph, path):
    onnx, _, helper, _ = _onnx()
    model = helper.make_model(graph, producer_name=PRODUCER, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)
    return model


def _vi(name, elem, shape):
    _, _, helper, _ = _onnx()
    return helper.make_tensor_value_info(name, elem, shape)


def build_text_embedding(path, vocab, hidden, seed=11, pos_rows=None, pos_seed=21):
    """pos_rows=None: the plain tiny model. Else adds a position-embedding Gather (Range(0, seq_len) into a
    pos_rows-row table) that is added to the token embedding."""
    onnx, T, h, nh = _onnx()
    rng = np.random.RandomState(seed)
    table = rng.uniform(-1, 1, size=(vocab, hidden)).astype(np.float32)
    inits = [nh.from_array(table, "table"), nh.from_array(np.array([2], dtype=np.int64), "ax2"),
             nh.from_array(np.array(0.0, dtype=np.float32), "zero")]
    nodes = [h.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0)]
    if pos_rows is not None:
        pos = np.random.RandomState(pos_seed).uniform(-1, 1, size=(pos_rows, hidden)).astype(np.float32)
        inits += [nh.from_array(pos, "pos_table"), nh.from_array(np.array(1, dtype=np.int64), "one_idx"),
                  nh.from_array(np.array(0, dtype=np.int64), "range_start"),
                  nh.from_array(np.array(1, dtype=np.int64), "range_delta")]
        nodes += [
            h.make_node("Shape", ["input_ids"], ["ids_shape"]),
            h.make_node("Gather", ["ids_shape", "one_idx"], ["seq_len"], axis=0),
            h.make_node("Range", ["range_start", "seq_len", "range_delta"], ["pos_ids"]),
            h.make_node("Gather", ["pos_table", "pos_ids"], ["pos_emb"], axis=0),
            h.make_node("Add", ["emb", "pos_emb"], ["emb_pos"]),
        ]
        emb = "emb_pos"
    else:
        emb = "emb"
    nodes += [
        h.make_node("Cast", ["attention_mask"], ["mask_f"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["mask_f", "ax2"], ["mask3"]),
        h.make_node("Mul", [emb, "mask3"], ["masked"]),
        h.make_node("Cast", ["token_type_ids"], ["tt_f"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["tt_f", "ax2"], ["tt3"]),
        h.make_node("Mul", ["tt3", "zero"], ["tt0"]),
        h.make_node("Add", ["masked", "tt0"], ["last_hidden_state"]),
    ]
    g = h.make_graph(nodes, "tiny_text_embedding" if pos_rows is None else "tiny_text_embedding_positions",
                     [_vi(n, T.INT64, ["batch", "sequence"]) for n in ("input_ids", "attention_mask",
                                                                        "token_type_ids")],
                     [_vi("last_hidden_state", T.FLOAT, ["batch", "sequence", hidden])], initializer=inits)
    return _finish(g, path)


def build_text_classification(path, vocab, labels, seed=12):
    onnx, T, h, nh = _onnx()
    rng = np.random.RandomState(seed)
    table = rng.uniform(-1, 1, size=(vocab, labels)).astype(np.float32)
    inits = [nh.from_array(table, "table"), nh.from_array(np.array([2], dtype=np.int64), "ax2"),
             nh.from_array(np.array([1], dtype=np.int64), "ax1")]
    nodes = [
        h.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0),
        h.make_node("Cast", ["attention_mask"], ["mask_f"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["mask_f", "ax2"], ["mask3"]),
        h.make_node("Mul", ["emb", "mask3"], ["masked"]),
        h.make_node("ReduceSum", ["masked", "ax1"], ["logits"], keepdims=0),
    ]
    g = h.make_graph(nodes, "tiny_text_classification",
                     [_vi(n, T.INT64, ["batch", "sequence"]) for n in ("input_ids", "attention_mask")],
                     [_vi("logits", T.FLOAT, ["batch", labels])], initializer=inits)
    return _finish(g, path)


def build_image_classification(path, height, width, labels, seed=13):
    onnx, T, h, nh = _onnx()
    rng = np.random.RandomState(seed)
    w = rng.uniform(-1, 1, size=(3, labels)).astype(np.float32)
    nodes = [
        h.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        h.make_node("MatMul", ["pooled", "w"], ["logits"]),
    ]
    g = h.make_graph(nodes, "tiny_image_classification", [_vi("pixel_values", T.FLOAT, ["batch", 3, height, width])],
                     [_vi("logits", T.FLOAT, ["batch", labels])], initializer=[nh.from_array(w, "w")])
    return _finish(g, path)


def build_object_detection(path, height, width, queries, classes, seed=14):
    onnx, T, h, nh = _onnx()
    rng = np.random.RandomState(seed)
    wl = rng.uniform(-1, 1, size=(3, queries * classes)).astype(np.float32)
    wb = rng.uniform(-1, 1, size=(3, queries * 4)).astype(np.float32)
    inits = [nh.from_array(wl, "wl"), nh.from_array(wb, "wb"),
             nh.from_array(np.array([-1, queries, classes], dtype=np.int64), "shape_l"),
             nh.from_array(np.array([-1, queries, 4], dtype=np.int64), "shape_b")]
    nodes = [
        h.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        h.make_node("MatMul", ["pooled", "wl"], ["flat_l"]),
        h.make_node("Reshape", ["flat_l", "shape_l"], ["logits"]),
        h.make_node("MatMul", ["pooled", "wb"], ["flat_b"]),
        h.make_node("Sigmoid", ["flat_b"], ["sig_b"]),
        h.make_node("Reshape", ["sig_b", "shape_b"], ["pred_boxes"]),
    ]
    g = h.make_graph(nodes, "tiny_object_detection", [_vi("pixel_values", T.FLOAT, ["batch", 3, height, width])],
                     [_vi("logits", T.FLOAT, ["batch", queries, classes]),
                      _vi("pred_boxes", T.FLOAT, ["batch", queries, 4])], initializer=inits)
    return _finish(g, path)


def build_wrong_io(path):
    onnx, T, h, nh = _onnx()
    inits = [nh.from_array(np.array(2.0, dtype=np.float32), "two"), nh.from_array(np.array(1.0, dtype=np.float32), "one")]
    nodes = [h.make_node("Mul", ["x", "two"], ["x2"]), h.make_node("Add", ["x2", "one"], ["y"])]
    g = h.make_graph(nodes, "wrong_io", [_vi("x", T.FLOAT, ["batch", 4])], [_vi("y", T.FLOAT, ["batch", 4])],
                     initializer=inits)
    return _finish(g, path)


def build_unknown_op(path, labels):
    onnx, T, h, nh = _onnx()
    nodes = [h.make_node("ModelPipelinesNotARealOp", ["input_ids", "attention_mask"], ["logits"])]
    g = h.make_graph(nodes, "unknown_op", [_vi(n, T.INT64, ["batch", "sequence"]) for n in ("input_ids",
                                                                                               "attention_mask")],
                     [_vi("logits", T.FLOAT, ["batch", labels])])
    return _finish(g, path)


# ----------------------------------------------------------------------------------------------------------------
# Case definitions
# ----------------------------------------------------------------------------------------------------------------

def params(variant):
    if variant == "visible":
        return dict(
            emb_hidden=4, emb_tok="bert-base-uncased", cls_tok="bert-base-uncased",
            cls_labels=["negative", "neutral", "positive"], cls_max_pos=128,
            emb_sbert=64, emb_max_pos=128, emb_pool_cls=True, emb_normalize=True, sidecar_tokcfg=256,
            img_h=32, img_w=32, img_labels=["cat", "café", "日本", "dog"], crop_pct=0.875, img_size=32,
            img_mean=[0.5, 0.4, 0.3], img_std=[0.2, 0.25, 0.3],
            det_h=32, det_w=32, det_q=5, det_labels=["person", "crate"],
            two_labels=["NEGATIVE", "POSITIVE"], mismatch_labels=["yes", "no"],
            small_vocab_rows=30000, pos_rows=32, pos_sbert=64)
    return dict(
        emb_hidden=6, emb_tok="t5-small", cls_tok="gpt2",
        cls_labels=["sunny", "rain", "ñandú", "snow", "fog"], cls_max_pos=96,
        emb_sbert=32, emb_max_pos=100, emb_pool_cls=False, emb_normalize=False, sidecar_tokcfg=200,
        img_h=48, img_w=48, img_labels=["sky", "sea", "лес", "sand"], crop_pct=0.9, img_size=48,
        img_mean=[0.45, 0.5, 0.55], img_std=[0.25, 0.2, 0.15],
        det_h=48, det_w=40, det_q=7, det_labels=["drone", "kite", "balloon"],
        two_labels=["BAD", "GOOD"], mismatch_labels=["lo", "mid", "hi", "top"],
        small_vocab_rows=-1000, pos_rows=16, pos_sbert=48)


def jbytes(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def full_license(case_id):
    return {"license": {"spdx_id": "Apache-2.0", "source_url": "https://example.invalid/" + case_id,
                        "attribution": "ModelPipelines test data"}}


def id2label(labels):
    return {str(i): n for i, n in enumerate(labels)}


def tokenizer_vocab(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path).get_vocab_size(with_added_tokens=True)


class Builder:
    def __init__(self, out_models, dev, variant):
        self.dev = dev
        self.root = os.path.join(out_models, "import_test")
        self.p = params(variant)
        self.tok = {}
        self.cases = []
        self.models_seen = {}
        for name in {self.p["emb_tok"], self.p["cls_tok"]}:
            tp = os.path.join(dev, "models", "tokenizers", name, "tokenizer.json")
            if not os.path.isfile(tp):
                raise SystemExit("NEEDS: tokenizer file missing: " + tp)
            self.tok[name] = tp

    def folder(self, case_id):
        d = os.path.join(self.root, case_id)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        return d

    def write(self, d, rel, data):
        path = os.path.join(d, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(path, "wb") as f:
            f.write(data)

    def copy_tokenizer(self, d, name):
        shutil.copyfile(self.tok[name], os.path.join(d, "tokenizer.json"))

    def text_cls_model(self, d, rel="model.onnx", labels=None):
        p = self.p
        build_text_classification(os.path.join(d, rel), tokenizer_vocab(self.tok[p["cls_tok"]]),
                                  labels or len(p["cls_labels"]))
        self.models_seen[os.path.basename(d)] = ("text", p["cls_tok"])

    def img_cls_model(self, d, rel="model.onnx"):
        p = self.p
        build_image_classification(os.path.join(d, rel), p["img_h"], p["img_w"], len(p["img_labels"]))
        self.models_seen[os.path.basename(d)] = ("image", None)

    def emb_model(self, d, vocab=None, pos_rows=None):
        p = self.p
        build_text_embedding(os.path.join(d, "model.onnx"), vocab or tokenizer_vocab(self.tok[p["emb_tok"]]),
                             p["emb_hidden"], pos_rows=pos_rows)
        self.models_seen[os.path.basename(d)] = ("text_emb", p["emb_tok"])

    def emb_folder_files(self, d, case_id, sbert, max_pos):
        p = self.p
        cfg = {"architectures": ["BertModel"], "model_type": "bert"}
        if max_pos is not None:
            cfg["max_position_embeddings"] = max_pos
        self.write(d, "config.json", jbytes(cfg))
        self.copy_tokenizer(d, p["emb_tok"])
        self.write(d, "tokenizer_config.json", jbytes({"model_max_length": 512}))
        mods = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
        if p["emb_normalize"]:
            mods.append({"idx": 2, "name": "2", "path": "2_Normalize",
                         "type": "sentence_transformers.models.Normalize"})
        self.write(d, "modules.json", jbytes(mods))
        self.write(d, "1_Pooling/config.json", jbytes({
            "word_embedding_dimension": p["emb_hidden"], "pooling_mode_cls_token": p["emb_pool_cls"],
            "pooling_mode_mean_tokens": not p["emb_pool_cls"]}))
        self.write(d, "sentence_bert_config.json", jbytes({"max_seq_length": sbert}))
        self.write(d, "modelpipelines.json", jbytes(full_license(case_id)))

    def cls_config(self, labels, **extra):
        c = {"architectures": ["DistilBertForSequenceClassification"], "model_type": "distilbert"}
        c.update(extra)
        c["id2label"] = id2label(labels)
        return c

    def img_config(self, labels):
        return {"architectures": ["ResNetForImageClassification"], "model_type": "resnet", "id2label": id2label(labels)}

    def img_pre(self):
        p = self.p
        return {"image_processor_type": "ConvNextImageProcessor", "size": p["img_size"], "crop_pct": p["crop_pct"],
                "do_resize": True, "do_rescale": True, "rescale_factor": 0.00392156862745098, "do_normalize": True,
                "image_mean": p["img_mean"], "image_std": p["img_std"]}

    def add(self, case_id, expect, errors=(), warnings=(), task_override="None"):
        self.cases.append(dict(id=case_id, expect=expect, errors=list(errors), warnings=list(warnings),
                               task_override=task_override))

    def build_all(self):
        p = self.p
        # tiny_text_embedding
        d = self.folder("tiny_text_embedding")
        self.emb_model(d)
        self.write(d, "config.json", jbytes({"architectures": ["BertModel"], "model_type": "bert",
                                             "max_position_embeddings": p["emb_max_pos"]}))
        self.copy_tokenizer(d, p["emb_tok"])
        self.write(d, "tokenizer_config.json", jbytes({"model_max_length": 512}))
        mods = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
        if p["emb_normalize"]:
            mods.append({"idx": 2, "name": "2", "path": "2_Normalize",
                         "type": "sentence_transformers.models.Normalize"})
        self.write(d, "modules.json", jbytes(mods))
        self.write(d, "1_Pooling/config.json", jbytes({
            "word_embedding_dimension": p["emb_hidden"], "pooling_mode_cls_token": p["emb_pool_cls"],
            "pooling_mode_mean_tokens": not p["emb_pool_cls"]}))
        self.write(d, "sentence_bert_config.json", jbytes({"max_seq_length": p["emb_sbert"]}))
        self.write(d, "modelpipelines.json", jbytes(full_license("tiny_text_embedding")))
        self.add("tiny_text_embedding", "valid")

        # tiny_text_classification
        d = self.folder("tiny_text_classification")
        self.text_cls_model(d)
        self.write(d, "config.json", jbytes(self.cls_config(
            p["cls_labels"], max_position_embeddings=p["cls_max_pos"], problem_type="multi_label_classification")))
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "tokenizer_config.json", b'{"model_max_length": 1000000000000000019884624838656}\n')
        self.write(d, "modelpipelines.json", jbytes(full_license("tiny_text_classification")))
        self.add("tiny_text_classification", "valid")

        # tiny_image_classification
        d = self.folder("tiny_image_classification")
        self.img_cls_model(d)
        self.write(d, "config.json", jbytes(self.img_config(p["img_labels"])))
        self.write(d, "preprocessor_config.json", jbytes(self.img_pre()))
        sc = full_license("tiny_image_classification")
        sc["preferred_device"] = "CPU"
        self.write(d, "modelpipelines.json", jbytes(sc))
        self.add("tiny_image_classification", "valid")

        # tiny_object_detection
        d = self.folder("tiny_object_detection")
        build_object_detection(os.path.join(d, "model.onnx"), p["det_h"], p["det_w"], p["det_q"],
                               len(p["det_labels"]))
        self.write(d, "config.json", jbytes({"architectures": ["RTDetrForObjectDetection"], "model_type": "rt_detr",
                                             "id2label": id2label(p["det_labels"])}))
        self.write(d, "preprocessor_config.json", jbytes({
            "image_processor_type": "RTDetrImageProcessor", "size": {"height": p["det_h"], "width": p["det_w"]},
            "do_resize": True, "do_rescale": True, "rescale_factor": 0.00392156862745098, "do_normalize": False}))
        self.write(d, "modelpipelines.json", jbytes(full_license("tiny_object_detection")))
        self.models_seen["tiny_object_detection"] = ("det", None)
        self.add("tiny_object_detection", "valid")

        # sidecar_task
        d = self.folder("sidecar_task")
        self.emb_model(d)
        self.write(d, "config.json", jbytes({"architectures": ["SomethingElse"], "model_type": "custom"}))
        self.copy_tokenizer(d, p["emb_tok"])
        self.write(d, "tokenizer_config.json", jbytes({"model_max_length": p["sidecar_tokcfg"]}))
        sc = full_license("sidecar_task")
        sc["task"] = "textembedding"
        self.write(d, "modelpipelines.json", jbytes(sc))
        self.add("sidecar_task", "valid")

        # readme_license
        d = self.folder("readme_license")
        self.img_cls_model(d, "onnx/model.onnx")
        self.write(d, "config.json", jbytes(self.img_config(p["img_labels"])))
        self.write(d, "preprocessor_config.json", jbytes(self.img_pre()))
        self.write(d, "README.md", "---\nlicense: mit\ntags:\n- test\n---\n# Tiny\n")
        self.add("readme_license", "valid", warnings=["LicenseSourceMissing", "LicenseAttributionMissing"])

        # no_license
        d = self.folder("no_license")
        self.text_cls_model(d)
        self.write(d, "config.json", jbytes(self.cls_config(
            p["cls_labels"], max_position_embeddings=p["cls_max_pos"], problem_type="multi_label_classification")))
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "tokenizer_config.json", b'{"model_max_length": 1000000000000000019884624838656}\n')
        self.add("no_license", "valid",
                 warnings=["LicenseSpdxMissing", "LicenseSourceMissing", "LicenseAttributionMissing"])

        # wrong_io_for_task
        d = self.folder("wrong_io_for_task")
        build_wrong_io(os.path.join(d, "model.onnx"))
        self.write(d, "config.json", jbytes(self.cls_config(p["two_labels"])))
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "modelpipelines.json", jbytes(full_license("wrong_io_for_task")))
        self.models_seen["wrong_io_for_task"] = ("wrong", None)
        self.add("wrong_io_for_task", "validation_error", errors=["MissingInput", "UnexpectedInput"])

        # label_mismatch
        d = self.folder("label_mismatch")
        self.text_cls_model(d)
        self.write(d, "config.json", jbytes(self.cls_config(
            p["mismatch_labels"], max_position_embeddings=p["cls_max_pos"])))
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "modelpipelines.json", jbytes(full_license("label_mismatch")))
        self.add("label_mismatch", "validation_error", errors=["LabelCountMismatch"])

        # unknown_op
        d = self.folder("unknown_op")
        build_unknown_op(os.path.join(d, "model.onnx"), 2)
        self.write(d, "config.json", jbytes(self.cls_config(p["two_labels"])))
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "modelpipelines.json", jbytes(full_license("unknown_op")))
        self.models_seen["unknown_op"] = ("unknown", None)
        self.add("unknown_op", "validation_error", errors=["ModelCreateFailed"])

        # missing_preprocessor
        d = self.folder("missing_preprocessor")
        self.img_cls_model(d)
        self.write(d, "config.json", jbytes(self.img_config(p["img_labels"])))
        self.write(d, "modelpipelines.json", jbytes(full_license("missing_preprocessor")))
        self.add("missing_preprocessor", "validation_error", errors=["InputDimMismatch"],
                 warnings=["MissingPreprocessorConfig"])

        # missing_model
        d = self.folder("missing_model")
        self.write(d, "config.json", jbytes(self.cls_config(
            p["cls_labels"], max_position_embeddings=p["cls_max_pos"], problem_type="multi_label_classification")))
        self.add("missing_model", "parse_error", errors=["MissingModelFile"])

        # bad_config_json
        d = self.folder("bad_config_json")
        self.img_cls_model(d)
        self.write(d, "config.json", b'{"architectures": [')
        self.add("bad_config_json", "parse_error", errors=["InvalidJson"])

        # no_task
        d = self.folder("no_task")
        self.img_cls_model(d)
        self.write(d, "config.json", jbytes({"model_type": "resnet"}))
        self.add("no_task", "parse_error", errors=["TaskUndetected"])

        # bad_id2label
        d = self.folder("bad_id2label")
        self.text_cls_model(d)
        self.copy_tokenizer(d, p["cls_tok"])
        self.write(d, "config.json", jbytes({
            "architectures": ["DistilBertForSequenceClassification"], "model_type": "distilbert",
            "id2label": {"0": "a", "2": "b", "3": "c"}}))
        self.add("bad_id2label", "parse_error", errors=["InvalidLabels"])

        # small_vocab: embedding table smaller than the tokenizer vocab (worst-case id is out of range)
        d = self.folder("small_vocab")
        rows = p["small_vocab_rows"]
        if rows < 0:
            rows += tokenizer_vocab(self.tok[p["emb_tok"]])
        self.emb_model(d, vocab=rows)
        self.emb_folder_files(d, "small_vocab", p["emb_sbert"], p["emb_max_pos"])
        self.add("small_vocab", "validation_error", errors=["SmokeRunFailed"])

        # short_positions: position table shorter than MaxTokens (no max_position_embeddings in config.json)
        d = self.folder("short_positions")
        self.emb_model(d, pos_rows=p["pos_rows"])
        self.emb_folder_files(d, "short_positions", p["pos_sbert"], None)
        self.add("short_positions", "validation_error", errors=["SmokeRunFailed"])


# ----------------------------------------------------------------------------------------------------------------
# Expectations derived from files
# ----------------------------------------------------------------------------------------------------------------

def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


ELEM = {1: "float32", 7: "int64", 6: "int32", 11: "float64"}


def onnx_block(model_path):
    onnx = _onnx()[0]
    m = onnx.load(model_path)

    def one(v):
        tt = v.type.tensor_type
        shape = [d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim]
        return {"name": v.name, "type": ELEM[tt.elem_type], "shape": shape}
    inits = {i.name for i in m.graph.initializer}
    return {"inputs": [one(i) for i in m.graph.input if i.name not in inits], "outputs": [one(o) for o in m.graph.output]}


def derive_full(folder, task_override="None"):
    """Everything the parse stage yields, derived from the files in `folder`. Returns dict or raises ValueError."""
    config = read_json(os.path.join(folder, "config.json"))
    sidecar = read_json(os.path.join(folder, "modelpipelines.json"))
    modules = read_json(os.path.join(folder, "modules.json"))
    sbert = read_json(os.path.join(folder, "sentence_bert_config.json"))
    tokcfg = read_json(os.path.join(folder, "tokenizer_config.json"))
    pre = read_json(os.path.join(folder, "preprocessor_config.json"))
    has_tok = os.path.isfile(os.path.join(folder, "tokenizer.json"))
    task, source = detect_task(task_override, sidecar, modules is not None, config, has_tok)
    if task is None:
        raise ValueError("task undetected in " + folder)
    labels, err = ([], None) if task == "TextEmbedding" else parse_labels((config or {}).get("id2label"))
    if err:
        raise ValueError(err)
    model_file = "model.onnx" if os.path.isfile(os.path.join(folder, "model.onnx")) else "onnx/model.onnx"
    pp = {"max_tokens": 512, "add_special_tokens": True, "pooling": "Mean", "normalize_embeddings": False}
    if task in ("TextEmbedding", "TextClassification"):
        pp["max_tokens"] = text_max_tokens(sbert, tokcfg, config)
    if task == "TextEmbedding":
        pp["pooling"], pp["normalize_embeddings"] = pooling_and_normalize(
            modules, lambda rel: read_json(os.path.join(folder, rel)))
    img = image_preprocess(pre if task in ("ImageClassification", "ObjectDetection") else None)
    if task in ("ImageClassification", "ObjectDetection") and pre is None:
        img = image_preprocess(None)
    pp.update(img)
    pp["score_activation"] = score_activation(task, config)
    readme_text = None
    readme = os.path.join(folder, "README.md")
    if os.path.isfile(readme):
        with open(readme, "r", encoding="utf-8") as f:
            readme_text = f.read()
    lic = resolve_license(sidecar, readme_text)
    device = resolve_device(sidecar)
    return {"task": task, "task_source": source, "model_file": model_file, "labels": labels, "preprocess": pp,
            "license": lic, "preferred_device": device, "onnx": onnx_block(os.path.join(folder, model_file))}


def case_record(case_id, folder_rel, reference, task_override, expect, errors, warnings, derived):
    rec = {"id": case_id, "folder": folder_rel, "reference": reference, "task_override": task_override,
           "expect": expect, "error_codes": errors}
    if derived is not None:
        rec["warning_codes"] = warnings
        for k in ("task", "task_source", "model_file", "labels", "preprocess", "license", "preferred_device", "onnx"):
            rec[k] = derived[k]
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Model self-checks (acceptance 3)
# ----------------------------------------------------------------------------------------------------------------

def check_tiny_models(builder):
    import onnx
    import onnxruntime as ort
    from tokenizers import Tokenizer
    texts = ["The blacksmith is in the village.", "Café 日本 🙂"]
    for case_id, (kind, tokname) in builder.models_seen.items():
        folder = os.path.join(builder.root, case_id)
        mp = os.path.join(folder, "model.onnx")
        if not os.path.isfile(mp):
            mp = os.path.join(folder, "onnx", "model.onnx")
        if kind == "unknown":
            try:
                ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
            except Exception:  # noqa: BLE001
                continue
            raise SystemExit("FAIL: unknown_op model unexpectedly loaded in onnxruntime")
        onnx.checker.check_model(onnx.load(mp))
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        if kind in ("text", "text_emb"):
            tok = Tokenizer.from_file(builder.tok[tokname])
            encs = tok.encode_batch(texts)
            n = max(len(e.ids) for e in encs)
            ids = np.zeros((2, n), dtype=np.int64)
            mask = np.zeros((2, n), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, :len(e.ids)] = e.ids
                mask[i, :len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            feed = {k: feed[k] for k in names}
        elif kind == "wrong":
            feed = {"x": np.ones((2, 4), dtype=np.float32)}
        else:
            shape = sess.get_inputs()[0].shape
            feed = {"pixel_values": np.random.RandomState(5).rand(2, *shape[1:]).astype(np.float32)}
        outs = sess.run(None, feed)
        decl = onnx_block(mp)["outputs"]
        for o, d in zip(outs, decl):
            want = [2 if s == -1 else s for s in d["shape"]]
            if kind in ("text", "text_emb"):
                want = [ids.shape[1] if (i > 0 and s == -1) else (2 if s == -1 else s) for i, s in enumerate(d["shape"])]
            if list(o.shape) != want:
                raise SystemExit("FAIL: %s output %s shape %s != %s" % (case_id, d["name"], o.shape, want))


def _text_feed(sess, ids):
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
    return {i.name: feed[i.name] for i in sess.get_inputs()}


def worst_case_run(model_path, tok_path, max_tokens):
    """Hello world (specials on) and the worst case ([1, max_tokens] of the highest tokenizer id) on ORT CPU.
    Returns (hello_ok, worst_ok, worst_error_text)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    hello = np.array([tok.encode("Hello world", add_special_tokens=True).ids], dtype=np.int64)
    hello_ok = True
    try:
        sess.run(None, _text_feed(sess, hello))
    except Exception:  # noqa: BLE001
        hello_ok = False
    worst = np.full((1, max_tokens), tok.get_vocab_size(with_added_tokens=True) - 1, dtype=np.int64)
    try:
        sess.run(None, _text_feed(sess, worst))
        return hello_ok, True, ""
    except Exception as e:  # noqa: BLE001
        return hello_ok, False, str(e).splitlines()[0][:160]


def check_worst_cases(builder, records):
    """small_vocab and short_positions: Hello world passes, worst case raises an ORT error. Fails loudly."""
    for case_id in ("small_vocab", "short_positions"):
        rec = next(r for r in records if r["id"] == case_id)
        mp = os.path.join(builder.root, case_id, "model.onnx")
        hello_ok, worst_ok, err = worst_case_run(mp, builder.tok[builder.p["emb_tok"]], rec["preprocess"]["max_tokens"])
        print("  worst-case self-check %s: hello_ok=%s worst_ok=%s max_tokens=%d %s" % (
            case_id, hello_ok, worst_ok, rec["preprocess"]["max_tokens"], err))
        if not hello_ok or worst_ok:
            raise SystemExit("FAIL: %s must pass Hello world and fail the worst case" % case_id)


def check_reference_worst_cases(models_root, records):
    for slug in ("all-MiniLM-L6-v2", "distilbert-sst2"):
        rec = next((r for r in records if r["id"] == slug), None)
        folder = os.path.join(models_root, "reference", slug)
        if rec is None:
            print("  reference worst case %s: skipped (folder missing)" % slug)
            continue
        hello_ok, worst_ok, err = worst_case_run(os.path.join(folder, rec["model_file"]),
                                                 os.path.join(folder, "tokenizer.json"),
                                                 rec["preprocess"]["max_tokens"])
        print("  reference worst case %s: hello_ok=%s worst_ok=%s max_tokens=%d %s" % (
            slug, hello_ok, worst_ok, rec["preprocess"]["max_tokens"], err))
        if not worst_ok:
            print("  FINDING: reference model %s fails the worst case on ORT" % slug)


# ----------------------------------------------------------------------------------------------------------------
# Reference cases and transformers cross-check
# ----------------------------------------------------------------------------------------------------------------

def crosscheck_reference(slug, folder, exp):
    import transformers
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
    pp = exp["preprocess"]
    cfg = AutoConfig.from_pretrained(folder)
    if exp["task"] != "TextEmbedding":
        want = [cfg.id2label[i] for i in range(len(cfg.id2label))]
        if exp["labels"] != want:
            raise SystemExit("FAIL: %s labels differ from transformers AutoConfig" % slug)
    if exp["task"] in ("ImageClassification", "ObjectDetection"):
        proc = AutoImageProcessor.from_pretrained(folder)
        out = proc(images=Image.new("RGB", (300, 200), (10, 120, 200)), return_tensors="np")["pixel_values"]
        if list(out.shape[-2:]) != [pp["image_height"], pp["image_width"]]:
            raise SystemExit("FAIL: %s processor output %s vs expected %dx%d" % (
                slug, out.shape, pp["image_height"], pp["image_width"]))
        if pp["resize_mode"] == "ShortestEdgeCenterCrop":
            s = proc.size["shortest_edge"]
            if int(s / proc.crop_pct) != pp["resize_shortest_edge"]:
                raise SystemExit("FAIL: %s resize_shortest_edge %d != transformers %d" % (
                    slug, pp["resize_shortest_edge"], int(s / proc.crop_pct)))
        else:
            if not (isinstance(proc.size, dict) and proc.size.get("height") == pp["image_height"]
                    and proc.size.get("width") == pp["image_width"]):
                raise SystemExit("FAIL: %s processor size %s disagrees" % (slug, proc.size))
        do_norm = bool(getattr(proc, "do_normalize", True))
        if do_norm != (pp["image_std"] != [1.0, 1.0, 1.0]):
            raise SystemExit("FAIL: %s do_normalize disagrees with transformers" % slug)
        if do_norm and (list(proc.image_mean) != pp["image_mean"] or list(proc.image_std) != pp["image_std"]):
            raise SystemExit("FAIL: %s mean/std disagree with transformers" % slug)
        rf = proc.rescale_factor if getattr(proc, "do_rescale", True) else 1.0
        if abs(rf - pp["rescale_factor"]) > 1e-9:
            raise SystemExit("FAIL: %s rescale_factor disagrees with transformers" % slug)
    if exp["task"] in ("TextEmbedding", "TextClassification"):
        tok = AutoTokenizer.from_pretrained(folder)
        mml = tok.model_max_length
        has_sbert = os.path.isfile(os.path.join(folder, "sentence_bert_config.json"))
        if mml <= 65536:
            if pp["max_tokens"] > mml or (not has_sbert and pp["max_tokens"] != mml):
                raise SystemExit("FAIL: %s max_tokens %d vs tokenizer model_max_length %d" % (
                    slug, pp["max_tokens"], mml))
    print("  cross-checked %s against transformers %s" % (slug, transformers.__version__))


def reference_cases(models_root):
    recs = []
    ref_root = os.path.join(models_root, "reference")
    for slug in REFERENCE_SLUGS + ["resnet_as_detection"]:
        real = "resnet-50" if slug == "resnet_as_detection" else slug
        folder = os.path.join(ref_root, real)
        if not os.path.isfile(os.path.join(folder, "model.onnx")):
            print("WARNING: reference folder missing, case %s left out: %s" % (slug, folder))
            continue
        if slug == "resnet_as_detection":
            d = derive_full(folder, "ObjectDetection")
            recs.append(case_record(slug, "reference/" + real, True, "ObjectDetection", "validation_error",
                                    ["MissingOutput", "OutputRankMismatch"], ["UnverifiedDetectionFamily"], d))
            continue
        d = derive_full(folder)
        crosscheck_reference(slug, folder, d)
        recs.append(case_record(slug, "reference/" + slug, True, "None", "valid", [], [], d))
    return recs


def main():
    dev = _paths.dev_dir()
    ap = argparse.ArgumentParser(description="Generate importer test folders and import_expectations.json.")
    ap.add_argument("--variant", choices=["visible", "hidden"], default="visible")
    ap.add_argument("--out-models", default=os.path.join(dev, "models"),
                    help="dir that receives import_test/<case> (default <dev>/models)")
    ap.add_argument("--out-goldens", default=os.path.join(dev, "goldens", "import", "import_expectations.json"))
    args = ap.parse_args()
    out_models = os.path.abspath(args.out_models)
    b = Builder(out_models, dev, args.variant)
    b.build_all()
    check_tiny_models(b)
    records = []
    for c in b.cases:
        folder = os.path.join(b.root, c["id"])
        rel = "import_test/" + c["id"]
        if c["expect"] == "parse_error":
            records.append(case_record(c["id"], rel, False, c["task_override"], c["expect"], c["errors"], [], None))
        else:
            records.append(case_record(c["id"], rel, False, c["task_override"], c["expect"], c["errors"],
                                       c["warnings"], derive_full(folder, c["task_override"])))
    check_worst_cases(b, records)
    models_root = os.path.join(dev, "models")
    records += reference_cases(models_root)
    check_reference_worst_cases(models_root, records)
    import onnx
    import onnxruntime
    import transformers
    doc = {"schema": "pipelines-import-expectations/1",
           "generator": {"script": "Tools/make_import_test_data.py", "variant": args.variant,
                         "transformers": transformers.__version__, "onnx": onnx.__version__,
                         "onnxruntime": onnxruntime.__version__},
           "models_root": models_root, "import_root": out_models, "cases": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_goldens)), exist_ok=True)
    with open(args.out_goldens, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote %d tiny + %d reference cases -> %s" % (len(b.cases), len(records) - len(b.cases), args.out_goldens))


if __name__ == "__main__":
    main()
