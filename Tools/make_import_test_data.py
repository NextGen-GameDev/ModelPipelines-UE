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
TASKS = ["TextEmbedding", "TextClassification", "ImageClassification", "ObjectDetection", "SpeechToText"]
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


# C25: union of (a) classes whose HF code calls get_size_dict(size, default_to_square=False) and (b) classes whose
# constructor turns size=40 into {shortest_edge: 40}: an integer `size` is the shortest edge there. Any other class
# treats an integer as a square. Pinned by a pytest that re-scans transformers.
SHORTEST_EDGE_INT_CLASSES = frozenset("""
BitImageProcessor BlipImageProcessor BridgeTowerImageProcessor ChameleonImageProcessor
ChineseCLIPImageProcessor CLIPImageProcessor ConditionalDetrImageProcessor ConvNextImageProcessor
DeepseekVLImageProcessor DeformableDetrImageProcessor DetaImageProcessor DetrImageProcessor
EfficientLoFTRImageProcessor EomtImageProcessor GotOcr2ImageProcessor GroundingDinoImageProcessor
InstructBlipVideoImageProcessor JanusImageProcessor LevitImageProcessor LightGlueImageProcessor LlavaImageProcessor
LlavaNextImageProcessor LlavaNextVideoImageProcessor LlavaOnevisionImageProcessor Mask2FormerImageProcessor
MaskFormerImageProcessor MobileNetV1ImageProcessor MobileNetV2ImageProcessor MobileViTImageProcessor
OneFormerImageProcessor Ovis2ImageProcessor PoolFormerImageProcessor RTDetrImageProcessor SuperGlueImageProcessor
SuperPointImageProcessor TextNetImageProcessor TvltImageProcessor TvpImageProcessor VideoLlavaImageProcessor
VideoMAEImageProcessor ViltImageProcessor ViTHybridImageProcessor VivitImageProcessor YolosImageProcessor
""".split())


def processor_class_name(pre):
    """C25: image_processor_type, else feature_extractor_type with FeatureExtractor -> ImageProcessor; trailing Fast
    removed; "" if neither key is a string."""
    name = ""
    v = (pre or {}).get("image_processor_type")
    if isinstance(v, str) and v:
        name = v
    else:
        v = (pre or {}).get("feature_extractor_type")
        if isinstance(v, str) and v:
            name = v[:-len("FeatureExtractor")] + "ImageProcessor" if v.endswith("FeatureExtractor") else v
    if name.endswith("Fast"):
        name = name[:-len("Fast")]
    return name


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _usable_crop(pre):
    """(w, h) if do_center_crop is true and crop_size is an integer or {height, width}; else None."""
    crop = pre.get("crop_size")
    if not pre.get("do_center_crop") or crop is None:
        return None
    if isinstance(crop, dict):
        if "height" in crop and "width" in crop:
            return int(crop["width"]), int(crop["height"])
        return None
    if _is_int(crop):
        return int(crop), int(crop)
    return None


# C32: the `resample` default of each slow image processor class whose __init__ default is not Bilinear (PIL values:
# 0 Nearest, 1 Lanczos, 3 Bicubic). Any other class (and an empty class name) defaults to Bilinear (2). Pinned by a
# pytest that re-scans transformers 4.57.6.
RESAMPLE_DEFAULTS = {n + "ImageProcessor": 3 for n in """
Aria Beit Bit Blip BridgeTower CLIP ChineseCLIP DPT DeepseekVLHybrid DeepseekVL DeiT Emu3 Flava Glm4v GotOcr2
InstructBlipVideo Janus Levit Llava LlavaNext LlavaNextVideo LlavaOnevision Ovis2 OwlViT Perceiver Pixtral PoolFormer
PromptDepthAnything Qwen2VL SegGpt Siglip VideoLlava Vilt
""".split()}
RESAMPLE_DEFAULTS.update({"ChameleonImageProcessor": 1, "Idefics3ImageProcessor": 1, "SmolVLMImageProcessor": 1,
                          "EfficientNetImageProcessor": 0})
RESAMPLE_NAMES = {0: "Nearest", 1: "Lanczos", 2: "Bilinear", 3: "Bicubic", 4: "Box", 5: "Hamming"}
LEVIT = "LevitImageProcessor"


def derive_resample(pre, cls):
    """C32 rule 1: (PIL int, warn). `resample` present and an integer 0..5 (int, or float with is_integer(); bool
    excluded) -> that value; absent -> class default (Bilinear if unlisted); anything else -> Bilinear + warning."""
    if "resample" not in pre:
        return RESAMPLE_DEFAULTS.get(cls, 2), False
    v = pre["resample"]
    if isinstance(v, bool):
        return 2, True
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and 0 <= v <= 5:
        return v, False
    return 2, True


def image_preprocess(pre):
    """Image fields from a preprocessor_config dict (None = file missing). Implements C14 rule 7 as replaced by C25."""
    out = {"image_width": 224, "image_height": 224, "resize_mode": "Stretch", "resize_shortest_edge": 0,
           "resize_width": 0, "resize_height": 0,
           "rescale_factor": F32_255, "image_mean": list(IMAGENET_MEAN), "image_std": list(IMAGENET_STD),
           "resample": "Bilinear", "nms_iou_threshold": 0.7, "max_detections": 100}
    if pre is None:
        return out
    size = pre.get("size")
    cls = processor_class_name(pre)
    out["resample"] = RESAMPLE_NAMES[derive_resample(pre, cls)[0]]
    hw = None  # (rw, rh) from an object size or a square-default integer
    shortest = None
    if isinstance(size, dict) and "height" in size and "width" in size:
        hw = (int(size["width"]), int(size["height"]))
    elif isinstance(size, dict) and "shortest_edge" in size:
        shortest = int(size["shortest_edge"])
    elif _is_int(size):
        if cls and "crop_pct" not in pre and cls not in SHORTEST_EDGE_INT_CLASSES:
            hw = (size, size)  # rule 3
        else:
            shortest = size
    if shortest is not None and "crop_pct" in pre:  # rule 1
        if shortest < 384:
            out["resize_mode"] = "ShortestEdgeCenterCrop"
            out["resize_shortest_edge"] = int(shortest / pre["crop_pct"])
        out["image_width"] = out["image_height"] = shortest
    elif hw is not None:  # rules 2, 3
        crop = _usable_crop(pre)
        if crop is not None:
            out["resize_mode"] = "StretchCenterCrop"
            out["resize_width"], out["resize_height"] = hw
            out["image_width"], out["image_height"] = crop
        else:
            out["image_width"], out["image_height"] = hw
    elif shortest is not None:  # rule 4
        crop = _usable_crop(pre)
        if crop is not None:
            out["resize_mode"] = "ShortestEdgeCenterCrop"
            # rule 2: LeViT resizes to int(256/224 * S) before the centre crop
            out["resize_shortest_edge"] = int((256.0 / 224.0) * shortest) if cls == LEVIT else shortest
            out["image_width"], out["image_height"] = crop
        else:
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


def image_warnings(pre):
    """Warning codes the parser must raise for a preprocessor_config dict, restricted to the C32 rules 1 and 3
    (UnsupportedResize for a bad `resample`, or do_center_crop without crop_size in the height/width paths)."""
    if pre is None:
        return []
    if derive_resample(pre, processor_class_name(pre))[1]:
        return ["UnsupportedResize"]
    size = pre.get("size")
    if (isinstance(size, dict) and "height" in size and "width" in size and pre.get("do_center_crop")
            and pre.get("crop_size") is None):
        return ["UnsupportedResize"]
    return []


def scan_resample_defaults():
    """Import every transformers/models/*/image_processing_*.py (no _fast); for each class ending in ImageProcessor
    with a `resample` parameter on __init__ record the default as an int. Returns (all {class: int}, {module: error})."""
    import glob
    import importlib
    import inspect
    import transformers
    root = os.path.join(os.path.dirname(transformers.__file__), "models")
    found, failed = {}, {}
    for path in sorted(glob.glob(os.path.join(root, "*", "image_processing_*.py"))):
        if path.endswith("_fast.py"):
            continue
        mod = "transformers.models.%s.%s" % (os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-3])
        try:
            mm = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            failed[mod] = repr(e)[:120]
            continue
        for n, c in inspect.getmembers(mm, inspect.isclass):
            if n.endswith("ImageProcessor") and c.__module__ == mod:
                p = inspect.signature(c.__init__).parameters.get("resample")
                if p is not None and p.default is not inspect.Parameter.empty:
                    found[n] = int(p.default)
    return found, failed


def scan_shortest_edge_classes():
    """Re-scan the installed transformers: classes in transformers/models/*/image_processing_*.py (no _fast) with a
    method that calls get_size_dict(size, default_to_square=False) where `size` is the first positional argument and
    the only other keyword allowed is max_size=self._something. Calls with param_name= (Siglip, Gemma3) or a plain
    max_size parameter (the DETR family) do not count; that is how C25's list for transformers 4.57.6 was built. The
    call sits in __init__ for most classes and in preprocess/resize for a few (Blip, Tvp)."""
    import ast
    import glob
    import transformers
    root = os.path.join(os.path.dirname(transformers.__file__), "models")
    found = set()
    for path in sorted(glob.glob(os.path.join(root, "*", "image_processing_*.py"))):
        if path.endswith("_fast.py"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:  # top-level classes
            if not isinstance(node, ast.ClassDef):
                continue
            for call in ast.walk(node):
                if not (isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", "")) == "get_size_dict"):
                    continue
                kws = {kw.arg: kw.value for kw in call.keywords}
                dts = kws.get("default_to_square")
                first = call.args[0] if call.args else None
                max_size = kws.get("max_size")
                if (isinstance(dts, ast.Constant) and dts.value is False
                        and isinstance(first, ast.Name) and first.id == "size"
                        and set(kws) <= {"default_to_square", "max_size"}
                        and (max_size is None or isinstance(max_size, ast.Attribute))):
                    found.add(node.name)
    return found


def scan_shortest_edge_classes_empirical():
    """C25 scan (b): every non-Fast class in IMAGE_PROCESSOR_MAPPING_NAMES is built with size=40 (classes that raise
    are skipped); it counts when the resulting .size is a dict with shortest_edge and no height."""
    import transformers
    from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
    names = set()
    for v in IMAGE_PROCESSOR_MAPPING_NAMES.values():
        for n in (v if isinstance(v, (list, tuple)) else [v]):
            if n and not n.endswith("Fast"):
                names.add(n)
    found = set()
    for n in sorted(names):
        cls = getattr(transformers, n, None)
        if cls is None:
            continue
        try:
            size = cls(size=40).size
        except Exception:
            continue
        if isinstance(size, dict) and "shortest_edge" in size and "height" not in size:
            found.add(n)
    return found


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
        if a == "ParakeetForTDT":
            return "SpeechToText", "architectures"
        if a.endswith("ForSequenceClassification"):
            return "TextClassification", "architectures"
        if a.endswith("ForImageClassification"):
            return "ImageClassification", "architectures"
        if a.endswith("ForObjectDetection"):
            return "ObjectDetection", "architectures"
        if a.endswith("Model") and has_tokenizer:
            return "TextEmbedding", "architectures"
    return None, None


# C78b: speech settings. JSON keys use the snake_case style of the other expectation fields; the UE names are in the
# comments. Defaults apply when a key is missing; a key of the wrong JSON type is an InvalidJson error naming
# "<file>: <key>".
SPEECH_DEFAULTS = {"sample_rate": 16000, "num_mel_bins": 128, "fft_size": 512, "window_length": 400,
                   "hop_length": 160, "preemphasis": 0.97, "blank_token_id": 8192, "durations": [0, 1, 2, 3, 4],
                   "max_symbols_per_step": 10, "subsampling_factor": 8, "silence_threshold_db": -60.0}
# (settings key, file, path inside the file's JSON, kind)
SPEECH_SOURCES = [("sample_rate", "processor_config.json", ("feature_extractor", "sampling_rate"), "int"),
                  ("num_mel_bins", "processor_config.json", ("feature_extractor", "feature_size"), "int"),
                  ("fft_size", "processor_config.json", ("feature_extractor", "n_fft"), "int"),
                  ("window_length", "processor_config.json", ("feature_extractor", "win_length"), "int"),
                  ("hop_length", "processor_config.json", ("feature_extractor", "hop_length"), "int"),
                  ("preemphasis", "processor_config.json", ("feature_extractor", "preemphasis"), "float"),
                  ("blank_token_id", "config.json", ("blank_token_id",), "int"),
                  ("durations", "config.json", ("durations",), "int_list"),
                  ("max_symbols_per_step", "config.json", ("max_symbols_per_step",), "int"),
                  ("subsampling_factor", "config.json", ("encoder_config", "subsampling_factor"), "int")]


class SpeechJsonError(ValueError):
    """InvalidJson; str() names "<file>: <key>"."""


def speech_settings(config, proc):
    """C78b rule 3. config / proc are the parsed config.json / processor_config.json dicts (None = file missing)."""
    out = dict(SPEECH_DEFAULTS)
    out["durations"] = list(SPEECH_DEFAULTS["durations"])
    docs = {"config.json": config, "processor_config.json": proc}
    for key, fname, path, kind in SPEECH_SOURCES:
        node = docs[fname]
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            continue
        ok = ((kind == "int" and _is_int(node)) or
              (kind == "float" and isinstance(node, (int, float)) and not isinstance(node, bool)) or
              (kind == "int_list" and isinstance(node, list) and all(_is_int(x) for x in node)))
        if not ok:
            raise SpeechJsonError("%s: %s" % (fname, path[-1]))
        out[key] = float(node) if kind == "float" else node
    return out


def speech_files(folder):
    """C78b rule 2: (encoder.onnx, decoder_joint.onnx) relative names; raises ValueError('MissingModelFile: <name>')
    when one is missing or empty. model.onnx is never used."""
    for name in ("encoder.onnx", "decoder_joint.onnx"):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise ValueError("MissingModelFile: " + name)
    return "encoder.onnx", "decoder_joint.onnx"


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
            small_vocab_rows=30000, pos_rows=32, pos_sbert=64,
            deit=dict(size=40, crop=32, hw_size=(48, 40), hw_crop=(36, 32), nocrop=32),
            rs=dict(levit_s=28, levit_crop=28, size=32))
    return dict(
        emb_hidden=6, emb_tok="t5-small", cls_tok="gpt2",
        cls_labels=["sunny", "rain", "ñandú", "snow", "fog"], cls_max_pos=96,
        emb_sbert=32, emb_max_pos=100, emb_pool_cls=False, emb_normalize=False, sidecar_tokcfg=200,
        img_h=48, img_w=48, img_labels=["sky", "sea", "лес", "sand"], crop_pct=0.9, img_size=48,
        img_mean=[0.45, 0.5, 0.55], img_std=[0.25, 0.2, 0.15],
        det_h=48, det_w=40, det_q=7, det_labels=["drone", "kite", "balloon"],
        two_labels=["BAD", "GOOD"], mismatch_labels=["lo", "mid", "hi", "top"],
        small_vocab_rows=-1000, pos_rows=16, pos_sbert=48,
        deit=dict(size=52, crop=44, hw_size=(60, 50), hw_crop=(48, 40), nocrop=44),
        rs=dict(levit_s=42, levit_crop=44, size=48))


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

    def add(self, case_id, expect, errors=(), warnings=(), task_override="None", message_contains=None,
            parse_only=False):
        self.cases.append(dict(id=case_id, expect=expect, errors=list(errors), warnings=list(warnings),
                               task_override=task_override, message_contains=message_contains,
                               parse_only=parse_only))

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


        # C25 resize-then-crop cases: tiny image classifiers whose input is the crop / stretch size
        dp = p["deit"]
        rs = p["rs"]
        labels3 = ["cat", "dog", "fox"]
        common = {"do_resize": True, "do_rescale": True, "rescale_factor": 0.00392156862745098, "do_normalize": True,
                  "image_mean": p["img_mean"], "image_std": p["img_std"]}
        hw = lambda wh: {"height": wh[1], "width": wh[0]}  # noqa: E731
        c25 = [
            ("deit_style", dict(image_processor_type="DeiTImageProcessor", size=dp["size"], do_center_crop=True,
                                crop_size=dp["crop"]), (dp["crop"], dp["crop"])),
            ("deit_hw_crop", dict(image_processor_type="DeiTImageProcessor", size=hw(dp["hw_size"]),
                                  do_center_crop=True, crop_size=hw(dp["hw_crop"])), dp["hw_crop"]),
            ("deit_legacy_fe", dict(feature_extractor_type="DeiTFeatureExtractor", size=dp["size"],
                                    do_center_crop=True, crop_size=dp["crop"]), (dp["crop"], dp["crop"])),
            ("clip_style_crop", dict(image_processor_type="CLIPImageProcessor", size=dp["size"], do_center_crop=True,
                                     crop_size=dp["crop"]), (dp["crop"], dp["crop"])),
            ("square_int_nocrop", dict(image_processor_type="DeiTImageProcessor", size=dp["nocrop"],
                                       do_center_crop=False), (dp["nocrop"], dp["nocrop"])),
            # WP4.4 / C32: resample, LeViT rule, crop without size
            ("levit_style", dict(image_processor_type="LevitImageProcessor", size={"shortest_edge": rs["levit_s"]},
                                 do_center_crop=True, crop_size=hw((rs["levit_crop"], rs["levit_crop"]))),
             (rs["levit_crop"], rs["levit_crop"])),
            ("crop_no_size", dict(image_processor_type="DeiTImageProcessor", size=hw((rs["size"], rs["size"])),
                                  do_center_crop=True, resample=2), (rs["size"], rs["size"])),
            ("resample_box", dict(image_processor_type="ViTImageProcessor", size=hw((rs["size"], rs["size"])),
                                  resample=4), (rs["size"], rs["size"])),
            ("resample_bad", dict(image_processor_type="ViTImageProcessor", size=hw((rs["size"], rs["size"])),
                                  resample="bicubic"), (rs["size"], rs["size"])),
            ("effnet_default", dict(image_processor_type="EfficientNetImageProcessor",
                                    size=hw((rs["size"], rs["size"])), do_center_crop=False), (rs["size"], rs["size"])),
        ]
        for case_id, pre, (mw, mh) in c25:
            d = self.folder(case_id)
            build_image_classification(os.path.join(d, "model.onnx"), mh, mw, len(labels3))
            self.models_seen[case_id] = ("image", None)
            self.write(d, "config.json", jbytes(self.img_config(labels3)))
            body = dict(pre)
            body.update(common)
            self.write(d, "preprocessor_config.json", jbytes(body))
            self.write(d, "modelpipelines.json", jbytes(full_license(case_id)))
            self.add(case_id, "valid", warnings=image_warnings(body))

        self.build_stt_cases()

    def build_stt_cases(self):
        """WP9S.6t / C78b: tiny speech folders (tiny graphs from make_test_models.py, Phonon's tokenizer.json)."""
        models = os.path.join(self.dev, "models")
        enc = os.path.join(models, "test", "tiny_stt_encoder.onnx")
        dec = os.path.join(models, "test", "tiny_stt_decoder_joint.onnx")
        tok = os.path.join(models, "stt", "phonon-2-onnx", "fp16w", "tokenizer.json")
        for need in (enc, dec, tok):
            if not os.path.isfile(need):
                raise SystemExit("NEEDS: %s missing (tiny STT graphs: make_test_models.py; tokenizer: "
                                 "export_phonon2_onnx.py --variant fp16w)" % need)

        def base(case_id, decoder=True, model_onnx=False):
            d = self.folder(case_id)
            shutil.copyfile(enc, os.path.join(d, "encoder.onnx"))
            if decoder:
                shutil.copyfile(dec, os.path.join(d, "decoder_joint.onnx"))
            if model_onnx:
                shutil.copyfile(enc, os.path.join(d, "model.onnx"))
            shutil.copyfile(tok, os.path.join(d, "tokenizer.json"))
            return d

        def config(arch=True):
            c = {"model_type": "parakeet_tdt", "blank_token_id": 5, "durations": [0, 1, 2],
                 "max_symbols_per_step": 10, "encoder_config": {"subsampling_factor": 8}}
            if arch:
                c = dict(architectures=["ParakeetForTDT"], **c)
            return c

        def processor():
            return {"processor_class": "ParakeetProcessor", "feature_extractor": {
                "feature_extractor_type": "ParakeetFeatureExtractor", "feature_size": 8,
                "hop_length": 160, "n_fft": 512, "padding_side": "right", "padding_value": 0.0, "preemphasis": 0.97,
                "return_attention_mask": True, "sampling_rate": 16000, "win_length": 400}}

        def mit(case_id):
            return {"license": {"spdx_id": "MIT", "source_url": "https://example.invalid/" + case_id,
                                "attribution": "ModelPipelines test data"}}

        for case_id, model_onnx in (("stt_tiny_arch", False), ("stt_with_model_onnx", True)):
            d = base(case_id, model_onnx=model_onnx)
            self.write(d, "config.json", jbytes(config()))
            self.write(d, "processor_config.json", jbytes(processor()))
            self.write(d, "modelpipelines.json", jbytes(mit(case_id)))
            self.add(case_id, "valid")

        d = base("stt_tiny_sidecar_task")
        self.write(d, "config.json", jbytes(config(arch=False)))
        self.write(d, "processor_config.json", jbytes(processor()))
        sc = {"task": "SpeechToText", "license": {
            "spdx_id": "CC-BY-4.0", "source_url": "https://example.invalid/stt_tiny_sidecar_task",
            "attribution": "ModelPipelines test data (CC-BY-4.0)"}}
        self.write(d, "modelpipelines.json", jbytes(sc))
        self.add("stt_tiny_sidecar_task", "valid")

        d = base("stt_no_decoder", decoder=False)
        self.write(d, "config.json", jbytes(config()))
        self.write(d, "processor_config.json", jbytes(processor()))
        self.write(d, "modelpipelines.json", jbytes(mit("stt_no_decoder")))
        self.add("stt_no_decoder", "parse_error", errors=["MissingModelFile"], message_contains=["decoder_joint.onnx"])

        d = base("stt_bad_speech_json")
        self.write(d, "config.json", jbytes(config()))
        pc = processor()
        pc["feature_extractor"]["feature_size"] = "8"
        self.write(d, "processor_config.json", jbytes(pc))
        self.write(d, "modelpipelines.json", jbytes(mit("stt_bad_speech_json")))
        self.add("stt_bad_speech_json", "parse_error", errors=["InvalidJson"],
                 message_contains=["processor_config.json: feature_size"])

        d = base("stt_defaults")
        self.write(d, "config.json", jbytes({"architectures": ["ParakeetForTDT"], "model_type": "parakeet_tdt"}))
        self.write(d, "processor_config.json", jbytes({"processor_class": "ParakeetProcessor", "feature_extractor": {
            "feature_extractor_type": "ParakeetFeatureExtractor"}}))
        self.write(d, "modelpipelines.json", jbytes(mit("stt_defaults")))
        # defaults (128 mel bins, blank 8192, ...) do not match the tiny graphs: the validation outcome is not specified
        self.add("stt_defaults", "validation_error", parse_only=True)


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
    labels, err = ([], None) if task in ("TextEmbedding", "SpeechToText") else parse_labels((config or {}).get("id2label"))
    if err:
        raise ValueError(err)
    decoder_file = ""
    speech = dict(SPEECH_DEFAULTS)
    if task == "SpeechToText":
        model_file, decoder_file = speech_files(folder)
        speech = speech_settings(config, read_json(os.path.join(folder, "processor_config.json")))
    else:
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
            "license": lic, "preferred_device": device, "onnx": onnx_block(os.path.join(folder, model_file)),
            "decoder_model_file": decoder_file, "speech": speech}


def check_stt_parse_error(case, folder):
    """The two speech parse-error cases must really fail under the C78b rules, with the message named in the case."""
    if not case["id"].startswith("stt_"):
        return
    try:
        derive_full(folder, case["task_override"])
    except ValueError as e:  # SpeechJsonError is a ValueError
        text = str(e)
    else:
        raise SystemExit("FAIL: %s was expected to fail to parse" % case["id"])
    for want in case["message_contains"] or []:
        if want not in text:
            raise SystemExit("FAIL: %s error %r lacks %r" % (case["id"], text, want))


def case_record(case_id, folder_rel, reference, task_override, expect, errors, warnings, derived,
                message_contains=None, parse_only=False):
    rec = {"id": case_id, "folder": folder_rel, "reference": reference, "task_override": task_override,
           "expect": expect, "error_codes": errors}
    if message_contains:  # optional: substrings the error message must contain (WP9S.6t)
        rec["error_message_contains"] = list(message_contains)
    if parse_only:  # optional: only the parse stage is specified; the validation outcome is not
        rec["parse_only"] = True
    if derived is not None:
        rec["warning_codes"] = warnings
        for k in ("task", "task_source", "model_file", "labels", "preprocess", "license", "preferred_device", "onnx",
                  "decoder_model_file", "speech"):
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


C25_CASES = ["deit_style", "deit_hw_crop", "deit_legacy_fe", "clip_style_crop", "square_int_nocrop",
             "levit_style", "resample_box", "effnet_default"]
# HF cross-check of the c25 mode/size is skipped for these: HF has no equivalent (crop_no_size: HF would crop to the
# processor default 224; resample_bad: HF keeps the string and fails/differs at resize time)
C25_SKIP = {"crop_no_size", "resample_bad"}
RESAMPLE_SKIP = {"resample_bad": "HF keeps resample='bicubic' as a string (not a PIL int); differs by design"}


def crosscheck_resample(case_id, folder, pp):
    """HF slow processor's .resample must equal the derived PIL int; LeViT: output on a 300x200 image must equal
    PIL resize to the derived shortest edge + centre crop + rescale + normalize."""
    from PIL import Image
    from transformers import AutoImageProcessor
    if case_id in RESAMPLE_SKIP:
        print("  resample cross-check %s: skipped (%s)" % (case_id, RESAMPLE_SKIP[case_id]))
        return
    import shutil
    import tempfile
    # load from a copy holding only preprocessor_config.json: transformers reads other files with the locale
    # encoding on Windows, which breaks on the non-ASCII labels in config.json
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(os.path.join(folder, "preprocessor_config.json"), os.path.join(tmp, "preprocessor_config.json"))
        proc = AutoImageProcessor.from_pretrained(tmp, use_fast=False)
    got = int(proc.resample)
    want = {v: k for k, v in RESAMPLE_NAMES.items()}[pp["resample"]]
    if got != want:
        raise SystemExit("FAIL: %s derived resample %s (%d) != transformers %d" % (case_id, pp["resample"], want, got))
    extra = ""
    if type(proc).__name__ == LEVIT:
        rng = np.random.RandomState(3)
        img = Image.fromarray(rng.randint(0, 256, (200, 300, 3), dtype=np.uint8))
        hf = proc(images=img, return_tensors="np")["pixel_values"][0]
        s = pp["resize_shortest_edge"]
        w, h = (int(s * 300 / 200), s)  # 300x200 landscape: short side = height
        r = img.resize((w, h), resample=want)
        cw, ch = pp["image_width"], pp["image_height"]
        l, t = int(round((w - cw) / 2.0)), int(round((h - ch) / 2.0))
        arr = np.asarray(r.crop((l, t, l + cw, t + ch)), dtype=np.float32) * np.float32(pp["rescale_factor"])
        arr = (arr - np.array(pp["image_mean"], np.float32)) / np.array(pp["image_std"], np.float32)
        mine = arr.transpose(2, 0, 1)
        d = float(np.abs(mine - hf).max())
        if d > 1e-4:
            raise SystemExit("FAIL: %s LeViT manual resize+crop differs from HF output by %g" % (case_id, d))
        extra = " levit_manual_max_abs_diff=%.2e" % d
    print("  resample cross-check %s: %s (%d) == transformers%s" % (case_id, pp["resample"], got, extra))


def crosscheck_c25(case_id, folder, pp):
    """The derived mode/sizes must agree with AutoImageProcessor.from_pretrained(folder) and a run on a 300x200 image."""
    from PIL import Image
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(folder)
    out = proc(images=Image.new("RGB", (300, 200), (10, 120, 200)), return_tensors="np")["pixel_values"]
    oh, ow = out.shape[-2:]
    mode = pp["resize_mode"]
    size = proc.size
    ok = True
    if mode == "StretchCenterCrop":
        ok = (bool(proc.do_center_crop) and size.get("height") == pp["resize_height"]
              and size.get("width") == pp["resize_width"]
              and proc.crop_size.get("height") == pp["image_height"] and proc.crop_size.get("width") == pp["image_width"]
              and (oh, ow) == (pp["image_height"], pp["image_width"]) and pp["resize_shortest_edge"] == 0)
    elif mode == "ShortestEdgeCenterCrop":
        s = size.get("shortest_edge")
        if type(proc).__name__.removesuffix("Fast") == LEVIT:  # HF LevitImageProcessor.resize scales S by 256/224 itself
            s = int((256 / 224) * s)
        ok = (bool(proc.do_center_crop) and s == pp["resize_shortest_edge"]
              and (oh, ow) == (pp["image_height"], pp["image_width"])
              and pp["resize_width"] == 0 and pp["resize_height"] == 0)
    else:
        ok = (not getattr(proc, "do_center_crop", False) and size.get("height") == pp["image_height"]
              and size.get("width") == pp["image_width"] and (oh, ow) == (pp["image_height"], pp["image_width"])
              and pp["resize_width"] == 0 and pp["resize_height"] == 0)
    print("  c25 cross-check %s: mode=%s hf size=%s crop=%s do_center_crop=%s out=%dx%d -> %s" % (
        case_id, mode, size, getattr(proc, "crop_size", None), getattr(proc, "do_center_crop", None), oh, ow,
        "ok" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit("FAIL: %s derived preprocess %s disagrees with transformers (size %s, output %dx%d)" % (
            case_id, {k: pp[k] for k in ("resize_mode", "resize_width", "resize_height", "resize_shortest_edge",
                                        "image_width", "image_height")}, size, oh, ow))


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
            check_stt_parse_error(c, folder)
            records.append(case_record(c["id"], rel, False, c["task_override"], c["expect"], c["errors"], [], None,
                                       c["message_contains"]))
        else:
            records.append(case_record(c["id"], rel, False, c["task_override"], c["expect"], c["errors"],
                                       c["warnings"], derive_full(folder, c["task_override"]), None,
                                       c["parse_only"]))
    check_worst_cases(b, records)
    for r in records:
        if r["id"] in C25_CASES:
            crosscheck_c25(r["id"], os.path.join(b.root, r["id"]), r["preprocess"])
    models_root = os.path.join(dev, "models")
    records += reference_cases(models_root)
    for r in records:
        if r.get("task") in ("ImageClassification", "ObjectDetection") and r["expect"] == "valid":
            base = models_root if r["reference"] else out_models
            crosscheck_resample(r["id"], os.path.join(base, r["folder"]), r["preprocess"])
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
