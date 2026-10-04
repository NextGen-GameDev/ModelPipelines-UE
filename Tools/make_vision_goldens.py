#!/usr/bin/env python
"""Vision goldens, part A (WP4.1, schema pipelines-vision-goldens/1, contract C30).

Writes deterministic synthetic test images, Pillow resize results (6 filters), full Hugging Face slow-processor outputs
for 7 processor configs (plus the intermediate 8-bit cropped image), and onnxruntime results for 4 tiny vision models
(two importer models, two new ones with designed constant detection outputs). Everything comes from Pillow,
transformers, onnxruntime or an independent numpy reference. Nothing comes from Unreal. No network.

Each preprocess case is cross-checked against an independent implementation of the C27 pixel rules; the script exits
non-zero if HF and C27 disagree. The detection `detections` lists come from a numpy implementation of the C29 decode
rules 1-4 and the C26 NMS (float64 IoU).

Part B (WP4.2, C31): reference photos (Wikimedia Commons CC0 / Public Domain, fetched by fetch_test_images.py) run through
resnet-50 and rtdetr_r18vd with PyTorch (offline HF cache, pinned revision) and cross-checked with onnxruntime.
Adds `reference_images` and `reference_models` to the JSON; the hidden variant uses the *_hidden pools.

Hidden variant: --variant hidden --seed 777 --out-dir <dir> --models-out <dir> (other images, resize pairs and designed
models; the two importer tiny models are copied into <models-out>/import_test so that models_root is self-contained).
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import make_import_test_data as mit  # noqa: E402

SCHEMA = "pipelines-vision-goldens/1"
SCRIPT = "Tools/make_vision_goldens.py"
DEFAULT_SEEDS = {"visible": 4242, "hidden": 777}
FILTERS = [("Nearest", 0), ("Lanczos", 1), ("Bilinear", 2), ("Bicubic", 3), ("Box", 4), ("Hamming", 5)]
RESAMPLE_NAMES = {i: n for n, i in FILTERS}
CROSSCHECK_TOL = 1e-5
MARGIN = 0.02
# (score_threshold, nms_iou_threshold, max_detections), the same list for all detection models (C30)
DET_PARAMS = [(0.5, 0.7, 100), (0.3, 0.5, 100), (0.5, 1.0, 100), (0.05, 0.5, 3)]
TINY_IMAGES = ["s00", "s02", "s04", "s07"]
DESIGNED_IMAGES = ["s02", "s07"]

IMAGE_SIZES = {
    "visible": [(640, 480), (480, 640), (300, 200), (1920, 1080), (37, 53), (1000, 50), (1, 1), (224, 224), (255, 256),
                (50, 1000)],
    "hidden": [(512, 384), (384, 512), (333, 250), (1280, 720), (41, 29), (900, 60), (2, 3), (256, 256), (199, 301),
               (60, 900)],
}
# (image index, out width, out height) in the C30 order: downscale, big downscale, extreme aspect, upscale, tiny,
# identity, near-identity, upscale to a square, anisotropic, extreme aspect
RESIZE_PAIRS = {
    "visible": [(0, 224, 168), (3, 455, 256), (5, 300, 15), (4, 224, 320), (6, 3, 5), (7, 224, 224), (8, 256, 255),
                (2, 640, 640), (1, 200, 700), (9, 7, 1000)],
    "hidden": [(0, 256, 192), (3, 427, 240), (5, 300, 20), (4, 256, 180), (6, 5, 7), (7, 256, 256), (8, 200, 300),
               (2, 512, 512), (1, 150, 600), (9, 11, 1000)],
}
# C31 reference models: slug, HF repo, task
REFERENCE_MODELS = [("resnet-50", "microsoft/resnet-50", "ImageClassification"),
                    ("rtdetr_r18vd", "PekingU/rtdetr_r18vd", "ObjectDetection")]
REF_POOLS = {"visible": ("cls_visible", "det_visible"), "hidden": ("cls_hidden", "det_hidden")}
REF_COUNTS = (20, 10)
CLS_MARGIN_MIN = 0.05
CLS_ORT_TOL = 1e-3
DET_ORT_IOU_MIN = 0.99      # C31 (amended 2026-10-03): detection-level ORT-vs-torch match
DET_ORT_SCORE_TOL = 1e-3
DET_SCORE_BAND = (0.48, 0.52)
DET_PAIR_IOU_BAND = (0.65, 0.75)
PREPROCESS_IMAGES = ["s00", "s01", "s02", "s03", "s04", "s05", "s07", "s08"]
RTDETR_IMAGES = ["s00", "s02", "s03", "s05"]

# ----------------------------------------------------------------------------------------------------------------
# Designed detection models.
#
# Every query: box (cx, cy, w, h) normalized, and the explicit class probabilities p {class index: probability}.
# Sigmoid models: logit = ln(p / (1 - p)) for the listed classes, every other logit = ln(0.005 / 0.995).
# Softmax models: the query's top listed class gets its p, every other class FLOOR, the rest goes to the no-object
# column; logits = ln(probability) (a softmax of log-probabilities gives the probabilities back).
# roles: query indices of the C30 situations.
#
# VISIBLE sigmoid (labels a b c, Q 8). Boxes as [x0 y0 x1 y1]:
#  q0 a .92 [.20 .25 .60 .65]  the best box of its class
#  q1 a .85 [.24 .25 .64 .65]  same class, IoU with q0 = .36*.4/(.32-.144) = .8182 (suppressed at nms .7 and .5)
#  q2 a .78 [.30 .25 .70 .65]  same class, IoU with q0 = .30/.50 = .6 (suppressed at .5, kept at .7)
#  q3 b .88 [.80 .38 1.04 .62] reaches outside [0,1] (x1 clamped to 1.0)
#  q4 c .70 [.812 .38 1.052 .62] different class, IoU with q3 = .228/.252 = .9048 (both kept at every nms)
#  q5 a .62 + c .56 [.05 .70 .25 .90] two classes above the threshold in one query (two detections, same box)
#  q6 c .40 (a .20) [.40 .75 .60 .95] mid score: appears only below threshold .4; a .20 is below every threshold but .05
#  q7 b .08 [.20 .05 .40 .25] appears only at threshold .05
# With threshold .05 there are 10 candidates for 8 queries: sigmoid top-Q cut drops the two lowest (.20, .08).
# VISIBLE softmax: same boxes; probabilities (a, b, c | no-object): q0 [.92 .02 .02 | .04], q1 a .85, q2 a .78, q3 b .88,
#  q4 c .70, q5 a .62, q6 c .40, q7 b .08 (others .02); one detection per query at most.
# HIDDEN sigmoid (labels w x y z, Q 10): q0 w .94, q1 w .86 (IoU .818 with q0), q2 w .74 (IoU .6), q3 x .91,
#  q4 y .72 (IoU .897 with q3), q5 z .67 (box reaches x1 1.08, y0 -.04), q6 x .63 + z .55, q7 y .41, q8 w .22, q9 x .12.
# ----------------------------------------------------------------------------------------------------------------
DESIGNS = {
    "visible": {
        "labels": ["a", "b", "c"],
        "floor": 0.02,
        "queries": [
            {"box": (0.40, 0.45, 0.40, 0.40), "p": {0: 0.92}},
            {"box": (0.44, 0.45, 0.40, 0.40), "p": {0: 0.85}},
            {"box": (0.50, 0.45, 0.40, 0.40), "p": {0: 0.78}},
            {"box": (0.92, 0.50, 0.24, 0.24), "p": {1: 0.88}},
            {"box": (0.932, 0.50, 0.24, 0.24), "p": {2: 0.70}},
            {"box": (0.15, 0.80, 0.20, 0.20), "p": {0: 0.62, 2: 0.56}},
            {"box": (0.50, 0.85, 0.20, 0.20), "p": {2: 0.40, 0: 0.20}},
            {"box": (0.30, 0.15, 0.20, 0.20), "p": {1: 0.08}},
        ],
        "roles": {"best": 0, "iou08": 1, "iou06": 2, "diff": (3, 4), "clamp": 3, "two_class": 5},
    },
    "hidden": {
        "labels": ["w", "x", "y", "z"],
        "floor": 0.01,
        "queries": [
            {"box": (0.35, 0.40, 0.30, 0.50), "p": {0: 0.94}},
            {"box": (0.38, 0.40, 0.30, 0.50), "p": {0: 0.86}},
            {"box": (0.425, 0.40, 0.30, 0.50), "p": {0: 0.74}},
            {"box": (0.70, 0.70, 0.24, 0.20), "p": {1: 0.91}},
            {"box": (0.713, 0.70, 0.24, 0.20), "p": {2: 0.72}},
            {"box": (0.93, 0.08, 0.30, 0.24), "p": {3: 0.67}},
            {"box": (0.12, 0.85, 0.20, 0.20), "p": {1: 0.63, 3: 0.55}},
            {"box": (0.50, 0.90, 0.20, 0.12), "p": {2: 0.41}},
            {"box": (0.85, 0.45, 0.10, 0.10), "p": {0: 0.22}},
            {"box": (0.50, 0.50, 0.10, 0.10), "p": {1: 0.12}},
        ],
        "roles": {"best": 0, "iou08": 1, "iou06": 2, "diff": (3, 4), "clamp": 5, "two_class": 6},
    },
}
DESIGNED_MODELS = [
    # slug, activation, config model_type, architecture, processor class
    ("tiny_det_designed_sigmoid", "Sigmoid", "rt_detr", "RTDetrForObjectDetection", "RTDetrImageProcessor"),
    ("tiny_det_designed_softmax", "Softmax", "detr", "DetrForObjectDetection", "DetrImageProcessor"),
]


# ----------------------------------------------------------------------------------------------------------------
# Pure helpers (unit tested)
# ----------------------------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def softmax64(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def sigmoid64(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def f32(v):
    """float32 value as a Python float."""
    return float(np.float32(v))


def box_iou(a, b):
    """C26 BoxIoU in float64. Boxes (x0, y0, x1, y1); width/height = max(0, x1 - x0); union <= 0 -> 0."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms(boxes, scores, classes, iou_threshold, max_keep):
    """C26 greedy class-wise NMS. Visit by score descending (ties: lower input index; NaN last). A box is kept unless a
    kept box of the same class has IoU > threshold. Threshold >= 1 or NaN: nothing is suppressed. Returns kept input
    indices in visit order, at most max_keep (<= 0 -> empty)."""
    n = len(boxes)
    if max_keep <= 0:
        return []
    off = iou_threshold != iou_threshold or iou_threshold >= 1.0

    def key(i):
        s = float(scores[i])
        return (1, 0.0, i) if s != s else (0, -s, i)

    kept = []
    for i in sorted(range(n), key=key):
        if len(kept) >= max_keep:
            break
        if not off and any(classes[k] == classes[i] and box_iou(boxes[i], boxes[k]) > iou_threshold for k in kept):
            continue
        kept.append(i)
    return kept


def cxcywh_to_xyxy(box):
    """C29 rule 2 in float32: x0 = cx - 0.5 w, ..."""
    cx, cy, w, h = (np.float32(v) for v in box)
    half = np.float32(0.5)
    return (cx - half * w, cy - half * h, cx + half * w, cy + half * h)


def map_to_source(xyxy, geom):
    """C29 rule 2: x_src = (CropLeft + x * CropWidth) / ResizedWidth, y likewise (float32). geom None = identity."""
    if geom is None:
        return tuple(xyxy)
    left, top = np.float32(geom["crop_left"]), np.float32(geom["crop_top"])
    cw, ch = np.float32(geom["crop_width"]), np.float32(geom["crop_height"])
    rw, rh = np.float32(geom["resized_width"]), np.float32(geom["resized_height"])
    x0, y0, x1, y1 = xyxy
    return ((left + x0 * cw) / rw, (top + y0 * ch) / rh, (left + x1 * cw) / rw, (top + y1 * ch) / rh)


def decode_candidates(logits, boxes, activation, score_threshold, geom=None):
    """C29 rule 1 + 2 (before NMS). logits [Q, C'], boxes [Q, 4] (cx cy w h). Returns a list of candidates in the
    sorted order, each (query, class, float32 score, unclamped mapped xyxy)."""
    logits = np.asarray(logits, dtype=np.float32)
    q_count = logits.shape[0]
    thr = np.float32(score_threshold)
    cands = []
    if activation == "Sigmoid":
        s = sigmoid64(logits.astype(np.float64)).astype(np.float32)
        flat = [(int(qi) * logits.shape[1] + int(c), int(qi), int(c), s[qi, c])
                for qi in range(q_count) for c in range(logits.shape[1]) if s[qi, c] > thr]
        flat.sort(key=lambda t: (-float(t[3]), t[0]))
        cands = [(qi, c, sc) for _, qi, c, sc in flat[:q_count]]
    elif activation == "Softmax":
        for qi in range(q_count):
            p = softmax64(logits[qi].astype(np.float64))[:-1]
            c = int(np.argmax(p))
            sc = np.float32(p[c])
            if sc > thr:
                cands.append((qi, c, sc))
        cands.sort(key=lambda t: (-float(t[2]), t[0]))
    else:
        raise ValueError("unknown activation %r" % activation)
    return [(qi, c, sc, map_to_source(cxcywh_to_xyxy(boxes[qi]), geom)) for qi, c, sc in cands]


def decode_reference(logits, boxes, labels, activation, score_threshold, nms_iou, max_det, geom=None):
    """C29 rules 1-4 + C26 NMS. Returns the detections list as in the goldens JSON."""
    logits = np.asarray(logits, dtype=np.float32)
    ncls = logits.shape[1] - (1 if activation == "Softmax" else 0)
    if len(labels) != ncls:
        raise ValueError("labels (%d) do not match the logit classes (%d)" % (len(labels), ncls))
    cands = decode_candidates(logits, boxes, activation, score_threshold, geom)
    keep = nms([c[3] for c in cands], [c[2] for c in cands], [c[1] for c in cands], nms_iou, max_det)
    out = []
    for i in keep:
        qi, c, sc, xyxy = cands[i]
        clamped = [f32(min(max(float(v), 0.0), 1.0)) for v in xyxy]
        out.append({"label": labels[c], "index": c, "score": f32(sc), "box": clamped})
    return out


def design_logits(design, activation):
    """(Q, C') float32 logits of a designed model."""
    labels, floor = design["labels"], design["floor"]
    ncls = len(labels)
    rows = []
    for q in design["queries"]:
        if activation == "Sigmoid":
            row = [math.log(0.005 / 0.995)] * ncls
            for c, p in q["p"].items():
                row[c] = math.log(p / (1.0 - p))
        else:
            top = max(q["p"], key=lambda c: q["p"][c])
            probs = [floor] * ncls
            probs[top] = q["p"][top]
            none = 1.0 - sum(probs)
            if none < 0.005:
                raise ValueError("no-object mass too small in a softmax design row")
            row = [math.log(p) for p in probs] + [math.log(none)]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def design_boxes(design):
    return np.asarray([q["box"] for q in design["queries"]], dtype=np.float32)


def check_design(design, activation):
    """Asserts the C30 required situations and the 0.02 margins; returns the realized IoUs and the facts."""
    logits = design_logits(design, activation)
    boxes = design_boxes(design)
    xy = [tuple(float(v) for v in cxcywh_to_xyxy(b)) for b in boxes]
    roles = design["roles"]
    r = {"iou_best_vs_08": box_iou(xy[roles["best"]], xy[roles["iou08"]]),
         "iou_best_vs_06": box_iou(xy[roles["best"]], xy[roles["iou06"]]),
         "iou_diff_class": box_iou(xy[roles["diff"][0]], xy[roles["diff"][1]])}
    assert 0.77 <= r["iou_best_vs_08"] <= 0.83, r
    assert 0.57 <= r["iou_best_vs_06"] <= 0.63, r
    assert 0.87 <= r["iou_diff_class"] <= 0.93, r
    cl = xy[roles["clamp"]]
    assert min(cl) < 0.0 or max(cl) > 1.0, "clamp query inside [0,1]"
    # classes of the pair
    cls_of = lambda qi: max(design["queries"][qi]["p"], key=lambda c: design["queries"][qi]["p"][c])
    assert cls_of(roles["best"]) == cls_of(roles["iou08"]) == cls_of(roles["iou06"])
    assert cls_of(roles["diff"][0]) != cls_of(roles["diff"][1])
    if activation == "Sigmoid":
        assert sum(1 for p in design["queries"][roles["two_class"]]["p"].values() if p > 0.5) >= 2
    # score margins: every class score against 0.5, 0.3, 0.05 (no-object column of a softmax model excluded)
    thresholds = sorted({t for t, _, _ in DET_PARAMS})
    scores = sigmoid64(logits) if activation == "Sigmoid" else np.asarray([softmax64(r_)[:-1] for r_ in logits])
    worst_score = min(abs(float(s) - t) for s in scores.ravel() for t in thresholds)
    assert worst_score >= MARGIN, "score within %.4f of a threshold" % worst_score
    # same-class IoU margins against every NMS threshold in use (pairs of explicitly designed entries)
    nms_ts = sorted({n for _, n, _ in DET_PARAMS if n < 1.0})
    worst_iou = 1.0
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            for c in set(design["queries"][i]["p"]) & set(design["queries"][j]["p"]):
                v = box_iou(xy[i], xy[j])
                worst_iou = min([worst_iou] + [abs(v - t) for t in nms_ts])
                assert all(abs(v - t) >= MARGIN for t in nms_ts), "IoU %.4f of q%d/q%d too close to an NMS threshold" % (v, i, j)
    r["min_score_margin"] = worst_score
    r["min_nms_margin"] = worst_iou
    return r


# ----------------------------------------------------------------------------------------------------------------
# C27 reference (independent of transformers)
# ----------------------------------------------------------------------------------------------------------------

def c27_geometry(width, height, s):
    """(resized (w, h), crop (left, top, w, h)) from the settings dict `s` (JSON names) for a width x height image."""
    mode = s["resize_mode"]
    if mode == "Stretch":
        rw, rh = s["image_width"], s["image_height"]
    elif mode == "ShortestEdgeCenterCrop":
        e = s["resize_shortest_edge"]
        if width <= height:
            rw, rh = e, int(float(e * height) / float(width))
        else:
            rh, rw = e, int(float(e * width) / float(height))
    elif mode == "StretchCenterCrop":
        rw, rh = s["resize_width"], s["resize_height"]
    else:
        raise ValueError("unknown resize_mode " + mode)
    if mode == "Stretch":
        return (rw, rh), (0, 0, rw, rh)
    cw, ch = s["image_width"], s["image_height"]
    if rw < cw or rh < ch:
        raise ValueError("resize %dx%d smaller than the crop %dx%d" % (rw, rh, cw, ch))
    return (rw, rh), ((rw - cw) // 2, (rh - ch) // 2, cw, ch)


def c27_preprocess(arr, s):
    """arr uint8 HxWx3. Returns (cropped uint8 HxWx3, float32 tensor [1, 3, H, W], geometry)."""
    from PIL import Image
    h, w = arr.shape[:2]
    (rw, rh), (left, top, cw, ch) = c27_geometry(w, h, s)
    img = Image.fromarray(arr).resize((rw, rh), resample=int(FILTER_BY_NAME[s["resample"]]))
    res = np.array(img)
    cropped = res[top:top + ch, left:left + cw, :]
    scale = np.float64(np.float32(s["rescale_factor"]))
    v = (cropped.astype(np.float64) * scale).astype(np.float32)
    mean = np.asarray(s["image_mean"], dtype=np.float32)
    std = np.asarray(s["image_std"], dtype=np.float32)
    out = (v - mean) / std
    chw = np.ascontiguousarray(out.transpose(2, 0, 1))[None]
    geom = {"source_width": w, "source_height": h, "resized_width": rw, "resized_height": rh, "crop_left": left,
            "crop_top": top, "crop_width": cw, "crop_height": ch}
    return np.ascontiguousarray(cropped), chw.astype(np.float32), geom


FILTER_BY_NAME = {n: i for n, i in FILTERS}


# ----------------------------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------------------------

def make_image(width, height, rng):
    """Deterministic RGB uint8 HxWx3: linear gradients, a seeded noise band, hard-edged coloured rectangles, a pure
    black rectangle next to a pure white one (so resize overshoot clips), and a 1-px black line."""
    yy, xx = np.mgrid[0:height, 0:width]
    u = (xx + 0.5) / width
    v = (yy + 0.5) / height
    coef = rng.uniform(-1.0, 1.0, size=(3, 3))
    base = rng.uniform(0.2, 0.8, size=3)
    img = np.empty((height, width, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = base[c] + 0.5 * coef[c, 0] * u + 0.5 * coef[c, 1] * v + 0.25 * coef[c, 2] * u * v
    img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)

    def span(frac0, frac1, size):
        a = min(int(frac0 * size), size - 1)
        b = max(min(int(frac1 * size), size), a + 1)
        return a, b

    # noise band (bottom-left)
    x0, x1 = span(0.0, 0.33, width)
    y0, y1 = span(0.5, 1.0, height)
    img[y0:y1, x0:x1, :] = rng.randint(0, 256, size=(y1 - y0, x1 - x0, 3)).astype(np.uint8)
    # three coloured rectangles
    for _ in range(3):
        fx, fy = rng.uniform(0.0, 0.7), rng.uniform(0.0, 0.7)
        fw, fh = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)
        col = rng.randint(0, 256, size=3).astype(np.uint8)
        xa, xb = span(fx, fx + fw, width)
        ya, yb = span(fy, fy + fh, height)
        img[ya:yb, xa:xb, :] = col
    # pure black next to pure white
    ya, yb = span(0.10, 0.35, height)
    xa, xb = span(0.55, 0.70, width)
    xc, xd = span(0.70, 0.85, width)
    img[ya:yb, xa:xb, :] = 0
    img[ya:yb, xc:xd, :] = 255
    # 1-px black line
    img[:, min(int(0.30 * width), width - 1), :] = 0
    return img


def probe_points(width, height, rng):
    pts = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1), (width // 2, height // 2)]
    for _ in range(11):
        pts.append((int(rng.randint(0, width)), int(rng.randint(0, height))))
    return pts


# ----------------------------------------------------------------------------------------------------------------
# Processors and settings
# ----------------------------------------------------------------------------------------------------------------

IMAGENET_DEFAULT = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
INV255 = 0.00392156862745098


def _cfg(cls, size, resample, mean, std, crop=None, **extra):
    d = {"image_processor_type": cls, "do_resize": True, "size": size, "resample": resample, "do_rescale": True,
         "rescale_factor": INV255, "do_normalize": True, "image_mean": mean, "image_std": std}
    if crop is not None:
        d["do_center_crop"] = True
        d["crop_size"] = crop
    d.update(extra)
    return d


def processor_configs(source_models):
    """The 7 C30 configs, in table order. convnext_crop and rtdetr_stretch are the exact reference files."""
    ref = os.path.join(source_models, "reference")
    out = [
        ("convnext_crop", read_json(os.path.join(ref, "resnet-50", "preprocessor_config.json")), PREPROCESS_IMAGES),
        ("rtdetr_stretch", read_json(os.path.join(ref, "rtdetr_r18vd", "preprocessor_config.json")), RTDETR_IMAGES),
        ("deit_crop", _cfg("DeiTImageProcessor", {"height": 256, "width": 256}, 3, *IMAGENET_DEFAULT, crop=224),
         PREPROCESS_IMAGES),
        ("deit_rect_crop", _cfg("DeiTImageProcessor", {"height": 240, "width": 260}, 2, *IMAGENET_DEFAULT,
                                crop={"height": 224, "width": 200}), PREPROCESS_IMAGES),
        ("clip_crop", _cfg("CLIPImageProcessor", {"shortest_edge": 224}, 3, CLIP_MEAN, CLIP_STD,
                           crop={"height": 224, "width": 224}, do_convert_rgb=True), PREPROCESS_IMAGES),
        ("vit_stretch", _cfg("ViTImageProcessor", {"height": 224, "width": 224}, 2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
         PREPROCESS_IMAGES),
        ("levit_crop", _cfg("LevitImageProcessor", {"shortest_edge": 224}, 3, *IMAGENET_DEFAULT,
                            crop={"height": 224, "width": 224}), PREPROCESS_IMAGES),
    ]
    return out


def processor_class(cfg):
    import transformers
    name = mit.processor_class_name(cfg)
    cls = getattr(transformers, name, None)
    if cls is None:
        raise SystemExit("FAIL: processor class %r not found in transformers" % name)
    return name, cls


def resample_name(cfg, cls):
    r = cfg.get("resample")
    if mit._is_int(r) and 0 <= r <= 5:
        return RESAMPLE_NAMES[r]
    p = inspect.signature(cls.__init__).parameters.get("resample")
    if p is None or p.default is inspect.Parameter.empty or p.default is None:
        return "Bilinear"
    return RESAMPLE_NAMES[int(p.default)]


def make_settings(cfg, cls_name, cls):
    """C14/C25 rules via make_import_test_data.image_preprocess, then the C32 LeViT rule and the resample name."""
    s = mit.image_preprocess(cfg)
    if cls_name == "LevitImageProcessor" and s["resize_mode"] == "ShortestEdgeCenterCrop":
        s["resize_shortest_edge"] = int((256 / 224) * s["resize_shortest_edge"])
    order = ["image_width", "image_height", "resize_mode", "resize_shortest_edge", "resize_width", "resize_height"]
    out = {k: s[k] for k in order}
    out["resample"] = resample_name(cfg, cls)
    out["rescale_factor"] = s["rescale_factor"]
    out["image_mean"] = list(s["image_mean"])
    out["image_std"] = list(s["image_std"])
    return out


class Preprocessor:
    """One HF slow processor + its settings; process() returns the HF tensor and cropped image and cross-checks C27."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cls_name, self.cls = processor_class(cfg)
        self.proc = self.cls.from_dict(dict(cfg))
        self.settings = make_settings(cfg, self.cls_name, self.cls)

    def process(self, arr, label):
        kw = dict(return_tensors="np", input_data_format="channels_last")
        hf = np.asarray(self.proc(images=arr, **kw)["pixel_values"])
        hf_crop = np.asarray(self.proc(images=arr, do_rescale=False, do_normalize=False, **kw)["pixel_values"])
        if hf.ndim != 4 or hf.shape[0] != 1 or hf.shape[1] != 3:
            raise SystemExit("FAIL: %s: unexpected HF output shape %s" % (label, hf.shape))
        hf = hf.astype(np.float32)
        hf_crop = hf_crop[0].transpose(1, 2, 0)
        if not (np.all(hf_crop == np.rint(hf_crop)) and hf_crop.min() >= 0 and hf_crop.max() <= 255):
            raise SystemExit("FAIL: %s: HF cropped image is not 8-bit" % label)
        hf_crop = hf_crop.astype(np.uint8)
        cropped, mine, geom = c27_preprocess(arr, self.settings)
        h, w = arr.shape[:2]
        if hf_crop.shape != cropped.shape:
            raise SystemExit("NEEDS: HF and C27 disagree on the crop size for %s: HF %s, C27 resized %s crop %s "
                             "(image %dx%d)" % (label, hf_crop.shape[:2][::-1], (geom["resized_width"], geom["resized_height"]),
                                                cropped.shape[:2][::-1], w, h))
        if not np.array_equal(hf_crop, cropped):
            bad = int(np.count_nonzero(hf_crop != cropped))
            raise SystemExit("NEEDS: HF and C27 disagree on the cropped pixels for %s: %d bytes differ, C27 resized "
                             "%dx%d crop %s" % (label, bad, geom["resized_width"], geom["resized_height"],
                                                (geom["crop_left"], geom["crop_top"], geom["crop_width"], geom["crop_height"])))
        diff = float(np.max(np.abs(hf.astype(np.float64) - mine.astype(np.float64))))
        if diff > CROSSCHECK_TOL:
            raise SystemExit("NEEDS: HF tensor and C27 differ by %.3e (> %g) for %s" % (diff, CROSSCHECK_TOL, label))
        return hf, hf_crop, geom, diff


# ----------------------------------------------------------------------------------------------------------------
# Designed ONNX models
# ----------------------------------------------------------------------------------------------------------------

def build_designed_onnx(path, logits, boxes):
    """pixel_values float32 [batch, 3, 32, 32] -> logits [batch, Q, C'] and pred_boxes [batch, Q, 4], designed constants
    broadcast over the (dynamic) batch with Expand(constant, Concat(Shape(pixel_values)[0:1], [Q, C']))."""
    onnx, T, h, nh = mit._onnx()
    q, c = logits.shape
    inits = [nh.from_array(logits.reshape(1, q, c).astype(np.float32), "const_logits"),
             nh.from_array(boxes.reshape(1, q, 4).astype(np.float32), "const_boxes"),
             nh.from_array(np.array([0], dtype=np.int64), "starts"), nh.from_array(np.array([1], dtype=np.int64), "ends"),
             nh.from_array(np.array([q, c], dtype=np.int64), "tail_l"), nh.from_array(np.array([q, 4], dtype=np.int64), "tail_b")]
    nodes = [
        h.make_node("Shape", ["pixel_values"], ["shp"]),
        h.make_node("Slice", ["shp", "starts", "ends"], ["batch1"]),
        h.make_node("Concat", ["batch1", "tail_l"], ["target_l"], axis=0),
        h.make_node("Concat", ["batch1", "tail_b"], ["target_b"], axis=0),
        h.make_node("Expand", ["const_logits", "target_l"], ["logits"]),
        h.make_node("Expand", ["const_boxes", "target_b"], ["pred_boxes"]),
    ]
    g = h.make_graph(nodes, "tiny_det_designed", [mit._vi("pixel_values", T.FLOAT, ["batch", 3, 32, 32])],
                     [mit._vi("logits", T.FLOAT, ["batch", q, c]), mit._vi("pred_boxes", T.FLOAT, ["batch", q, 4])],
                     initializer=inits)
    m = mit._finish(g, path)
    onnx.checker.check_model(m)


def write_designed_folder(folder, slug, activation, model_type, arch, proc_cls, design):
    os.makedirs(folder, exist_ok=True)
    build_designed_onnx(os.path.join(folder, "model.onnx"), design_logits(design, activation), design_boxes(design))
    labels = design["labels"]
    files = {
        "config.json": {"architectures": [arch], "model_type": model_type, "id2label": mit.id2label(labels)},
        "preprocessor_config.json": {"image_processor_type": proc_cls, "size": {"height": 32, "width": 32},
                                     "do_resize": True, "do_rescale": True, "rescale_factor": INV255,
                                     "do_normalize": False, "do_pad": False},
        "modelpipelines.json": mit.full_license(slug),
    }
    for name, obj in files.items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(mit.jbytes(obj))


# ----------------------------------------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------------------------------------

def write_f32(path, arr):
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def labels_from_config(config):
    items = sorted(((int(k), v) for k, v in config["id2label"].items()))
    return [v for _, v in items]


def margin_skip_reason(logits, boxes, activation, thr, nms_iou, geom):
    """C30 skip rule for tiny_object_detection: a score within MARGIN of the threshold, or a same-class pair of
    candidates with IoU within MARGIN of the NMS threshold. Returns a reason string or None."""
    logits = np.asarray(logits, dtype=np.float64)
    if activation == "Sigmoid":
        scores = sigmoid64(logits).ravel()
    else:
        scores = np.concatenate([softmax64(r)[:-1] for r in logits])
    near = [float(s) for s in scores if abs(float(s) - thr) < MARGIN]
    if near:
        return "score %.6f within %.2f of threshold %.2f" % (near[0], MARGIN, thr)
    if nms_iou < 1.0:
        cands = decode_candidates(logits.astype(np.float32), boxes, activation, thr, geom)
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                if cands[i][1] == cands[j][1]:
                    v = box_iou(cands[i][3], cands[j][3])
                    if abs(v - nms_iou) < MARGIN:
                        return "same-class IoU %.6f within %.2f of nms %.2f" % (v, MARGIN, nms_iou)
    return None


# ----------------------------------------------------------------------------------------------------------------
# Part B: reference images (C31)
# ----------------------------------------------------------------------------------------------------------------

def load_test_images(path):
    doc = read_json(path)
    if doc.get("schema") != "pipelines-test-images/1":
        raise SystemExit("FAIL: %s is not schema pipelines-test-images/1" % path)
    return doc


def det_selection_reason(logits, boxes):
    """C31 detection selection rules on the torch outputs (logits [Q, C], boxes [Q, 4]); None = selected."""
    scores = sigmoid64(np.asarray(logits, dtype=np.float64))
    lo, hi = DET_SCORE_BAND
    near = scores[(scores >= lo) & (scores <= hi)]
    if near.size:
        return "torch score %.4f in [%.2f, %.2f]" % (float(near[0]), lo, hi)
    cands = decode_candidates(np.asarray(logits, dtype=np.float32), boxes, "Sigmoid", 0.5, None)
    lo, hi = DET_PAIR_IOU_BAND
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if cands[i][1] == cands[j][1]:
                v = box_iou(cands[i][3], cands[j][3])
                if lo <= v <= hi:
                    return "same-class IoU %.4f in [%.2f, %.2f]" % (v, lo, hi)
    return None


def hf_detection_list(proc, outputs, labels):
    """post_process_object_detection(threshold=0.5), no target sizes: normalized xyxy, unclamped."""
    res = proc.post_process_object_detection(outputs, threshold=0.5)[0]
    out = []
    for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
        c = int(lb)
        out.append({"label": labels[c], "index": c, "score": f32(float(sc)), "box": [f32(float(v)) for v in bx]})
    return out


def run_reference(variant, out_dir, models_root, ref_png_dir, test_images_path, report):
    """Returns (reference_images, reference_models, reference_generator)."""
    import onnxruntime as ort
    import torch
    import transformers
    from PIL import Image
    torch.set_num_threads(1)
    cache = os.path.join(os.path.dirname(os.path.abspath(models_root)), "cache", "hf")
    manifest = read_json(os.path.join(models_root, "reference", "manifest.json"))
    pools = load_test_images(test_images_path)["pools"]
    cls_pool, det_pool = REF_POOLS[variant]
    meta = {}
    for e in pools[cls_pool] + pools[det_pool]:
        meta[e["id"]] = e
    selected_images = []  # ids in first-use order
    models_out = []
    for (slug, repo, task), want, pool_name in zip(REFERENCE_MODELS, REF_COUNTS, (cls_pool, det_pool)):
        folder = os.path.join(models_root, "reference", slug)
        config = read_json(os.path.join(folder, "config.json"))
        pre = read_json(os.path.join(folder, "preprocessor_config.json"))
        labels = labels_from_config(config)
        activation = mit.score_activation(task, config)
        pp = Preprocessor(pre)
        entry_m = manifest[slug]
        cls = transformers.AutoModelForImageClassification if task == "ImageClassification" \
            else transformers.AutoModelForObjectDetection
        model = cls.from_pretrained(repo, revision=entry_m["revision"], cache_dir=cache, local_files_only=True,
                                    dtype=torch.float32).eval()
        onnx_path = os.path.join(folder, "model.onnx")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        entry = {"slug": slug, "folder": "reference/" + slug, "task": task, "reference": "torch",
                 "repo": repo, "revision": entry_m["revision"], "license": entry_m["license"],
                 "onnx_sha256": sha256_file(onnx_path),
                 "package": {"settings": pp.settings, "labels": labels, "score_activation": activation,
                             "nms_iou_threshold": 0.7, "max_detections": 100}, "cases": [], "skipped": []}
        if task == "ObjectDetection":
            entry["score_threshold"] = 0.5
        worst = {"ort_logits": 0.0, "ort_boxes": 0.0, "min_margin": 1.0, "boxes": 0, "classes": {}, "min_score_gap": 1.0}
        for e in pools[pool_name]:
            if len(entry["cases"]) >= want:
                break
            sid = e["id"]
            png = os.path.join(ref_png_dir, sid + ".png")
            if not os.path.isfile(png):
                raise SystemExit("NEEDS: %s missing; run Tools/fetch_test_images.py" % png)
            arr = np.array(Image.open(png).convert("RGB"))
            hf, _, _, _ = pp.process(arr, "%s/%s" % (slug, sid))
            with torch.no_grad():
                out = model(pixel_values=torch.from_numpy(hf))
            ort_out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"pixel_values": hf})))
            lg = out.logits.detach().numpy().astype(np.float32)
            if task == "ImageClassification":
                logits = lg[0]
                scores = softmax64(logits)
                order = np.argsort(-scores, kind="stable")
                idx = int(order[0])
                margin = float(scores[order[0]] - scores[order[1]])
                diff = float(np.max(np.abs(ort_out["logits"][0].astype(np.float64) - logits.astype(np.float64))))
                ort_idx = int(np.argmax(ort_out["logits"][0]))
                why = None
                if diff > CLS_ORT_TOL:
                    why = "ORT logits differ from torch by %.2e > %g" % (diff, CLS_ORT_TOL)
                elif margin < CLS_MARGIN_MIN:
                    why = "top_margin %.4f < %.2f" % (margin, CLS_MARGIN_MIN)
                elif ort_idx != idx:
                    why = "ORT argmax %d != torch argmax %d" % (ort_idx, idx)
                if why:
                    entry["skipped"].append({"image": sid, "reason": why})
                    continue
                worst["ort_logits"] = max(worst["ort_logits"], diff)
                worst["min_margin"] = min(worst["min_margin"], margin)
                entry["cases"].append({"image": sid, "logits": [f32(v) for v in logits], "scores": [float(v) for v in scores],
                                       "index": idx, "label": labels[idx], "top_margin": margin, "ort_max_abs_diff": diff,
                                       "ort_index": ort_idx})
            else:
                bx = out.pred_boxes.detach().numpy().astype(np.float32)
                d_l = float(np.max(np.abs(ort_out["logits"].astype(np.float64) - lg.astype(np.float64))))
                d_b = float(np.max(np.abs(ort_out["pred_boxes"].astype(np.float64) - bx.astype(np.float64))))
                dets = decode_reference(lg[0], bx[0], labels, activation, 0.5, 0.7, 100, None)
                if not dets:
                    why = "no detection above 0.5"
                else:
                    why = det_selection_reason(lg[0], bx[0])
                if why:
                    entry["skipped"].append({"image": sid, "reason": why})
                    continue
                # C31 detection-level ORT check: decode ORT exactly like torch; one-to-one match by index
                ort_dets = decode_reference(ort_out["logits"][0].astype(np.float32),
                                            ort_out["pred_boxes"][0].astype(np.float32), labels, activation,
                                            0.5, 0.7, 100, None)
                min_iou, max_sd = 1.0, 0.0
                if len(ort_dets) != len(dets):
                    raise SystemExit("ORT detections differ from torch on %s/%s: %d vs %d detections"
                                     % (slug, sid, len(ort_dets), len(dets)))
                for a in dets:
                    m = [b for b in ort_dets if b["index"] == a["index"]
                         and box_iou(a["box"], b["box"]) > DET_ORT_IOU_MIN and abs(a["score"] - b["score"]) <= DET_ORT_SCORE_TOL]
                    if len(m) != 1:
                        raise SystemExit("ORT detections differ from torch on %s/%s: no one-to-one match for %s"
                                         % (slug, sid, a))
                    min_iou = min(min_iou, box_iou(a["box"], m[0]["box"]))
                    max_sd = max(max_sd, abs(a["score"] - m[0]["score"]))
                worst["ort_logits"] = max(worst["ort_logits"], d_l)
                worst["ort_boxes"] = max(worst["ort_boxes"], d_b)
                worst["ort_det_min_iou"] = min(worst.get("ort_det_min_iou", 1.0), min_iou)
                worst["ort_det_max_score_diff"] = max(worst.get("ort_det_max_score_diff", 0.0), max_sd)
                worst["boxes"] += len(dets)
                for dd in dets:
                    worst["classes"][dd["label"]] = worst["classes"].get(dd["label"], 0) + 1
                sc_all = sigmoid64(lg[0].astype(np.float64))
                worst["min_score_gap"] = min(worst["min_score_gap"], float(np.min(np.abs(sc_all - 0.5))))
                entry["cases"].append({"image": sid, "detections": dets,
                                       "hf_detections": hf_detection_list(pp.proc, out, labels),
                                       "ort_max_abs_diff_logits": d_l, "ort_max_abs_diff_boxes": d_b,
                                       "ort_detection_min_iou": min_iou, "ort_detection_max_score_diff": max_sd})
        if len(entry["cases"]) < want:
            raise SystemExit("NEEDS: pool %s gave only %d of %d %s cases after the selection rules (skipped %d); add "
                             "candidates to Tools/data/test_images.json; skipped: %s" % (
                                 pool_name, len(entry["cases"]), want, slug, len(entry["skipped"]), entry["skipped"]))
        selected_images += [c["image"] for c in entry["cases"]]
        report["reference"][slug] = dict(worst, skipped=entry["skipped"], selected=[c["image"] for c in entry["cases"]])
        models_out.append(entry)
    # reference_images: the selected ones, PNGs copied under out_dir/reference_images when out_dir differs
    dst_dir = os.path.join(out_dir, "reference_images")
    os.makedirs(dst_dir, exist_ok=True)
    images = []
    for sid in selected_images:
        src = os.path.join(ref_png_dir, sid + ".png")
        dst = os.path.join(dst_dir, sid + ".png")
        if os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dst)):
            shutil.copyfile(src, dst)
        arr = np.array(Image.open(dst).convert("RGB"))
        e = meta[sid]
        images.append({"id": sid, "file": "reference_images/%s.png" % sid, "width": int(arr.shape[1]),
                       "height": int(arr.shape[0]), "rgb_sum": int(arr.astype(np.int64).sum()), "source_url": e["url"],
                       "license": e["license"], "author": e["author"], "title": e["title"]})
    gen = {"torch": torch.__version__, "script": SCRIPT, "pools": list(REF_POOLS[variant]),
           "test_images_sha256": sha256_file(test_images_path)}
    return images, models_out, gen


def run_variant(args):
    import onnxruntime as ort
    import PIL
    import transformers
    from PIL import Image
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()
    variant, seed = args.variant, args.seed
    out_dir = os.path.abspath(args.out_dir)
    models_out = os.path.abspath(args.models_out)
    source_models = os.path.abspath(args.source_models)
    for sub in ("images", "resize", "preprocess"):
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
        os.makedirs(os.path.join(out_dir, sub))
    report = {"skipped": [], "crosscheck": {}, "designs": {}, "reference": {}}

    # ---- images
    images = []
    arrays = {}
    for idx, (w, h) in enumerate(IMAGE_SIZES[variant]):
        sid = "s%02d" % idx
        arr = make_image(w, h, np.random.RandomState(seed * 100 + idx))
        rel = "images/%s.png" % sid
        Image.fromarray(arr).save(os.path.join(out_dir, rel), format="PNG")
        arr = np.array(Image.open(os.path.join(out_dir, rel)).convert("RGB"))  # the loaded PNG is the reference
        arrays[sid] = arr
        prng = np.random.RandomState(seed * 1000 + idx + 7)
        images.append({"id": sid, "file": rel, "width": w, "height": h, "rgb_sum": int(arr.astype(np.int64).sum()),
                       "probes": [{"x": x, "y": y, "rgb": [int(v) for v in arr[y, x]]}
                                  for x, y in probe_points(w, h, prng)]})

    # ---- resize cases
    resize_cases = []
    for fname, fi in FILTERS:
        for img_idx, ow, oh in RESIZE_PAIRS[variant]:
            sid = "s%02d" % img_idx
            cid = "r%03d" % len(resize_cases)
            src = Image.open(os.path.join(out_dir, "images", sid + ".png")).convert("RGB")
            res = src.resize((ow, oh), resample=fi)
            rel = "resize/%s.png" % cid
            res.save(os.path.join(out_dir, rel), format="PNG")
            resize_cases.append({"id": cid, "image": sid, "filter": fname, "pil_filter": fi, "width": ow, "height": oh,
                                 "expected": rel})

    # ---- preprocess cases
    pre_cases = []
    for cname, cfg, img_ids in processor_configs(source_models):
        pp = Preprocessor(cfg)
        worst = 0.0
        for sid in img_ids:
            cid = "p%03d" % len(pre_cases)
            hf, hf_crop, geom, diff = pp.process(arrays[sid], "%s/%s" % (cname, sid))
            worst = max(worst, diff)
            Image.fromarray(hf_crop).save(os.path.join(out_dir, "preprocess", cid + "_cropped.png"), format="PNG")
            write_f32(os.path.join(out_dir, "preprocess", cid + ".f32"), hf)
            pre_cases.append({
                "id": cid, "image": sid, "config": cname, "settings": pp.settings,
                "resized": [geom["resized_width"], geom["resized_height"]],
                "crop": [geom["crop_left"], geom["crop_top"], geom["crop_width"], geom["crop_height"]],
                "cropped": "preprocess/%s_cropped.png" % cid, "tensor": "preprocess/%s.f32" % cid,
                "tensor_shape": list(hf.shape), "tensor_sum": float(np.sum(hf, dtype=np.float64)),
                "tensor_abs_max": f32(np.max(np.abs(hf))), "crosscheck_max_abs_diff": diff})
        report["crosscheck"][cname] = worst

    # ---- designed models (visible must not depend on the hidden run: each writes under its own models_out)
    design = DESIGNS[variant]
    for slug, activation, model_type, arch, proc_cls in DESIGNED_MODELS:
        report["designs"][slug] = check_design(design, activation)
        folder = os.path.join(models_out, "vision_test", slug)
        shutil.rmtree(folder, ignore_errors=True)
        write_designed_folder(folder, slug, activation, model_type, arch, proc_cls, design)
    if os.path.normcase(models_out) != os.path.normcase(source_models):
        for slug in ("tiny_image_classification", "tiny_object_detection"):
            dst = os.path.join(models_out, "import_test", slug)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(source_models, "import_test", slug), dst)

    # ---- tiny models
    tiny = []
    specs = [("tiny_image_classification", "import_test/tiny_image_classification", "ImageClassification", TINY_IMAGES),
             ("tiny_object_detection", "import_test/tiny_object_detection", "ObjectDetection", TINY_IMAGES),
             ("tiny_det_designed_sigmoid", "vision_test/tiny_det_designed_sigmoid", "ObjectDetection", DESIGNED_IMAGES),
             ("tiny_det_designed_softmax", "vision_test/tiny_det_designed_softmax", "ObjectDetection", DESIGNED_IMAGES)]
    for slug, folder_rel, task, img_ids in specs:
        folder = os.path.join(models_out, folder_rel)
        config = read_json(os.path.join(folder, "config.json"))
        pre = read_json(os.path.join(folder, "preprocessor_config.json"))
        labels = labels_from_config(config)
        activation = mit.score_activation(task, config)
        pp = Preprocessor(pre)
        sess = ort.InferenceSession(os.path.join(folder, "model.onnx"), providers=["CPUExecutionProvider"])
        entry = {"slug": slug, "folder": folder_rel, "task": task,
                 "package": {"settings": pp.settings, "labels": labels, "score_activation": activation,
                             "nms_iou_threshold": 0.7, "max_detections": 100}, "cases": []}
        for sid in img_ids:
            hf, _, geom, diff = pp.process(arrays[sid], "%s/%s" % (slug, sid))
            report["crosscheck"][slug] = max(report["crosscheck"].get(slug, 0.0), diff)
            outs = sess.run(None, {"pixel_values": hf})
            names = [o.name for o in sess.get_outputs()]
            res = dict(zip(names, outs))
            if task == "ImageClassification":
                logits = np.asarray(res["logits"], dtype=np.float32)[0]
                scores = mit_activation(logits, activation)
                idx = int(np.argmax(scores))
                entry["cases"].append({"image": sid, "logits": [f32(v) for v in logits],
                                       "scores": [f32(v) for v in scores], "index": idx, "label": labels[idx]})
                continue
            lg = np.asarray(res["logits"], dtype=np.float32)
            bx = np.asarray(res["pred_boxes"], dtype=np.float32)
            for thr, nms_iou, max_det in DET_PARAMS:
                if slug == "tiny_object_detection":
                    why = margin_skip_reason(lg[0], bx[0], activation, thr, nms_iou, geom)
                    if why:
                        report["skipped"].append("%s %s (%.2f, %.2f, %d): %s" % (slug, sid, thr, nms_iou, max_det, why))
                        continue
                dets = decode_reference(lg[0], bx[0], labels, activation, thr, nms_iou, max_det, geom)
                entry["cases"].append({"image": sid, "score_threshold": thr, "nms_iou_threshold": nms_iou,
                                       "max_detections": max_det,
                                       "logits": [f32(v) for v in lg.ravel()], "logits_shape": list(lg.shape),
                                       "pred_boxes": [f32(v) for v in bx.ravel()], "pred_boxes_shape": list(bx.shape),
                                       "detections": dets})
        tiny.append(entry)

    ref_images, ref_models, ref_gen = run_reference(variant, out_dir, source_models, os.path.abspath(args.reference_images),
                                                     os.path.abspath(args.test_images), report)

    gen = {"script": SCRIPT, "variant": variant, "seed": seed, "python": platform.python_version(),
           "pillow": PIL.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
           "onnxruntime": ort.__version__}
    doc = {"schema": SCHEMA, "generator": gen, "root": out_dir.replace("\\", "/"),
           "models_root": models_out.replace("\\", "/"), "images": images, "resize_cases": resize_cases,
           "preprocess_cases": pre_cases, "tiny_models": tiny, "reference_images": ref_images,
           "reference_models": ref_models, "reference_generator": ref_gen}
    json_path = os.path.join(out_dir, "vision_goldens.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    return doc, json_path, report


def mit_activation(logits, mode):
    return softmax64(logits) if mode == "Softmax" else sigmoid64(logits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="Writes vision_goldens.json plus images/, resize/, preprocess/ under --out-dir "
                                        "and the designed model folders under <models-out>/vision_test/.")
    dev = _paths.dev_dir()
    ap.add_argument("--variant", choices=["visible", "hidden"], default="visible")
    ap.add_argument("--seed", type=int, default=None, help="image seed (default 4242 visible, 777 hidden)")
    ap.add_argument("--out-dir", default=os.path.join(dev, "goldens", "vision"))
    ap.add_argument("--models-out", default=os.path.join(dev, "models"),
                    help="where the designed models are written (hidden: the two importer models are copied here too)")
    ap.add_argument("--source-models", default=os.path.join(dev, "models"),
                    help="models root with import_test/ and reference/ (read only)")
    ap.add_argument("--reference-images", default=os.path.join(dev, "goldens", "vision", "reference_images"),
                    help="folder with the PNGs written by fetch_test_images.py")
    ap.add_argument("--test-images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_images.json"),
                    help="pinned pools (default Tools/data/test_images.json)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = DEFAULT_SEEDS[args.variant]
    doc, json_path, report = run_variant(args)
    print("images %d, resize cases %d, preprocess cases %d, tiny models %d (cases %s)" % (
        len(doc["images"]), len(doc["resize_cases"]), len(doc["preprocess_cases"]), len(doc["tiny_models"]),
        [len(m["cases"]) for m in doc["tiny_models"]]))
    for k, v in report["crosscheck"].items():
        print("crosscheck max abs diff %-26s %.3e" % (k, v))
    for slug, r in report["designs"].items():
        print("design %-26s IoU best/.8 %.4f  best/.6 %.4f  diff-class %.4f  min score margin %.4f  min nms margin %.4f" % (
            slug, r["iou_best_vs_08"], r["iou_best_vs_06"], r["iou_diff_class"], r["min_score_margin"], r["min_nms_margin"]))
    for slug, r in report["reference"].items():
        print("reference %-13s selected %d, skipped %d %s" % (slug, len(r["selected"]), len(r["skipped"]), r["skipped"]))
        if slug == "resnet-50":
            print("  min top margin %.4f, ORT max abs diff (logits) %.3e" % (r["min_margin"], r["ort_logits"]))
        else:
            print("  boxes total %d, classes %s, min |score-0.5| over all scores %.4f, ORT max diff logits %.3e boxes %.3e" % (
                r["boxes"], r["classes"], r["min_score_gap"], r["ort_logits"], r["ort_boxes"]))
    for s in report["skipped"]:
        print("SKIPPED " + s)
    if not report["skipped"]:
        print("no case skipped by the margin rule")
    print("wrote %s (%d bytes, sha256 %s)" % (json_path, os.path.getsize(json_path), sha256_file(json_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
