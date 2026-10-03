import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import _paths  # noqa: E402
import make_vision_goldens as g  # noqa: E402

DEV = _paths.dev_dir()
VIS_DIR = os.path.join(DEV, "goldens", "vision")
GOLDEN = os.path.join(VIS_DIR, "vision_goldens.json")
SCRIPT = os.path.join(TOOLS, "make_vision_goldens.py")
PY = sys.executable
F255 = 0.003921568859368563  # float(np.float32(1/255)), written by hand


# ---- pure helpers with hand-computed values ---------------------------------------------------------------------

def test_box_iou_hand_values():
    assert g.box_iou((0, 0, 1, 1), (0, 0, 1, 0.5)) == 0.5
    assert g.box_iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3, abs=1e-15)  # 2 / (4 + 4 - 2)
    assert g.box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert g.box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0  # union 0
    assert g.box_iou((1, 1, 0, 0), (0, 0, 1, 1)) == 0.0  # inverted box has zero area


def test_nms_iou_exactly_at_threshold_is_kept():
    boxes = [(0, 0, 1, 1), (0, 0, 1, 0.5)]  # IoU exactly 0.5
    assert g.nms(boxes, [0.9, 0.8], [0, 0], 0.5, 10) == [0, 1]
    assert g.nms(boxes, [0.9, 0.8], [0, 0], 0.49, 10) == [0]


def test_nms_is_class_wise_and_order_is_by_score():
    boxes = [(0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1)]
    # classes 0, 1, 0: the third (class 0) is suppressed by the first; the second (class 1) survives
    assert g.nms(boxes, [0.5, 0.9, 0.4], [0, 1, 0], 0.5, 10) == [1, 0]


def test_nms_ties_lower_index_max_keep_and_disabled():
    far = [(0, 0, 1, 1), (2, 0, 3, 1), (4, 0, 5, 1)]
    assert g.nms(far, [0.5, 0.5, 0.5], [0, 0, 0], 0.5, 10) == [0, 1, 2]
    assert g.nms(far, [0.5, 0.5, 0.5], [0, 0, 0], 0.5, 2) == [0, 1]
    assert g.nms(far, [0.5, 0.9, 0.9], [0, 0, 0], 0.5, 10) == [1, 2, 0]
    assert g.nms(far, [0.5, 0.5, 0.5], [0, 0, 0], 0.5, 0) == []
    same = [(0, 0, 1, 1)] * 3
    assert g.nms(same, [0.3, 0.2, 0.1], [0, 0, 0], 1.0, 10) == [0, 1, 2]  # >= 1 disables
    assert g.nms(same, [0.3, 0.2, 0.1], [0, 0, 0], float("nan"), 10) == [0, 1, 2]
    assert g.nms(same, [float("nan"), 0.2, 0.1], [0, 0, 0], 1.0, 10) == [1, 2, 0]  # NaN visited last


def test_box_conversion_and_mapping():
    x0, y0, x1, y1 = g.cxcywh_to_xyxy((0.5, 0.5, 0.25, 0.5))
    assert (float(x0), float(y0), float(x1), float(y1)) == (0.375, 0.25, 0.625, 0.75)
    geom = {"crop_left": 58, "crop_top": 16, "crop_width": 224, "crop_height": 224, "resized_width": 341,
            "resized_height": 256}
    m = g.map_to_source((np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(1.0)), geom)
    assert float(m[0]) == pytest.approx(58 / 341, abs=1e-6) and float(m[2]) == pytest.approx(282 / 341, abs=1e-6)
    assert float(m[1]) == pytest.approx(16 / 256, abs=1e-6) and float(m[3]) == pytest.approx(240 / 256, abs=1e-6)


def test_c27_geometry_by_hand():
    sec = {"resize_mode": "ShortestEdgeCenterCrop", "resize_shortest_edge": 256, "image_width": 224, "image_height": 224}
    assert g.c27_geometry(640, 480, sec) == ((341, 256), (58, 16, 224, 224))  # int(256*640/480) = 341
    assert g.c27_geometry(480, 640, sec) == ((256, 341), (16, 58, 224, 224))
    scc = {"resize_mode": "StretchCenterCrop", "resize_width": 260, "resize_height": 240, "image_width": 200,
           "image_height": 224}
    assert g.c27_geometry(300, 200, scc) == ((260, 240), (30, 8, 200, 224))
    st = {"resize_mode": "Stretch", "image_width": 32, "image_height": 40}
    assert g.c27_geometry(5, 5, st) == ((32, 40), (0, 0, 32, 40))
    with pytest.raises(ValueError):
        g.c27_geometry(10, 10, {"resize_mode": "StretchCenterCrop", "resize_width": 10, "resize_height": 10,
                                "image_width": 20, "image_height": 20})


def test_decode_reference_hand_cases():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    # softmax over (x, y, no-object): row 0 = (ln 3, 0, 0) -> (0.6, 0.2, 0.2) -> class x, score 0.6; row 1 favours no-object
    logits = np.array([[math.log(3.0), 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    dets = g.decode_reference(logits, boxes, ["x", "y"], "Softmax", 0.5, 0.7, 100)
    assert len(dets) == 1 and dets[0]["label"] == "x" and dets[0]["index"] == 0
    assert dets[0]["score"] == pytest.approx(0.6, abs=1e-6)
    assert dets[0]["box"] == pytest.approx([0.4, 0.4, 0.6, 0.6], abs=1e-6)
    # sigmoid: logit 0 -> exactly 0.5, not above a 0.5 threshold; ln 3 -> 0.75
    sl = np.array([[0.0, math.log(3.0)], [0.0, 0.0]], dtype=np.float32)
    d2 = g.decode_reference(sl, boxes, ["p", "q"], "Sigmoid", 0.5, 0.7, 100)
    assert [(d["label"], round(d["score"], 6)) for d in d2] == [("q", 0.75)]
    with pytest.raises(ValueError):
        g.decode_reference(sl, boxes, ["p"], "Sigmoid", 0.5, 0.7, 100)


def test_images_are_deterministic_and_have_pure_areas():
    a = g.make_image(64, 48, np.random.RandomState(1))
    b = g.make_image(64, 48, np.random.RandomState(1))
    c = g.make_image(64, 48, np.random.RandomState(2))
    assert a.shape == (48, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (a == 0).any() and (a == 255).any()
    assert g.make_image(1, 1, np.random.RandomState(3)).shape == (1, 1, 3)
    assert g.make_image(2, 3, np.random.RandomState(3)).shape == (3, 2, 3)


def test_designs_cover_the_required_situations():
    for variant in ("visible", "hidden"):
        for act in ("Sigmoid", "Softmax"):
            r = g.check_design(g.DESIGNS[variant], act)
            assert 0.77 <= r["iou_best_vs_08"] <= 0.83 and 0.57 <= r["iou_best_vs_06"] <= 0.63
            assert r["min_score_margin"] >= 0.02 and r["min_nms_margin"] >= 0.02


# ---- generated visible goldens ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc():
    if not os.path.isfile(GOLDEN):
        pytest.skip("run Tools/make_vision_goldens.py first (%s missing)" % GOLDEN)
    with open(GOLDEN, "r", encoding="utf-8") as f:
        return json.load(f)


def png_size(root, rel):
    with Image.open(os.path.join(root, rel)) as im:
        assert im.mode == "RGB"
        return im.size


def check_files_and_counts(d, hidden):
    root = d["root"]
    assert d["schema"] == "pipelines-vision-goldens/1"
    for k in ("python", "pillow", "numpy", "transformers", "onnxruntime", "seed", "variant", "script"):
        assert d["generator"][k]
    assert d["generator"]["variant"] == ("hidden" if hidden else "visible")
    assert "\\" not in d["root"] and "\\" not in d["models_root"]
    assert [i["id"] for i in d["images"]] == ["s%02d" % i for i in range(10)]
    assert len(d["resize_cases"]) == 60 and len(d["preprocess_cases"]) == 52
    assert [m["slug"] for m in d["tiny_models"]] == ["tiny_image_classification", "tiny_object_detection",
                                                     "tiny_det_designed_sigmoid", "tiny_det_designed_softmax"]
    for im in d["images"]:
        assert png_size(root, im["file"]) == (im["width"], im["height"])
        assert len(im["probes"]) == 16
    for rc in d["resize_cases"]:
        assert png_size(root, rc["expected"]) == (rc["width"], rc["height"])
    cfgs = {}
    for pc in d["preprocess_cases"]:
        cfgs.setdefault(pc["config"], []).append(pc["image"])
        assert png_size(root, pc["cropped"]) == (pc["crop"][2], pc["crop"][3])
        n = 1
        for v in pc["tensor_shape"]:
            n *= v
        assert pc["tensor_shape"] == [1, 3, pc["crop"][3], pc["crop"][2]]
        assert os.path.getsize(os.path.join(root, pc["tensor"])) == 4 * n
        assert 0.0 <= pc["crosscheck_max_abs_diff"] <= 1e-5
    assert list(cfgs) == ["convnext_crop", "rtdetr_stretch", "deit_crop", "deit_rect_crop", "clip_crop", "vit_stretch",
                          "levit_crop"]
    for name, ids in cfgs.items():
        assert ids == (["s00", "s02", "s03", "s05"] if name == "rtdetr_stretch"
                       else ["s00", "s01", "s02", "s03", "s04", "s05", "s07", "s08"])
    for m in d["tiny_models"]:
        assert os.path.isfile(os.path.join(d["models_root"], m["folder"], "model.onnx"))
        assert os.path.isfile(os.path.join(d["models_root"], m["folder"], "config.json"))
        assert m["cases"]
        for c in m["cases"]:
            if m["task"] == "ImageClassification":
                assert len(c["logits"]) == len(c["scores"]) == len(m["package"]["labels"])
                assert c["label"] == m["package"]["labels"][c["index"]]
            else:
                assert len(c["logits"]) == int(np.prod(c["logits_shape"]))
                assert len(c["pred_boxes"]) == int(np.prod(c["pred_boxes_shape"]))
                assert c["logits_shape"][0] == 1 and c["pred_boxes_shape"][2] == 4
    for k in ("tiny_det_designed_sigmoid", "tiny_det_designed_softmax"):
        m = next(x for x in d["tiny_models"] if x["slug"] == k)
        assert len(m["cases"]) == 8
        assert [(c["score_threshold"], c["nms_iou_threshold"], c["max_detections"]) for c in m["cases"][:4]] == \
            [tuple(p) for p in g.DET_PARAMS]
        assert {c["image"] for c in m["cases"]} == {"s02", "s07"}


def test_schema_files_and_counts(doc):
    check_files_and_counts(doc, hidden=False)
    assert [(i["width"], i["height"]) for i in doc["images"]] == g.IMAGE_SIZES["visible"]
    assert (doc["images"][0]["width"], doc["images"][0]["height"]) == (640, 480)
    assert doc["generator"]["seed"] == 4242


def test_resize_case_grid(doc):
    combos = {(r["filter"], r["pil_filter"]) for r in doc["resize_cases"]}
    assert combos == {("Nearest", 0), ("Lanczos", 1), ("Bilinear", 2), ("Bicubic", 3), ("Box", 4), ("Hamming", 5)}
    pairs = {(r["image"], r["width"], r["height"]) for r in doc["resize_cases"]}
    assert pairs == {("s00", 224, 168), ("s03", 455, 256), ("s05", 300, 15), ("s04", 224, 320), ("s06", 3, 5),
                     ("s07", 224, 224), ("s08", 256, 255), ("s02", 640, 640), ("s01", 200, 700), ("s09", 7, 1000)}
    assert doc["resize_cases"][0]["id"] == "r000" and doc["resize_cases"][-1]["id"] == "r059"


def load(doc, rel):
    return np.array(Image.open(os.path.join(doc["root"], rel)).convert("RGB"))


def test_identity_resize_equals_source(doc):
    for rc in doc["resize_cases"]:
        if rc["image"] == "s07":
            assert np.array_equal(load(doc, rc["expected"]), load(doc, "images/s07.png"))


def test_one_resize_case_recomputed_with_pillow(doc):
    rc = next(r for r in doc["resize_cases"] if r["image"] == "s00" and r["filter"] == "Bicubic")
    src = Image.open(os.path.join(doc["root"], "images/s00.png")).convert("RGB")
    again = np.array(src.resize((rc["width"], rc["height"]), resample=3))
    assert np.array_equal(again, load(doc, rc["expected"]))
    rc2 = next(r for r in doc["resize_cases"] if r["image"] == "s04" and r["filter"] == "Lanczos")
    src2 = Image.open(os.path.join(doc["root"], "images/s04.png")).convert("RGB")
    assert np.array_equal(np.array(src2.resize((rc2["width"], rc2["height"]), resample=1)), load(doc, rc2["expected"]))


def test_rgb_sum_and_probes_match_pngs(doc):
    for im in doc["images"]:
        arr = load(doc, im["file"])
        assert int(arr.astype(np.int64).sum()) == im["rgb_sum"]
        for p in im["probes"]:
            assert arr[p["y"], p["x"]].tolist() == p["rgb"]
        pts = [(p["x"], p["y"]) for p in im["probes"]]
        w, h = im["width"], im["height"]
        assert pts[:5] == [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, h // 2)]
    for im in doc["images"]:
        if im["width"] * im["height"] >= 1000:
            arr = load(doc, im["file"])
            assert (arr == 0).any() and (arr == 255).any()


IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EXPECTED_SETTINGS = {
    "convnext_crop": dict(image_width=224, image_height=224, resize_mode="ShortestEdgeCenterCrop", resize_shortest_edge=256,
                          resize_width=0, resize_height=0, resample="Bicubic", rescale_factor=F255,
                          image_mean=IMAGENET[0], image_std=IMAGENET[1]),
    "rtdetr_stretch": dict(image_width=640, image_height=640, resize_mode="Stretch", resize_shortest_edge=0,
                           resize_width=0, resize_height=0, resample="Bilinear", rescale_factor=F255,
                           image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0]),
    "deit_crop": dict(image_width=224, image_height=224, resize_mode="StretchCenterCrop", resize_shortest_edge=0,
                      resize_width=256, resize_height=256, resample="Bicubic", rescale_factor=F255,
                      image_mean=IMAGENET[0], image_std=IMAGENET[1]),
    "deit_rect_crop": dict(image_width=200, image_height=224, resize_mode="StretchCenterCrop", resize_shortest_edge=0,
                           resize_width=260, resize_height=240, resample="Bilinear", rescale_factor=F255,
                           image_mean=IMAGENET[0], image_std=IMAGENET[1]),
    "clip_crop": dict(image_width=224, image_height=224, resize_mode="ShortestEdgeCenterCrop", resize_shortest_edge=224,
                      resize_width=0, resize_height=0, resample="Bicubic", rescale_factor=F255,
                      image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711]),
    "vit_stretch": dict(image_width=224, image_height=224, resize_mode="Stretch", resize_shortest_edge=0,
                        resize_width=0, resize_height=0, resample="Bilinear", rescale_factor=F255,
                        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]),
    "levit_crop": dict(image_width=224, image_height=224, resize_mode="ShortestEdgeCenterCrop", resize_shortest_edge=256,
                       resize_width=0, resize_height=0, resample="Bicubic", rescale_factor=F255,
                       image_mean=IMAGENET[0], image_std=IMAGENET[1]),
}


def test_settings_of_all_configs_equal_hand_written(doc):
    seen = set()
    for pc in doc["preprocess_cases"]:
        exp = EXPECTED_SETTINGS[pc["config"]]
        s = pc["settings"]
        assert list(s) == list(exp)
        for k, v in exp.items():
            if isinstance(v, float):
                assert s[k] == pytest.approx(v, rel=1e-15, abs=0)
            elif isinstance(v, list):
                assert s[k] == pytest.approx(v, rel=1e-15, abs=0)
            else:
                assert s[k] == v, (pc["config"], k)
        seen.add(pc["config"])
    assert seen == set(EXPECTED_SETTINGS)


def case(doc, config, image):
    return next(p for p in doc["preprocess_cases"] if p["config"] == config and p["image"] == image)


def test_geometry_by_hand(doc):
    p = case(doc, "convnext_crop", "s00")  # 640x480: 480 is the short side -> 256; int(256*640/480) = 341
    assert (p["resized"], p["crop"]) == ([341, 256], [58, 16, 224, 224])
    p = case(doc, "clip_crop", "s01")  # 480x640: width short -> 224; int(224*640/480) = 298; top (298-224)//2 = 37
    assert (p["resized"], p["crop"]) == ([224, 298], [0, 37, 224, 224])
    p = case(doc, "levit_crop", "s05")  # 1000x50: int(256*1000/50) = 5120; left (5120-224)//2 = 2448
    assert (p["resized"], p["crop"]) == ([5120, 256], [2448, 16, 224, 224])
    p = case(doc, "deit_rect_crop", "s02")  # stretch 260x240, crop 200x224 at (30, 8)
    assert (p["resized"], p["crop"]) == ([260, 240], [30, 8, 200, 224])
    assert p["tensor_shape"] == [1, 3, 224, 200]
    p = case(doc, "rtdetr_stretch", "s03")
    assert (p["resized"], p["crop"]) == ([640, 640], [0, 0, 640, 640]) and p["tensor_shape"] == [1, 3, 640, 640]


def test_tensor_matches_hand_pixel(doc):
    # vit_stretch on s07 (224x224 -> identity resize): R of pixel (0, 0): (p/255 - 0.5)/0.5
    p = case(doc, "vit_stretch", "s07")
    t = np.fromfile(os.path.join(doc["root"], p["tensor"]), dtype="<f4").reshape(p["tensor_shape"])
    px = load(doc, "images/s07.png")
    for c in range(3):
        want = (np.float32(np.float64(px[5, 7, c]) * np.float64(np.float32(1 / 255))) - np.float32(0.5)) / np.float32(0.5)
        # HF multiplies by the config's double 1/255, C27 by the float32-rounded factor: they differ by ~1e-7
        assert float(t[0, c, 5, 7]) == pytest.approx(float(want), abs=1e-5)
    assert p["tensor_sum"] == pytest.approx(float(np.sum(t, dtype=np.float64)), rel=1e-12)
    assert p["tensor_abs_max"] == pytest.approx(float(np.abs(t).max()), abs=1e-6)


def det_case(doc, slug, image, thr, nms, mx):
    m = next(x for x in doc["tiny_models"] if x["slug"] == slug)
    return next(c for c in m["cases"] if c["image"] == image and c["score_threshold"] == thr
                and c["nms_iou_threshold"] == nms and c["max_detections"] == mx)


def check_dets(dets, expected):
    assert [(d["label"], d["index"]) for d in dets] == [(e[0], e[1]) for e in expected]
    for d, e in zip(dets, expected):
        assert d["score"] == pytest.approx(e[2], abs=1e-5)
        assert d["box"] == pytest.approx(e[3], abs=1e-6)


# Hand-derived from the design table in make_vision_goldens.py:
# q0 a .92 [.2 .25 .6 .65]; q1 a .85 [.24 .25 .64 .65] (IoU .818 with q0); q2 a .78 [.3 .25 .7 .65] (IoU .6 with q0);
# q3 b .88 [.8 .38 1.04 .62] (clamped to x1 = 1); q4 c .70 [.812 .38 1.052 .62] (IoU .905 with q3, other class);
# q5 a .62 + c .56 [.05 .7 .25 .9]; q6 c .40 (a .20) [.4 .75 .6 .95]; q7 b .08 [.2 .05 .4 .25].
Q0 = ("a", 0, 0.92, [0.2, 0.25, 0.6, 0.65])
Q1 = ("a", 0, 0.85, [0.24, 0.25, 0.64, 0.65])
Q2 = ("a", 0, 0.78, [0.3, 0.25, 0.7, 0.65])
Q3 = ("b", 1, 0.88, [0.8, 0.38, 1.0, 0.62])
Q4 = ("c", 2, 0.70, [0.812, 0.38, 1.0, 0.62])
Q5A = ("a", 0, 0.62, [0.05, 0.7, 0.25, 0.9])
Q5C = ("c", 2, 0.56, [0.05, 0.7, 0.25, 0.9])
Q6C = ("c", 2, 0.40, [0.4, 0.75, 0.6, 0.95])


def test_designed_sigmoid_detections_by_hand(doc):
    # thr .5: candidates q0 q3 q1 q2 q4 q5a q5c; nms .7 drops q1 (IoU .818 > .7), keeps q2 (.6)
    check_dets(det_case(doc, "tiny_det_designed_sigmoid", "s02", 0.5, 0.7, 100)["detections"], [Q0, Q3, Q2, Q4, Q5A, Q5C])
    # thr .3 adds q6c (.40); nms .5 drops q1 and q2
    check_dets(det_case(doc, "tiny_det_designed_sigmoid", "s02", 0.3, 0.5, 100)["detections"], [Q0, Q3, Q4, Q5A, Q5C, Q6C])
    # nms disabled
    check_dets(det_case(doc, "tiny_det_designed_sigmoid", "s07", 0.5, 1.0, 100)["detections"],
               [Q0, Q3, Q1, Q2, Q4, Q5A, Q5C])
    # thr .05, nms .5, max 3: top 8 candidates (10 above .05 -> top-Q cut), nms, first 3
    check_dets(det_case(doc, "tiny_det_designed_sigmoid", "s02", 0.05, 0.5, 3)["detections"], [Q0, Q3, Q4])
    c = det_case(doc, "tiny_det_designed_sigmoid", "s02", 0.5, 0.7, 100)
    assert c["logits_shape"] == [1, 8, 3] and c["pred_boxes_shape"] == [1, 8, 4]
    # raw logit of q0 class a = ln(.92/.08)
    assert c["logits"][0] == pytest.approx(math.log(0.92 / 0.08), abs=1e-6)
    assert c["pred_boxes"][:4] == pytest.approx([0.40, 0.45, 0.40, 0.40], abs=1e-7)


def test_designed_softmax_detections_by_hand(doc):
    check_dets(det_case(doc, "tiny_det_designed_softmax", "s02", 0.5, 0.7, 100)["detections"], [Q0, Q3, Q2, Q4, Q5A])
    check_dets(det_case(doc, "tiny_det_designed_softmax", "s02", 0.3, 0.5, 100)["detections"], [Q0, Q3, Q4, Q5A, Q6C])
    c = det_case(doc, "tiny_det_designed_softmax", "s07", 0.5, 1.0, 100)
    check_dets(c["detections"], [Q0, Q3, Q1, Q2, Q4, Q5A])
    assert c["logits_shape"] == [1, 8, 4]


def test_designed_models_have_dynamic_batch_and_constant_outputs(doc):
    import onnxruntime as ort
    for slug, ncls in (("tiny_det_designed_sigmoid", 3), ("tiny_det_designed_softmax", 4)):
        folder = os.path.join(doc["models_root"], "vision_test", slug)
        sess = ort.InferenceSession(os.path.join(folder, "model.onnx"), providers=["CPUExecutionProvider"])
        assert sess.get_inputs()[0].shape == ["batch", 3, 32, 32]
        one = sess.run(None, {"pixel_values": np.zeros((1, 3, 32, 32), np.float32)})
        three = sess.run(None, {"pixel_values": np.random.RandomState(0).rand(3, 3, 32, 32).astype(np.float32)})
        assert one[0].shape == (1, 8, ncls) and one[1].shape == (1, 8, 4)
        assert three[0].shape == (3, 8, ncls) and three[1].shape == (3, 8, 4)
        for b in range(3):
            assert np.array_equal(three[0][b], one[0][0]) and np.array_equal(three[1][b], one[1][0])
        cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        assert cfg["id2label"] == {"0": "a", "1": "b", "2": "c"}
        lic = json.load(open(os.path.join(folder, "modelpipelines.json"), encoding="utf-8"))
        assert list(lic) == ["license"] and lic["license"]["spdx_id"] == "Apache-2.0"


def test_tiny_model_blocks(doc):
    ic = doc["tiny_models"][0]
    assert ic["task"] == "ImageClassification" and ic["package"]["score_activation"] == "Softmax"
    assert ic["package"]["labels"] == ["cat", "café", "日本", "dog"]
    assert [c["image"] for c in ic["cases"]] == ["s00", "s02", "s04", "s07"]
    for c in ic["cases"]:
        assert sum(c["scores"]) == pytest.approx(1.0, abs=1e-6)
        assert c["index"] == int(np.argmax(c["scores"]))
    assert ic["package"]["settings"]["resize_mode"] == "ShortestEdgeCenterCrop"
    assert ic["package"]["settings"]["resize_shortest_edge"] == 36  # int(32 / 0.875)
    for m in doc["tiny_models"]:
        assert m["package"]["nms_iou_threshold"] == 0.7 and m["package"]["max_detections"] == 100
    assert doc["tiny_models"][2]["package"]["score_activation"] == "Sigmoid"
    assert doc["tiny_models"][3]["package"]["score_activation"] == "Softmax"
    od = doc["tiny_models"][1]
    for c in od["cases"]:  # the margin rule: no kept case has a score within 0.02 of its threshold
        lg = np.array(c["logits"], dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-lg))
        assert np.min(np.abs(s - c["score_threshold"])) >= 0.02


# ---- hidden variant (run into a temp dir) -----------------------------------------------------------------------

def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_hidden(out_dir, models_out):
    r = subprocess.run([PY, SCRIPT, "--variant", "hidden", "--seed", "777", "--out-dir", out_dir, "--models-out",
                        models_out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def hidden(tmp_path_factory):
    if not os.path.isdir(os.path.join(DEV, "models", "reference", "resnet-50")):
        pytest.skip("reference models missing")
    base = tmp_path_factory.mktemp("vision_hidden")
    out_dir, models_out = str(base / "g"), str(base / "m")
    vis_before = sha(GOLDEN) if os.path.isfile(GOLDEN) else None
    run_hidden(out_dir, models_out)
    path = os.path.join(out_dir, "vision_goldens.json")
    first = {"json": sha(path)}
    for sub in ("images", "resize", "preprocess"):
        for n in sorted(os.listdir(os.path.join(out_dir, sub))):
            first[sub + "/" + n] = sha(os.path.join(out_dir, sub, n))
    first["onnx"] = sha(os.path.join(models_out, "vision_test", "tiny_det_designed_sigmoid", "model.onnx"))
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {"doc": d, "out": out_dir, "models": models_out, "first": first, "vis_before": vis_before}


def test_hidden_structure_and_files(hidden):
    d = hidden["doc"]
    check_files_and_counts(d, hidden=True)
    assert d["generator"]["seed"] == 777
    assert [(i["width"], i["height"]) for i in d["images"]] == [(512, 384), (384, 512), (333, 250), (1280, 720),
                                                                (41, 29), (900, 60), (2, 3), (256, 256), (199, 301),
                                                                (60, 900)]
    assert d["root"] == hidden["out"].replace("\\", "/") and d["models_root"] == hidden["models"].replace("\\", "/")
    assert os.path.isdir(os.path.join(hidden["models"], "import_test", "tiny_image_classification"))


def test_hidden_designed_models(hidden):
    d = hidden["doc"]
    for slug, ncls in (("tiny_det_designed_sigmoid", 4), ("tiny_det_designed_softmax", 5)):
        m = next(x for x in d["tiny_models"] if x["slug"] == slug)
        assert m["package"]["labels"] == ["w", "x", "y", "z"]
        for c in m["cases"]:
            assert c["logits_shape"] == [1, 10, ncls] and c["pred_boxes_shape"] == [1, 10, 4]
    # hand-derived: sigmoid, thr .5, nms .7: q0 w .94, q3 x .91, q1 w .86 (IoU .818 with q0: dropped), q2 w .74 (IoU .6:
    # kept), q4 y .72, q5 z .67 (box [.78 -.04 1.08 .2] clamped to [.78 0 1 .2]), q6 x .63, q6 z .55
    c = det_case(d, "tiny_det_designed_sigmoid", "s02", 0.5, 0.7, 100)
    check_dets(c["detections"], [
        ("w", 0, 0.94, [0.20, 0.15, 0.50, 0.65]),
        ("x", 1, 0.91, [0.58, 0.60, 0.82, 0.80]),
        ("w", 0, 0.74, [0.275, 0.15, 0.575, 0.65]),
        ("y", 2, 0.72, [0.593, 0.60, 0.833, 0.80]),
        ("z", 3, 0.67, [0.78, 0.0, 1.0, 0.20]),
        ("x", 1, 0.63, [0.02, 0.75, 0.22, 0.95]),
        ("z", 3, 0.55, [0.02, 0.75, 0.22, 0.95])])


def test_hidden_differs_from_visible_and_leaves_visible_alone(hidden, doc):
    assert [i["rgb_sum"] for i in hidden["doc"]["images"]] != [i["rgb_sum"] for i in doc["images"]]
    assert hidden["vis_before"] == sha(GOLDEN)


def test_hidden_is_deterministic(hidden):
    out_dir, models_out = hidden["out"], hidden["models"]
    run_hidden(out_dir, models_out)
    first = hidden["first"]
    assert sha(os.path.join(out_dir, "vision_goldens.json")) == first["json"]
    for key, digest in first.items():
        if key in ("json", "onnx"):
            continue
        assert sha(os.path.join(out_dir, key)) == digest, key
    assert sha(os.path.join(models_out, "vision_test", "tiny_det_designed_sigmoid", "model.onnx")) == first["onnx"]
