#!/usr/bin/env python
"""Sample photos for the sample maps (WP5.3, contract C36).

Downloads the images listed in Tools/data/sample_images.json (pinned file URL + sha256 + licence + author), refuses
any entry whose licence is not CC0-1.0 or Public Domain ("PD"), verifies the sha256 of every download, and writes
each image as <name>.png (RGB, long edge <= 1024 px, Pillow LANCZOS) into --out (default <dev>/sample_images/).
The PNGs are committed as sample textures in the plugin Content, so only CC0 / public domain is accepted.

--check additionally runs the PyTorch reference models (resnet-50 for `classify`, rtdetr_r18vd for `detect`, the
same loading and preprocessing as make_vision_goldens.py) on the written PNGs and asserts the `expect` block of each
image. The exit code is 0 only if every image passes. The reference models come from the folders written by
fetch_reference_models.py (--models, default <dev>/models/reference) and the HF cache (--hf-cache, offline).
--verify-license re-reads the licence of each image from the Commons API (network).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import fetch_test_images as fti  # noqa: E402

SCHEMA = "pipelines-sample-images/1"
DATA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_images.json")
MAX_SIDE = 1024
PURPOSES = ("classify", "detect")
LICENSES = ("CC0-1.0", "PD")
NMS_IOU = 0.7
MAX_DET = 100


def classify_license(raw):
    """Maps a Commons licence string (LicenseShortName / License) to "CC0-1.0" or "PD"; anything else -> None."""
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s in ("cc0", "cc-zero", "cc zero", "cc0-1.0") or s.startswith("cc0 ") or s.startswith("cc0-") \
            or s.startswith("cc-zero") or s.startswith("cc zero"):
        return "CC0-1.0"
    if s == "public domain" or s.startswith("public domain ") or s == "pd" or s.startswith("pd-") \
            or s.startswith("pd ") or s == "pdm" or s.startswith("pdm-"):
        return "PD"
    return None


def validate_doc(doc):
    """Structural checks of the pinned JSON; returns a list of problems (empty = fine)."""
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append("schema is not " + SCHEMA)
    images = doc.get("images")
    if not isinstance(images, list) or not images:
        return bad + ["images is missing or empty"]
    names = set()
    for i, e in enumerate(images):
        tag = "images[%d] %s" % (i, e.get("name"))
        for k in ("name", "purpose", "page_url", "file_url", "sha256", "license", "license_source", "author", "expect"):
            if not e.get(k):
                bad.append("%s has no %s" % (tag, k))
        if e.get("name") in names:
            bad.append("%s: duplicate name" % tag)
        names.add(e.get("name"))
        if e.get("purpose") not in PURPOSES:
            bad.append("%s: purpose %r is not one of %s" % (tag, e.get("purpose"), PURPOSES))
        if e.get("license") not in LICENSES:
            bad.append("%s: licence %r is not CC0-1.0 or PD" % (tag, e.get("license")))
        elif classify_license(e.get("license_source")) != e.get("license"):
            bad.append("%s: license_source %r does not give %s" % (tag, e.get("license_source"), e.get("license")))
        sha = e.get("sha256") or ""
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            bad.append("%s: sha256 is not 64 lowercase hex digits" % tag)
        ex = e.get("expect") or {}
        if e.get("purpose") == "classify":
            if not ex.get("top1_any_of") or not isinstance(ex.get("min_score"), (int, float)):
                bad.append("%s: classify expect needs top1_any_of and min_score" % tag)
        elif e.get("purpose") == "detect":
            if not isinstance(ex.get("threshold"), (int, float)) or not isinstance(ex.get("min_detections"), int) \
                    or not ex.get("labels_include"):
                bad.append("%s: detect expect needs threshold, min_detections and labels_include" % tag)
    n_cls = sum(1 for e in images if e.get("purpose") == "classify")
    n_det = sum(1 for e in images if e.get("purpose") == "detect")
    if n_cls != 4 or n_det != 2:
        bad.append("need 4 classify and 2 detect images, have %d and %d" % (n_cls, n_det))
    return bad


def resize_to_png(src, dst):
    """RGB, long edge <= MAX_SIDE (LANCZOS, aspect kept); returns (width, height)."""
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > MAX_SIDE:
            s = MAX_SIDE / float(max(w, h))
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
        im.save(dst, format="PNG")
        return im.size


def label_matches(label, wanted):
    """True if `wanted` equals the whole label or one of its comma-separated synonyms (case-insensitive)."""
    parts = [p.strip().lower() for p in label.split(",")]
    return wanted.strip().lower() in parts or wanted.strip().lower() == label.strip().lower()


def eval_classify(label, score, expect):
    """Returns (ok, message) for a classification result against an `expect` block."""
    ok_label = any(label_matches(label, w) for w in expect["top1_any_of"])
    ok_score = score >= expect["min_score"]
    msg = []
    if not ok_label:
        msg.append("top-1 %r not in %s" % (label, expect["top1_any_of"]))
    if not ok_score:
        msg.append("score %.4f < %.2f" % (score, expect["min_score"]))
    return ok_label and ok_score, "; ".join(msg)


def eval_detect(labels, expect):
    """labels: label of every detection at the threshold. Returns (ok, message)."""
    msg = []
    if len(labels) < expect["min_detections"]:
        msg.append("%d detections < %d" % (len(labels), expect["min_detections"]))
    missing = [w for w in expect["labels_include"] if w not in labels]
    if missing:
        msg.append("missing labels %s" % missing)
    if len(set(labels)) < 2:
        msg.append("fewer than 2 distinct labels")
    return not msg, "; ".join(msg)


def fetch_all(doc, out_dir, cache_dir):
    """Downloads (or reuses cached files with matching sha256), verifies, writes the PNGs. Returns {name: png path}."""
    bad = validate_doc(doc)
    if bad:
        for b in bad:
            print("REFUSED " + b)
        raise SystemExit("FAIL: sample_images.json is invalid (%d problems)" % len(bad))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for e in doc["images"]:
        src = os.path.join(cache_dir, "%s.%s" % (e["name"], fti.ext_of(e["file_url"])))
        if os.path.isfile(src) and fti.sha256_file(src) == e["sha256"]:
            how = "cached"
        else:
            data = fti.http_get(e["file_url"])
            got = fti.sha256_bytes(data)
            if got != e["sha256"]:
                raise SystemExit("FAIL: sha256 mismatch for %s (%s): got %s, pinned %s" % (e["name"], e["file_url"], got,
                                                                                         e["sha256"]))
            with open(src, "wb") as f:
                f.write(data)
            how = "downloaded"
        dst = os.path.join(out_dir, e["name"] + ".png")
        size = resize_to_png(src, dst)
        print("%-22s %-10s sha256 ok  %-7s %dx%d  %s  by %s" % (e["name"], how, e["license"], size[0], size[1],
                                                               e["page_url"], e["author"]))
        paths[e["name"]] = dst
    return paths


def verify_licenses(doc):
    """Re-reads the licence of every image from the Commons API; returns the number of mismatches."""
    bad = 0
    for e in doc["images"]:
        title = "File:" + fti.urllib.parse.unquote(e["page_url"].rsplit("/File:", 1)[1]).replace("_", " ")
        r = fti.api_get({"action": "query", "prop": "imageinfo", "iiprop": "extmetadata", "titles": title,
                         "iiextmetadatafilter": "LicenseShortName|License"})
        meta = r["query"]["pages"][0]["imageinfo"][0]["extmetadata"]
        raw = fti.strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
        mapped = classify_license(raw)
        ok = mapped == e["license"]
        bad += 0 if ok else 1
        print("licence %-22s Commons says %-12r -> %s  %s" % (e["name"], raw, mapped, "OK" if ok else "MISMATCH"))
    return bad


def run_check(doc, paths, models_dir, hf_cache):
    """Runs the torch reference models on the PNGs; returns the list of (name, result text, ok, message)."""
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    import make_vision_goldens as g
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(1)
    transformers.logging.set_verbosity_error()
    manifest = g.read_json(os.path.join(models_dir, "manifest.json"))
    refs = {}
    for slug, repo, task in g.REFERENCE_MODELS:
        folder = os.path.join(models_dir, slug)
        if not os.path.isfile(os.path.join(folder, "config.json")):
            raise SystemExit("NEEDS: %s not found; run fetch_reference_models.py first (or pass --models)" % folder)
        labels = g.labels_from_config(g.read_json(os.path.join(folder, "config.json")))
        pp = g.Preprocessor(g.read_json(os.path.join(folder, "preprocessor_config.json")))
        cls = transformers.AutoModelForImageClassification if task == "ImageClassification" \
            else transformers.AutoModelForObjectDetection
        model = cls.from_pretrained(repo, revision=manifest[slug]["revision"], cache_dir=hf_cache,
                                    local_files_only=True, dtype=torch.float32).eval()
        refs[slug] = (labels, pp, model)
    results = []
    for e in doc["images"]:
        arr = np.array(Image.open(paths[e["name"]]).convert("RGB"))
        slug = "resnet-50" if e["purpose"] == "classify" else "rtdetr_r18vd"
        labels, pp, model = refs[slug]
        hf, _, _, _ = pp.process(arr, e["name"])
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(hf))
        ex = e["expect"]
        if e["purpose"] == "classify":
            p = g.softmax64(out.logits[0].numpy())
            i = int(np.argmax(p))
            ok, msg = eval_classify(labels[i], float(p[i]), ex)
            text = "top-1 %s %.4f" % (labels[i], float(p[i]))
        else:
            lg = out.logits.numpy().astype(np.float32)
            bx = out.pred_boxes.numpy().astype(np.float32)
            dets = g.decode_reference(lg[0], bx[0], labels, "Sigmoid", ex["threshold"], NMS_IOU, MAX_DET, None)
            names = [d["label"] for d in dets]
            ok, msg = eval_detect(names, ex)
            counts = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            text = "%d detections %s" % (len(names), dict(sorted(counts.items())))
        results.append((e["name"], text, ok, msg))
    return results


def main():
    dev = _paths.dev_dir()
    ap = argparse.ArgumentParser(description="Download the sample photos (CC0 / public domain only), verify sha256 and "
                                             "write <name>.png (long edge <= 1024). --check runs the reference models.")
    ap.add_argument("--out", help="output folder (default <dev>/sample_images)")
    ap.add_argument("--data", default=DATA_JSON, help="pinned metadata JSON (default Tools/data/sample_images.json)")
    ap.add_argument("--cache", help="folder for the downloaded originals (default <dev>/cache/sample_images, or "
                                    "<out>/.originals when --out is given)")
    ap.add_argument("--check", action="store_true",
                    help="run resnet-50 / rtdetr_r18vd (torch) on the PNGs and assert each image's expectation")
    ap.add_argument("--models", default=os.path.join(dev, "models", "reference"),
                    help="reference model folders with manifest.json, for --check")
    ap.add_argument("--hf-cache", default=os.path.join(dev, "cache", "hf"), help="HF cache with the weights, for --check")
    ap.add_argument("--verify-license", action="store_true", help="re-read each licence from the Commons API (network)")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(dev, "sample_images")
    cache_dir = os.path.abspath(args.cache) if args.cache else (
        os.path.join(out_dir, ".originals") if args.out else os.path.join(dev, "cache", "sample_images"))
    with open(args.data, "r", encoding="utf-8") as f:
        doc = json.load(f)
    try:
        paths = fetch_all(doc, out_dir, cache_dir)
    except OSError as ex:
        raise SystemExit("FAIL: cannot write to %s or %s: %s" % (out_dir, cache_dir, ex))
    code = 0
    if args.verify_license and verify_licenses(doc):
        code = 1
    if args.check:
        results = run_check(doc, paths, os.path.abspath(args.models), os.path.abspath(args.hf_cache))
        print()
        print("%-22s %-9s %-7s %-60s %s" % ("image", "purpose", "licence", "model output", "result"))
        by = {e["name"]: e for e in doc["images"]}
        for name, text, ok, msg in results:
            print("%-22s %-9s %-7s %-60s %s" % (name, by[name]["purpose"], by[name]["license"], text,
                                                "PASS" if ok else "FAIL " + msg))
            code |= 0 if ok else 1
    print("%d images -> %s%s" % (len(paths), out_dir, "" if code == 0 else "  (FAILED)"))
    return code


if __name__ == "__main__":
    sys.exit(main())
