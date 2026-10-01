"""Fetch and export the 4 reference models (WP2.1).

Output: <dev>/models/reference/<slug>/ (model.onnx + the repo's own files + modelpipelines.json)
and <dev>/models/reference/manifest.json. HF cache: <dev>/cache/hf.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

MODELS = [
    # slug, repo, revision prefix, task, expected inputs, expected outputs
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "1110a243", "feature-extraction",
     ["input_ids", "attention_mask", "token_type_ids"], ["last_hidden_state"]),
    ("distilbert-sst2", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0fa",
     "text-classification", ["input_ids", "attention_mask"], ["logits"]),
    ("resnet-50", "microsoft/resnet-50", "34c2154c", "image-classification", ["pixel_values"], ["logits"]),
    ("rtdetr_r18vd", "PekingU/rtdetr_r18vd", "ac77a11f", "object-detection", ["pixel_values"],
     ["logits", "pred_boxes"]),
]
COPY_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
              "preprocessor_config.json", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json",
              "README.md"]
OPSET = 17
TEXT = "The blacksmith is in the village."


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def front_matter_license(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    for line in text[3:end].splitlines():
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def resolve_revision(api, repo, prefix):
    commits = [c.commit_id for c in api.list_repo_commits(repo)]
    hits = [c for c in commits if c.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("FAIL: revision prefix %s for %s resolved to %d commits" % (prefix, repo, len(hits)))
    return hits[0]


def export_optimum(repo_dir, task, out_dir):
    from optimum.exporters.onnx import main_export
    main_export(repo_dir, output=out_dir, task=task, opset=OPSET, do_validation=False, no_post_process=True,
                device="cpu")


def export_torch(repo_dir, task, out_path, in_names, out_names):
    import torch
    from transformers import AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(repo_dir).eval()
    model.config.return_dict = True

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        torch.onnx.export(Wrap(model), (x,), out_path, input_names=in_names, output_names=out_names,
                          opset_version=OPSET, dynamo=False,
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"},
                                        "pred_boxes": {0: "batch"}})


def finalize_onnx(path):
    """Re-save as a single file without external data; return (opset, ir_version)."""
    import onnx
    m = onnx.load(path, load_external_data=True)
    onnx.save(m, path, save_as_external_data=False)
    opset = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0]
    return m, opset, m.ir_version


def sanity_check(slug, task, folder, onnx_path):
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import (AutoModel, AutoModelForImageClassification, AutoModelForObjectDetection,
                              AutoModelForSequenceClassification, AutoTokenizer)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    if task in ("feature-extraction", "text-classification"):
        tok = AutoTokenizer.from_pretrained(folder)
        enc = tok(TEXT, return_tensors="np")
        feed = {n: enc[n].astype(np.int64) for n in in_names}
        cls = AutoModel if task == "feature-extraction" else AutoModelForSequenceClassification
        model = cls.from_pretrained(folder).eval()
        targs = {n: torch.from_numpy(feed[n]) for n in in_names}
    else:
        size = 224 if task == "image-classification" else 640
        rng = np.random.RandomState(1234)
        arr = rng.rand(1, 3, size, size).astype(np.float32)
        feed = {"pixel_values": arr}
        cls = AutoModelForImageClassification if task == "image-classification" else AutoModelForObjectDetection
        model = cls.from_pretrained(folder).eval()
        targs = {"pixel_values": torch.from_numpy(arr)}
    out_names = [o.name for o in sess.get_outputs()]
    ort_out = sess.run(out_names, feed)
    with torch.no_grad():
        t = model(**targs)
    diffs = {}
    for name, o in zip(out_names, ort_out):
        ref = getattr(t, name).numpy()
        diffs[name] = float(np.max(np.abs(o - ref)))
        print("  %s %s: max |ort - torch| = %.3g" % (slug, name, diffs[name]))
        if diffs[name] > 1e-3:
            raise SystemExit("FAIL: %s output %s differs by %g > 1e-3" % (slug, name, diffs[name]))
    return diffs


def describe_io(onnx_path):
    import onnx
    m = onnx.load(onnx_path)

    def one(v):
        tt = v.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param or "?" for d in tt.shape.dim]
        return {"name": v.name, "elem_type": onnx.TensorProto.DataType.Name(tt.elem_type), "shape": dims}
    return [one(i) for i in m.graph.input], [one(o) for o in m.graph.output]


PARITY_STRINGS = ["The blacksmith is in the village.", "I can't believe it's not butter!",
                  "Café déjà vu — naïve", "", "Hello, world! 123 4.5 foo-bar",
                  "UPPER lower MiXeD   spaces\tand tabs"]


def ensure_tokenizer_json(slug, folder, snap, entry):
    """Make sure folder/tokenizer.json exists. A repo's own file is never overwritten.
    If the repo has none, convert from vocab.txt and verify against the slow tokenizer. Updates entry."""
    dst = os.path.join(folder, "tokenizer.json")
    if os.path.isfile(dst) and entry.get("tokenizer_json", "repo") == "repo":
        entry["tokenizer_json"] = "repo"
        return
    if os.path.isfile(dst):  # converted earlier; keep, but re-verify below
        pass
    else:
        if os.path.isfile(os.path.join(snap, "tokenizer.json")):
            shutil.copyfile(os.path.join(snap, "tokenizer.json"), dst)
            entry["tokenizer_json"] = "repo"
            return
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
        fast = AutoTokenizer.from_pretrained(snap, use_fast=True)
        fast.backend_tokenizer.save(dst)
        entry["tokenizer_json"] = "converted (transformers %s, tokenizers %s) from vocab.txt" % (
            transformers.__version__, tokenizers.__version__)
        print("  %s: wrote tokenizer.json converted from vocab.txt" % slug)
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tk = Tokenizer.from_file(dst)
    slow = AutoTokenizer.from_pretrained(snap, use_fast=False)
    bad = 0
    for s in PARITY_STRINGS:
        a = tk.encode(s, add_special_tokens=True).ids
        b = slow(s, add_special_tokens=True)["input_ids"]
        ok = a == b
        bad += 0 if ok else 1
        print("  parity %-45r %s (%d ids)" % (s, "EQUAL" if ok else "DIFFERENT", len(a)))
        if not ok:
            print("    tokenizers:", a, "\n    slow      :", b)
    if bad:
        raise SystemExit("FAIL: %s tokenizer.json differs from the slow tokenizer on %d strings" % (slug, bad))


def process(slug, repo, prefix, task, exp_in, exp_out, dev, manifest, force):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    cache = os.path.join(dev, "cache", "hf")
    folder = os.path.join(dev, "models", "reference", slug)
    onnx_path = os.path.join(folder, "model.onnx")
    entry = manifest.get(slug)
    if (not force and entry and os.path.isfile(onnx_path) and sha256_file(onnx_path) == entry.get("sha256")):
        print("%s: up to date (sha256 matches manifest), skipping export" % slug)
        if task in ("feature-extraction", "text-classification"):
            pats = [f for f in COPY_FILES if f != "tokenizer.json"] + ["tokenizer.json"]
            try:
                snap = snapshot_download(repo, revision=entry["revision"], cache_dir=cache,
                                         allow_patterns=pats, local_files_only=True)
            except Exception:  # noqa: BLE001
                snap = snapshot_download(repo, revision=entry["revision"], cache_dir=cache, allow_patterns=pats)
            ensure_tokenizer_json(slug, folder, snap, entry)
        return
    api = HfApi()
    sha = resolve_revision(api, repo, prefix)
    print("%s: %s @ %s" % (slug, repo, sha))
    readme = hf_hub_download(repo, "README.md", revision=sha, cache_dir=cache)
    lic = front_matter_license(readme)
    if lic != "apache-2.0":
        raise SystemExit("FAIL: %s license is %r, only apache-2.0 accepted; nothing more downloaded" % (repo, lic))
    files = set(api.list_repo_files(repo, revision=sha))
    patterns = [f for f in COPY_FILES if f in files]
    if "2_Normalize/config.json" in files:
        patterns.append("2_Normalize/config.json")
    if "model.safetensors" in files:
        patterns.append("model.safetensors")
    elif "pytorch_model.bin" in files:
        patterns.append("pytorch_model.bin")
    else:
        raise SystemExit("FAIL: %s has no safetensors/pytorch weights" % repo)
    snap = snapshot_download(repo, revision=sha, cache_dir=cache, allow_patterns=patterns)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    tmp = os.path.join(dev, "tmp", "export_" + slug)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    exporter = "optimum"
    try:
        if slug == "rtdetr_r18vd":
            raise RuntimeError("use torch.onnx.export for RT-DETR")
        export_optimum(snap, task, tmp)
        src = os.path.join(tmp, "model.onnx")
    except Exception as e:  # noqa: BLE001
        print("  optimum not used (%s); falling back to torch.onnx.export" % e)
        exporter = "torch.onnx.export"
        if task != "object-detection":
            raise
        src = os.path.join(tmp, "model.onnx")
        export_torch(snap, task, src, exp_in, exp_out)
    m, opset, ir = finalize_onnx(src)
    # Drop extra graph outputs only if clean
    names = [o.name for o in m.graph.output]
    if names != exp_out:
        extra = [n for n in names if n not in exp_out]
        missing = [n for n in exp_out if n not in names]
        if missing:
            raise SystemExit("NEEDS: %s export lacks outputs %s (has %s)" % (slug, missing, names))
        import onnx
        keep = [o for o in m.graph.output if o.name in exp_out]
        del m.graph.output[:]
        m.graph.output.extend(keep)
        onnx.save(m, src, save_as_external_data=False)
        print("  dropped extra outputs", extra)
    if slug == "rtdetr_r18vd":
        # The tracer leaves the query dim as a generated dim_param; it is 300 x (80 | 4) for this model.
        # Shape metadata only: the graph is unchanged.
        import onnx
        for o, tail in zip(m.graph.output, ([300, 80], [300, 4])):
            dims = o.type.tensor_type.shape.dim
            for d, v in zip(list(dims)[1:], tail):
                d.ClearField("dim_param")
                d.dim_value = v
        onnx.save(m, src, save_as_external_data=False)
    shutil.copyfile(src, onnx_path)
    for f in patterns:
        if f.endswith((".safetensors", ".bin")):
            continue
        dst = os.path.join(folder, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(snap, f), dst)
    # weights needed in folder for sanity check only via snapshot; load torch model from snapshot
    sidecar = {"license": {"spdx_id": "Apache-2.0",
                           "source_url": "https://huggingface.co/%s/tree/%s" % (repo, sha),
                           "attribution": "%s (Hugging Face), Apache-2.0" % repo}}
    with open(os.path.join(folder, "modelpipelines.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")
    ins, outs = describe_io(onnx_path)
    if [i["name"] for i in ins] != exp_in or [o["name"] for o in outs] != exp_out:
        raise SystemExit("FAIL: %s I/O names %s -> %s differ from contract" % (slug, ins, outs))
    diffs = sanity_check(slug, task, snap, onnx_path)
    shutil.rmtree(tmp, ignore_errors=True)
    manifest[slug] = {"repo": repo, "revision": sha, "license": lic, "opset": opset, "ir_version": ir,
                      "sha256": sha256_file(onnx_path), "bytes": os.path.getsize(onnx_path),
                      "exporter": exporter, "max_diff": diffs, "inputs": ins, "outputs": outs}
    if task in ("feature-extraction", "text-classification"):
        ensure_tokenizer_json(slug, folder, snap, manifest[slug])
    else:
        manifest[slug]["tokenizer_json"] = "n/a (image model)"
    print("  wrote %s (%d bytes, opset %d, ir %d, exporter %s)" % (onnx_path, manifest[slug]["bytes"], opset, ir,
                                                                 exporter))


def main():
    ap = argparse.ArgumentParser(description="Fetch the 4 reference models and export them to ONNX.")
    ap.add_argument("--force", action="store_true", help="redo download and export")
    ap.add_argument("--only", help="comma-separated slugs")
    args = ap.parse_args()
    dev = _paths.dev_dir()
    os.environ["HF_HOME"] = os.path.join(dev, "cache", "hf")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.makedirs(os.path.join(dev, "models", "reference"), exist_ok=True)
    mpath = os.path.join(dev, "models", "reference", "manifest.json")
    manifest = {}
    if os.path.isfile(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    only = set(args.only.split(",")) if args.only else None
    for slug, repo, prefix, task, ein, eout in MODELS:
        if only and slug not in only:
            continue
        process(slug, repo, prefix, task, ein, eout, dev, manifest, args.force)
        with open(mpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
    print("done; manifest:", mpath)


if __name__ == "__main__":
    main()
