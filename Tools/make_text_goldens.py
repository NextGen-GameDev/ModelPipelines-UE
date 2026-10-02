#!/usr/bin/env python
"""Text goldens for M3 (WP3.3, schema pipelines-text-goldens/1).

Reads a sentence file (schema pipelines-text-sentences/1) and writes, for four models (MiniLM and DistilBERT-SST2 with
a PyTorch reference plus an onnxruntime cross-check, and two tiny importer models with the onnxruntime result as the
reference), token ids, embeddings / logits / scores, and a top-5 similarity table for MiniLM.

Everything comes from Python (tokenizers, transformers/torch, onnxruntime on CPU). Nothing comes from Unreal.
Offline: weights are read from the HF cache <dev>/cache/hf at the revision pinned in the reference manifest.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

SCHEMA = "pipelines-text-goldens/1"
SENT_SCHEMA = "pipelines-text-sentences/1"
SCRIPT = "Tools/make_text_goldens.py"
TOOLS = os.path.dirname(os.path.abspath(__file__))

# slug, folder (under models root), task, reference, HF repo (None for the tiny models)
MODELS = [
    ("all-MiniLM-L6-v2", "reference/all-MiniLM-L6-v2", "TextEmbedding", "torch",
     "sentence-transformers/all-MiniLM-L6-v2"),
    ("distilbert-sst2", "reference/distilbert-sst2", "TextClassification", "torch",
     "distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
    ("tiny_text_embedding", "import_test/tiny_text_embedding", "TextEmbedding", "onnxruntime", None),
    ("tiny_text_classification", "import_test/tiny_text_classification", "TextClassification", "onnxruntime", None),
]
COS_MIN = 0.99999
LOGIT_DIFF_MAX = 1e-4
MARGIN_MIN = 0.01


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- pure helpers (unit tested) -------------------------------------------------------------------------------

def softmax64(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def sigmoid64(logits):
    x = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def activation(logits, mode):
    if mode == "Softmax":
        return softmax64(logits)
    if mode == "Sigmoid":
        return sigmoid64(logits)
    raise ValueError("unknown score_activation %r" % mode)


def top_and_margin(scores):
    """(argmax with ties to the lower index, best - second best)."""
    s = np.asarray(scores, dtype=np.float64)
    idx = int(np.argmax(s))  # numpy argmax returns the first (lowest) index on ties
    if s.size < 2:
        return idx, float(s[idx])
    rest = np.delete(s, idx)
    return idx, float(s[idx] - np.max(rest))


def pool_np(hidden, mask, mode, normalize):
    """hidden [T, D] float32, mask [T] ints; returns float32 [D]. Mean over the mask in float64, then cast."""
    h = np.asarray(hidden, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if mode == "Mean":
        v = (h * m[:, None]).sum(axis=0) / max(m.sum(), 1e-9)
    elif mode == "CLS":
        v = h[0]
    else:
        raise ValueError("unsupported pooling %r" % mode)
    if normalize:
        v = v / max(np.linalg.norm(v), 1e-12)
    return v.astype(np.float32)


def cosine64(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_top(embs, query_idx, k):
    """k best corpus entries for one query: float64 cosine, ties to the lower index."""
    scored = [(cosine64(embs[query_idx], e), i) for i, e in enumerate(embs)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"index": i, "score": s} for s, i in scored[:k]]


# ---- package derivation ---------------------------------------------------------------------------------------

def derive_package(folder_abs, task):
    cfg = read_json(os.path.join(folder_abs, "config.json"))
    max_pos = cfg.get("max_position_embeddings")
    limit = None
    sb = os.path.join(folder_abs, "sentence_bert_config.json")
    if os.path.isfile(sb):
        limit = read_json(sb).get("max_seq_length")
    if limit is None:
        tc = os.path.join(folder_abs, "tokenizer_config.json")
        if os.path.isfile(tc):
            v = read_json(tc).get("model_max_length")
            if isinstance(v, (int, float)) and 0 < v < 1e9:  # the 1e30 "unset" sentinel is ignored
                limit = int(v)
    if limit is None and max_pos is None:
        raise SystemExit("FAIL: %s: cannot derive max_tokens" % folder_abs)
    max_tokens = int(min(x for x in (limit, max_pos) if x is not None))
    pooling = "Mean"
    pc = os.path.join(folder_abs, "1_Pooling", "config.json")
    if os.path.isfile(pc):
        p = read_json(pc)
        if p.get("pooling_mode_cls_token"):
            pooling = "CLS"
        elif p.get("pooling_mode_mean_tokens"):
            pooling = "Mean"
        else:
            raise SystemExit("FAIL: %s: unsupported pooling mode in 1_Pooling/config.json" % folder_abs)
    normalize = False
    mp = os.path.join(folder_abs, "modules.json")
    if os.path.isfile(mp):
        normalize = any(m.get("type", "").endswith("Normalize") for m in read_json(mp))
    labels = []
    activation_name = "Softmax"
    if task == "TextClassification":
        id2label = cfg.get("id2label") or {}
        labels = [id2label[k] for k in sorted(id2label, key=int)]
        if cfg.get("problem_type") == "multi_label_classification":
            activation_name = "Sigmoid"
    return {"max_tokens": max_tokens, "add_special_tokens": True, "pooling": pooling,
            "normalize_embeddings": normalize, "score_activation": activation_name, "labels": labels}


def expected_package(expectations, folder):
    for c in expectations["cases"]:
        if c.get("folder") == folder:
            p = c["preprocess"]
            return {"max_tokens": p["max_tokens"], "add_special_tokens": p["add_special_tokens"],
                    "pooling": p["pooling"], "normalize_embeddings": p["normalize_embeddings"],
                    "score_activation": p["score_activation"], "labels": list(c.get("labels", []))}
    raise SystemExit("FAIL: no import expectation case for folder %s" % folder)


# ---- model runs -----------------------------------------------------------------------------------------------

def encode_all(tok_path, texts, pkg):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    tok.no_padding()
    tok.enable_truncation(max_length=pkg["max_tokens"])
    return [tok.encode(t, add_special_tokens=pkg["add_special_tokens"]).ids for t in texts]


def ort_run(onnx_path, ids_list):
    """All sentences in one right-padded batch (pad id 0, mask 0). Returns (output array, input names)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    n = len(ids_list)
    t = max(len(x) for x in ids_list)
    ids = np.zeros((n, t), dtype=np.int64)
    mask = np.zeros((n, t), dtype=np.int64)
    for r, x in enumerate(ids_list):
        ids[r, :len(x)] = x
        mask[r, :len(x)] = 1
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    out = sess.run(None, {k: feed[k] for k in names})[0]
    return out, names, sess


def ort_single(sess, names, ids):
    a = np.asarray([ids], dtype=np.int64)
    feed = {"input_ids": a, "attention_mask": np.ones_like(a), "token_type_ids": np.zeros_like(a)}
    return sess.run(None, {k: feed[k] for k in names})[0]


def load_torch(repo, revision, task, cache):
    import torch
    import transformers
    torch.set_num_threads(1)
    cls = transformers.AutoModel if task == "TextEmbedding" else transformers.AutoModelForSequenceClassification
    model = cls.from_pretrained(repo, revision=revision, cache_dir=cache, local_files_only=True,
                                dtype=torch.float32)
    return model.eval()


def torch_forward(model, ids, with_type_ids):
    import torch
    a = torch.tensor([ids], dtype=torch.long)
    kw = {"input_ids": a, "attention_mask": torch.ones_like(a)}
    if with_type_ids:
        kw["token_type_ids"] = torch.zeros_like(a)
    with torch.no_grad():
        return model(**kw)


def torch_embedding(model, ids, pkg, with_type_ids):
    import torch
    h = torch_forward(model, ids, with_type_ids).last_hidden_state  # [1, T, D]
    mask = torch.ones(1, h.shape[1], 1, dtype=h.dtype)
    if pkg["pooling"] == "Mean":
        v = (h * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
    elif pkg["pooling"] == "CLS":
        v = h[:, 0]
    else:
        raise SystemExit("FAIL: unsupported pooling " + pkg["pooling"])
    if pkg["normalize_embeddings"]:
        v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v[0].numpy()


def torch_logits(model, ids, with_type_ids):
    return torch_forward(model, ids, with_type_ids).logits[0].numpy()


def ort_embedding_rows(out, ids_list, pkg):
    return [pool_np(out[r, :len(x)], [1] * len(x), pkg["pooling"], pkg["normalize_embeddings"])
            for r, x in enumerate(ids_list)]


def class_fields(logits32, pkg):
    scores = activation(logits32, pkg["score_activation"])
    idx, margin = top_and_margin(scores)
    labels = pkg["labels"]
    return {"logits": [float(v) for v in logits32], "scores": [float(v) for v in scores],
            "label": labels[idx] if idx < len(labels) else "", "index": idx, "top_margin": margin}


def build_model(slug, folder, task, reference, repo, models_root, sentences, expectations, manifest, cache):
    folder_abs = os.path.join(models_root, folder)
    onnx_path = os.path.join(folder_abs, "model.onnx")
    tok_path = os.path.join(folder_abs, "tokenizer.json")
    pkg = derive_package(folder_abs, task)
    exp = expected_package(expectations, folder)
    if pkg != exp:
        raise SystemExit("FAIL: %s: package derived from the folder files differs from the import expectations\n"
                         "  derived : %s\n  expected: %s" % (slug, json.dumps(pkg), json.dumps(exp)))
    texts = [s["text"] for s in sentences]
    ids_list = encode_all(tok_path, texts, pkg)
    out, names, sess = ort_run(onnx_path, ids_list)
    with_type_ids = "token_type_ids" in names
    cases = []
    stats = {"worst_cos": 1.0, "worst_diff": 0.0, "min_margin": None, "truncated": 0, "padded_vs_single": None}
    revision = None
    lic = None
    if repo is not None:
        revision = manifest[slug]["revision"]
        lic = manifest[slug]["license"]
        model = load_torch(repo, revision, task, cache)
    for r, ids in enumerate(ids_list):
        case = {"sentence_id": sentences[r]["id"], "input_ids": ids}
        if len(ids) == pkg["max_tokens"]:
            stats["truncated"] += 1
        cases.append(case)
    if task == "TextEmbedding":
        ort_embs = ort_embedding_rows(out, ids_list, pkg)
        for r, ids in enumerate(ids_list):
            if repo is not None:
                ref = torch_embedding(model, ids, pkg, with_type_ids)
                cos = cosine64(ort_embs[r], ref)
                diff = float(np.max(np.abs(ort_embs[r].astype(np.float64) - ref.astype(np.float64))))
            else:
                ref = ort_embs[r]
                cos, diff = 1.0, 0.0
                single = pool_np(ort_single(sess, names, ids)[0], [1] * len(ids), pkg["pooling"],
                                 pkg["normalize_embeddings"])
                d = float(np.max(np.abs(single.astype(np.float64) - ref.astype(np.float64))))
                stats["padded_vs_single"] = max(stats["padded_vs_single"] or 0.0, d)
            cases[r]["embedding"] = [float(v) for v in ref]
            cases[r]["ort_cosine"] = cos
            cases[r]["ort_max_abs_diff"] = diff
            stats["worst_cos"] = min(stats["worst_cos"], cos)
            stats["worst_diff"] = max(stats["worst_diff"], diff)
    else:
        for r, ids in enumerate(ids_list):
            ort_logits = out[r].astype(np.float32)
            if repo is not None:
                ref = torch_logits(model, ids, with_type_ids).astype(np.float32)
                diff = float(np.max(np.abs(ort_logits.astype(np.float64) - ref.astype(np.float64))))
            else:
                ref = ort_logits
                diff = 0.0
                single = ort_single(sess, names, ids)[0].astype(np.float32)
                d = float(np.max(np.abs(single.astype(np.float64) - ref.astype(np.float64))))
                stats["padded_vs_single"] = max(stats["padded_vs_single"] or 0.0, d)
            cases[r].update(class_fields(ref, pkg))
            cases[r]["ort_max_abs_diff"] = diff
            stats["worst_diff"] = max(stats["worst_diff"], diff)
            m = cases[r]["top_margin"]
            stats["min_margin"] = m if stats["min_margin"] is None else min(stats["min_margin"], m)
    entry = {"slug": slug, "folder": folder, "task": task, "reference": reference, "package": pkg,
             "source": {"repo": repo, "revision": revision, "license": lic, "onnx_sha256": sha256_file(onnx_path),
                        "tokenizer_json_sha256": sha256_file(tok_path),
                        "tokenization": "tokenizers.Tokenizer.from_file, no_padding, enable_truncation(max_length="
                                        "max_tokens; longest_first, stride 0, right), encode(add_special_tokens)",
                        "torch_run": "batch 1, unpadded, fp32, eval, no_grad" if repo else None,
                        "ort_run": "CPUExecutionProvider, one right-padded batch (pad id 0, mask 0), int64, "
                                   "1 intra-op thread"},
             "cases": cases}
    if slug == "all-MiniLM-L6-v2":
        embs = [np.asarray(c["embedding"], dtype=np.float32) for c in cases]
        k = min(5, len(embs))
        entry["similarity"] = {"k": k, "queries": [
            {"sentence_id": sentences[q]["id"], "top": similarity_top(embs, q, k)} for q in range(min(5, len(embs)))]}
    return entry, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    dev = _paths.dev_dir()
    ap.add_argument("--sentences", default=os.path.join(TOOLS, "data", "text_sentences.json"),
                    help="sentence file (schema %s), at least 5 entries" % SENT_SCHEMA)
    ap.add_argument("--out", default=os.path.join(dev, "goldens", "text", "text_goldens.json"),
                    help="output goldens JSON")
    ap.add_argument("--variant", default="visible", help="written to generator.variant")
    args = ap.parse_args()

    t0 = time.time()
    os.environ["HF_HUB_OFFLINE"] = "1"
    models_root = os.path.join(dev, "models")
    cache = os.path.join(dev, "cache", "hf")
    os.environ["HF_HOME"] = cache
    sdoc = read_json(args.sentences)
    if sdoc.get("schema") != SENT_SCHEMA:
        raise SystemExit("FAIL: %s is not schema %s" % (args.sentences, SENT_SCHEMA))
    sentences = sdoc["sentences"]
    if len(sentences) < 5:
        raise SystemExit("FAIL: need at least 5 sentences, got %d" % len(sentences))
    manifest = read_json(os.path.join(models_root, "reference", "manifest.json"))
    expectations = read_json(os.path.join(dev, "goldens", "import", "import_expectations.json"))

    import onnxruntime
    import tokenizers
    import torch
    import transformers
    gen = {"script": SCRIPT, "variant": args.variant, "python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
           "onnxruntime": onnxruntime.__version__, "numpy": np.__version__}

    models, failures = [], []
    for slug, folder, task, reference, repo in MODELS:
        entry, st = build_model(slug, folder, task, reference, repo, models_root, sentences, expectations, manifest,
                                cache)
        models.append(entry)
        line = "%-26s ref=%-11s worst ort_max_abs_diff=%.3e" % (slug, reference, st["worst_diff"])
        if task == "TextEmbedding":
            line += "  worst ort_cosine=%.9f" % st["worst_cos"]
        if st["min_margin"] is not None:
            line += "  min top_margin=%.6f" % st["min_margin"]
        if st["padded_vs_single"] is not None:
            line += "  (padded-vs-single ORT diff %.3e)" % st["padded_vs_single"]
        line += "  truncated=%d/%d (max_tokens=%d)" % (st["truncated"], len(sentences), entry["package"]["max_tokens"])
        print(line)
        if reference == "torch" and task == "TextEmbedding" and st["worst_cos"] < COS_MIN:
            failures.append("%s: ort_cosine %.9f < %g" % (slug, st["worst_cos"], COS_MIN))
        if reference == "torch" and task == "TextClassification" and st["worst_diff"] > LOGIT_DIFF_MAX:
            failures.append("%s: ort_max_abs_diff %.3e > %g" % (slug, st["worst_diff"], LOGIT_DIFF_MAX))
        if slug == "distilbert-sst2" and st["min_margin"] < MARGIN_MIN:
            bad = [c["sentence_id"] for c in entry["cases"] if c["top_margin"] < MARGIN_MIN]
            failures.append("%s: top_margin < %g for %s" % (slug, MARGIN_MIN, ", ".join(bad)))
    if failures:
        for f in failures:
            print("FAIL: " + f)
        return 1

    doc = {"schema": SCHEMA, "generator": gen, "models_root": os.path.abspath(models_root).replace("\\", "/"),
           "sentences": [{"id": s["id"], "text": s["text"], "text_utf8_hex": s["text"].encode("utf-8").hex()}
                         for s in sentences],
           "models": models}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s (%d bytes, sha256 %s) in %.1f s" % (args.out, os.path.getsize(args.out), sha256_file(args.out),
                                                         time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
