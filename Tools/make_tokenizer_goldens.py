#!/usr/bin/env python
"""Generate tokenizer goldens (schema pipelines-tokenizer-goldens/1) from the Python
`tokenizers` package. Strings come from a JSON file: [{"id": "sNN", "text": "..."}]."""
import argparse
import hashlib
import json
import os
import platform
import sys

import tokenizers
from tokenizers import Tokenizer

SCHEMA = "pipelines-tokenizer-goldens/1"
SCRIPT = "Tools/make_tokenizer_goldens.py"

# (name, pad_id) ; other metadata comes from <models>/<name>/SOURCE.json
TOKENIZERS = [
    ("bert-base-uncased", "wordpiece", 0),
    ("gpt2", "bpe", 50256),
    ("t5-small", "sentencepiece-unigram", 0),
]

BATCHES = [
    ("b00", ["s00", "s03", "s07"], True, None),
    ("b01", ["s00", "s13", "s22", "s18"], True, 8),
    ("b02", ["s05", "s10", "s16"], False, 5),
    ("b03", ["s22"], True, None),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_batch(path, texts, add_special, max_length, pad_id):
    tok = Tokenizer.from_file(path)
    if max_length is not None:
        tok.enable_truncation(max_length)
    tok.enable_padding(direction="right", pad_id=pad_id, pad_token=tok.id_to_token(pad_id))
    encs = tok.encode_batch(texts, add_special_tokens=add_special)
    return [e.ids for e in encs], [e.attention_mask for e in encs]


def build(name, kind, pad_id, models_dir, out_dir, strings):
    mdir = os.path.join(models_dir, name)
    tpath = os.path.join(mdir, "tokenizer.json")
    with open(os.path.join(mdir, "SOURCE.json"), encoding="utf-8") as f:
        src = json.load(f)
    tok = Tokenizer.from_file(tpath)
    try:
        rel = os.path.relpath(tpath, out_dir).replace("\\", "/")
    except ValueError:  # different drives: no relative path exists
        rel = os.path.abspath(tpath).replace("\\", "/")
    cases = []
    for s in strings:
        text = s["text"]
        with_sp = tok.encode(text, add_special_tokens=True).ids
        without = tok.encode(text, add_special_tokens=False).ids
        cases.append({
            "id": s["id"],
            "text": text,
            "text_utf8_hex": text.encode("utf-8").hex(),
            "ids_with_special": with_sp,
            "ids_without_special": without,
            "decode_with_special_keep": tok.decode(with_sp, skip_special_tokens=False),
            "decode_with_special_skip": tok.decode(with_sp, skip_special_tokens=True),
        })
    by_id = {s["id"]: s["text"] for s in strings}
    batches = []
    for bid, case_ids, add_special, max_len in BATCHES:
        if not all(c in by_id for c in case_ids):
            continue  # strings file lacks these ids (e.g. custom strings)
        ids, mask = run_batch(tpath, [by_id[c] for c in case_ids], add_special, max_len, pad_id)
        batches.append({"id": bid, "case_ids": case_ids, "add_special_tokens": add_special,
                        "max_length": max_len, "pad_id": pad_id,
                        "input_ids": ids, "attention_mask": mask})
    return {
        "schema": SCHEMA,
        "tokenizer": {
            "name": name, "repo": src["repo"], "revision": src["revision"],
            "license": src["license"], "kind": kind, "file": rel,
            "file_sha256": sha256_file(tpath),
            "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
        },
        "generator": {"script": SCRIPT, "python_tokenizers": tokenizers.__version__,
                      "python": platform.python_version()},
        "cases": cases,
        "batches": batches,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="folder written by fetch_tokenizers.py")
    ap.add_argument("--strings", required=True, help="JSON array of {id, text}")
    ap.add_argument("--out", required=True, help="output folder for <name>.json")
    args = ap.parse_args(argv)
    with open(args.strings, encoding="utf-8") as f:
        strings = json.load(f)
    strings = sorted(strings, key=lambda s: s["id"])
    os.makedirs(args.out, exist_ok=True)
    for name, kind, pad_id in TOKENIZERS:
        g = build(name, kind, pad_id, args.models, args.out, strings)
        p = os.path.join(args.out, name + ".json")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(g, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("wrote %s (%d bytes)" % (p, os.path.getsize(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
