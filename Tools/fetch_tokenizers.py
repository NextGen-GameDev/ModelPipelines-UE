#!/usr/bin/env python
"""Fetch the pinned tokenizer files for the M0 parity goldens.

Checks each model's license on its Hugging Face card before downloading anything.
Only the listed small files are downloaded, never weights.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

MODELS = [
    {"name": "bert-base-uncased", "repo": "google-bert/bert-base-uncased",
     "revision": "86b5e0934494bd15c9632b12f734a8a67f723594", "license": "apache-2.0",
     "kind": "wordpiece", "files": ["tokenizer.json", "tokenizer_config.json", "LICENSE"]},
    {"name": "gpt2", "repo": "openai-community/gpt2",
     "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "license": "mit",
     "kind": "bpe", "files": ["tokenizer.json", "tokenizer_config.json"]},
    {"name": "t5-small", "repo": "google-t5/t5-small",
     "revision": "df1b051c49625cf57a3d0d8d3863ed4d13564fe4", "license": "apache-2.0",
     "kind": "sentencepiece-unigram", "files": ["tokenizer.json", "tokenizer_config.json"]},
]

ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}


def fetch_api(repo, revision):
    """Return the parsed model-info JSON for repo at revision (network)."""
    url = "https://huggingface.co/api/models/%s/revision/%s" % (repo, revision)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download_file(repo, revision, filename):
    """Return the bytes of one file at a pinned revision (network)."""
    url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, filename)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def license_of(info):
    card = info.get("cardData") or {}
    lic = card.get("license")
    if lic is None:
        for t in info.get("tags") or []:
            if t.startswith("license:"):
                lic = t[len("license:"):]
    return lic


def fetch_model(m, out_dir):
    """Return (ok, downloaded_count)."""
    info = fetch_api(m["repo"], m["revision"])
    lic = license_of(info)
    if lic != m["license"] or lic not in ALLOWED_LICENSES:
        print("REFUSED %s: license %r (expected %r, allowed %s)"
              % (m["name"], lic, m["license"], sorted(ALLOWED_LICENSES)), file=sys.stderr)
        return False, 0
    if info.get("sha") and info["sha"] != m["revision"]:
        print("REFUSED %s: API sha %s != pinned %s" % (m["name"], info["sha"], m["revision"]),
              file=sys.stderr)
        return False, 0
    dest = os.path.join(out_dir, m["name"])
    src_path = os.path.join(dest, "SOURCE.json")
    known = {}
    if os.path.isfile(src_path):
        try:
            with open(src_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("revision") == m["revision"]:
                known = prev.get("files", {})
        except (OSError, ValueError):
            known = {}
    os.makedirs(dest, exist_ok=True)
    shas = {}
    downloaded = 0
    for fn in m["files"]:
        p = os.path.join(dest, fn)
        if os.path.isfile(p) and known.get(fn) == sha256_file(p):
            shas[fn] = known[fn]
            continue
        data = download_file(m["repo"], m["revision"], fn)
        with open(p, "wb") as f:
            f.write(data)
        shas[fn] = hashlib.sha256(data).hexdigest()
        downloaded += 1
    with open(src_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": m["repo"], "revision": m["revision"], "license": lic,
                   "files": shas}, f, indent=1)
        f.write("\n")
    print("%s: %d downloaded, %d already present" % (m["name"], downloaded, len(m["files"]) - downloaded))
    return True, downloaded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output folder (one subfolder per model)")
    args = ap.parse_args(argv)
    total = 0
    failed = False
    for m in MODELS:
        ok, n = fetch_model(m, args.out)
        failed |= not ok
        total += n
    if failed:
        return 1
    if total == 0:
        print("Nothing downloaded: all files already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
