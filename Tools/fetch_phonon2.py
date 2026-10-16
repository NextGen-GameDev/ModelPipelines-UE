"""One command: download Phonon-2 (pinned) + the base-model config files, verify, unpack, export fp32 and build a variant
into a folder you choose (WP9S.6c, contract C36: needs no dev folder when --out is given).

  python fetch_phonon2.py --out DIR [--variant fp16w|fp16|int8|fp32] [--cache HF_CACHE_DIR] [--keep-fp32]

Result: DIR/phonon-2-<variant>/ (encoder.onnx, decoder_joint.onnx, tokenizer/config files, SOURCE.json, NOTICE, licence
text, modelpipelines.json). DIR/_work/ (container + fp32 export) is deleted at the end unless --keep-fp32.
Licence: Phonon-2 weights are CC-BY-4.0 (attribution kept in NOTICE and the sidecar).
Disk: peak about 4 GB (177 MB download + 2.5 GB fp32 + variant). RAM: peak about 6.5 GB (the fp32 export).
Run with the pipelines-stt env (torch, transformers 5.x, onnx, zstandard, huggingface_hub).
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHONON_ID = "FermionResearch/Phonon-2"
PHONON_REV = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
BASE_ID = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REV = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
CONTAINER_SHA256 = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"
# BASE_COPY of phonon2_export.py (+ README for the licence record). Never the 2.5 GB weights.
BASE_FILES = ["tokenizer.json", "tokenizer_config.json", "processor_config.json", "generation_config.json",
              "config.json", "README.md"]
FORBIDDEN_PATTERNS = ("*.safetensors", "*.nemo", "*.bin")


class FetchError(Exception):
    pass


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def parse_sha256sums(text):
    """'<hex>  <name>' or '<hex> *<name>' per line -> {name: hex}. Blank lines and '#' comments are skipped."""
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise FetchError("bad SHA256SUMS line: %r" % ln)
        name = parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        out[name.replace("\\", "/")] = parts[0].lower()
    return out


def verify_sums(folder, sums_name="SHA256SUMS.txt"):
    """Verify every file listed in SHA256SUMS.txt; raises FetchError naming the first bad/missing file."""
    with open(os.path.join(folder, sums_name), encoding="utf-8") as f:
        sums = parse_sha256sums(f.read())
    for name, want in sums.items():
        fp = os.path.join(folder, *name.split("/"))
        if not os.path.isfile(fp):
            raise FetchError("file listed in %s is missing: %s" % (sums_name, name))
        got = sha256_file(fp)
        if got != want:
            raise FetchError("sha256 mismatch: %s (expected %s, got %s)" % (name, want, got))
    return sorted(sums)


def check_member(name, dest):
    """Refuse absolute paths, drive letters and '..'. Returns the destination path inside dest."""
    n = name.replace("\\", "/")
    if n.startswith("/") or (len(n) > 1 and n[1] == ":") or ".." in n.split("/"):
        raise FetchError("unsafe archive member: %r" % name)
    target = os.path.abspath(os.path.join(dest, *n.split("/")))
    root = os.path.abspath(dest)
    if target != root and not target.startswith(root + os.sep):
        raise FetchError("unsafe archive member: %r" % name)
    return target


def safe_extract_tar(tf, dest):
    """Extract regular files and dirs only (no links/devices) from an open tarfile into dest."""
    os.makedirs(dest, exist_ok=True)
    names = []
    for m in tf:
        target = check_member(m.name, dest)
        if m.isdir():
            os.makedirs(target, exist_ok=True)
        elif m.isreg():
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as o:
                shutil.copyfileobj(src, o, 1 << 22)
            names.append(m.name)
        else:
            raise FetchError("unsupported archive member type (link/device): %r" % m.name)
    return names


def unpack_zst(archive, dest):
    import zstandard
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with open(archive, "rb") as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as r:
            with tarfile.open(fileobj=r, mode="r|") as tf:
                names = safe_extract_tar(tf, tmp)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return names


def base_allow_patterns():
    return list(BASE_FILES)


def default_download(repo, revision, allow_patterns, cache):
    """-> local snapshot folder. Uses huggingface_hub (the only network call of this tool)."""
    from huggingface_hub import snapshot_download
    kw = {"repo_id": repo, "revision": revision, "cache_dir": cache}
    if allow_patterns is not None:
        kw["allow_patterns"] = allow_patterns
    return snapshot_download(**kw)


def list_files(folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp):
            rows.append((f, os.path.getsize(fp)))
    return rows


def run(out, variant, cache=None, keep_fp32=False, download=None, export=None, log=print):
    """download(repo, rev, allow_patterns, cache) -> folder; export(variant, out_dir) -> exit code (default: the real
    export_phonon2_onnx.main with injected paths). Returns the variant folder."""
    from stt import phonon2_export as ex
    download = download or default_download
    out = os.path.abspath(out)
    final = os.path.join(out, "phonon-2-" + variant)
    if ex.up_to_date(final):
        log("up to date, skipping: %s" % final)
        return final
    os.makedirs(out, exist_ok=True)

    log("downloading %s @ %s ..." % (PHONON_ID, PHONON_REV[:8]))
    snap = download(PHONON_ID, PHONON_REV, None, cache)
    verified = verify_sums(snap)
    log("verified %d files from SHA256SUMS.txt" % len(verified))
    archives = [f for f in os.listdir(snap) if f.endswith(".tar.zst")]
    if len(archives) != 1:
        raise FetchError("expected exactly one .tar.zst in the Phonon-2 snapshot, found %s" % archives)

    log("downloading %s @ %s (config files only) ..." % (BASE_ID, BASE_REV[:8]))
    base = download(BASE_ID, BASE_REV, base_allow_patterns(), cache)
    for f in BASE_FILES:
        if not os.path.isfile(os.path.join(base, f)):
            raise FetchError("base-model file missing: " + f)

    work = os.path.join(out, "_work")
    unpacked = os.path.join(work, "phonon-2", "unpacked")
    container = os.path.join(unpacked, "model.fermion")
    if not (os.path.isfile(container) and sha256_file(container) == CONTAINER_SHA256):
        log("unpacking %s ..." % archives[0])
        unpack_zst(os.path.join(snap, archives[0]), unpacked)
    if not os.path.isfile(container):
        raise FetchError("model.fermion not found in the archive")
    got = sha256_file(container)
    if got != CONTAINER_SHA256:
        raise FetchError("model.fermion sha256 mismatch: %s" % got)

    ex.set_paths(container, base, snap, os.path.join(work))
    try:
        if export is None:
            import export_phonon2_onnx as eo

            def export(v, dest):
                return eo.main(["--variant", v, "--out", dest])
        if variant != "fp32":
            rc = export("fp32", os.path.join(work, "fp32"))
            if rc:
                raise FetchError("fp32 export failed (exit %s)" % rc)
        rc = export(variant, final)
        if rc:
            raise FetchError("%s export failed (exit %s)" % (variant, rc))
    finally:
        ex.clear_paths()
    for f in ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt", "SOURCE.json", "modelpipelines.json"):
        if not os.path.isfile(os.path.join(final, f)):
            raise FetchError("output folder lacks " + f)
    if not keep_fp32:
        shutil.rmtree(work, ignore_errors=True)
    return final


def main(argv=None, download=None, export=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (created); result in <out>/phonon-2-<variant>/")
    ap.add_argument("--variant", default="fp16w", choices=["fp16w", "fp16", "int8", "fp32"],
                    help="encoder variant to build (default fp16w, the shipping one)")
    ap.add_argument("--cache", default=None, help="Hugging Face cache folder (default: the default HF cache)")
    ap.add_argument("--keep-fp32", action="store_true", help="keep <out>/_work (fp32 export, ~2.5 GB, and the container)")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        final = run(a.out, a.variant, a.cache and os.path.abspath(a.cache), a.keep_fp32, download, export)
    except FetchError as e:
        print("ERROR: %s" % e)
        return 1
    except ImportError as e:
        print("NEEDS: package missing: %s (use the pipelines-stt env)" % e)
        return 2
    print("done: %s" % final)
    for name, size in list_files(final):
        print("  %-32s %14d bytes" % (name, size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
