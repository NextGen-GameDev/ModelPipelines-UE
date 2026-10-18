#!/usr/bin/env python
"""Sets up the English G2P bench under <dev>/tts/ (TTS exploration, see Tools/tts/README.md).

1. Clones larroy/kokoro.cpp at a pinned commit into <dev>/tts/kokoro.cpp (MIT; G2P data CMUdict BSD-style,
   g2p_en weights Apache-2.0).
2. Builds en_dump and misaki_dump with CMake into <dev>/tts/build.
3. Converts misaki's us_gold/us_silver JSON (Apache-2.0, from the installed misaki package) to
   <dev>/tts/misaki_us.tsv: word<TAB>tier<TAB>phonemes; homographs keep only DEFAULT (the C++ port has no
   POS tagger); capitalization growth matches misaki's Lexicon.grow_dictionary.
4. --with-model: downloads Kokoro v1.0 ONNX + voices (Apache-2.0, kokoro-onnx GitHub release mirror of
   hexgrad/Kokoro-82M) into <dev>/models/kokoro/ and checks the pinned sha256. Only synth.py needs them.

Safe to rerun: each step is skipped when its output is already there (a model file whose sha256 matches).
"""
import argparse
import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: E402

KOKORO_CPP_URL = "https://github.com/larroy/kokoro.cpp"
KOKORO_CPP_COMMIT = "788e07961b65fe253d0de404704627a005920da1"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def tts_dir():
    return os.path.join(_paths.dev_dir(), "tts")


def model_dir():
    return os.path.join(_paths.dev_dir(), "models", "kokoro")


def exe(name):
    path = os.path.join(tts_dir(), "build", name)
    for candidate in (path, path + ".exe", os.path.join(tts_dir(), "build", "Release", name + ".exe")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} not built; run setup_g2p_bench.py")


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def clone():
    dst = os.path.join(tts_dir(), "kokoro.cpp")
    if not os.path.isdir(os.path.join(dst, ".git")):
        run(["git", "clone", "-q", KOKORO_CPP_URL, dst])
    run(["git", "-C", dst, "checkout", "-q", KOKORO_CPP_COMMIT])
    return dst


def build(kokoro_cpp):
    out = os.path.join(tts_dir(), "build")
    run(["cmake", "-S", os.path.join(HERE, "cpp"), "-B", out, f"-DKOKORO_CPP_DIR={kokoro_cpp}",
         "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", out, "--config", "Release"])


def grow(d):
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def make_lexicon():
    out = os.path.join(tts_dir(), "misaki_us.tsv")
    if os.path.exists(out):
        return
    from misaki import data
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for tier in ("gold", "silver"):
            with importlib.resources.files(data).joinpath(f"us_{tier}.json").open(encoding="utf-8") as r:
                for k, v in grow(json.load(r)).items():
                    ps = v["DEFAULT"] if isinstance(v, dict) else v
                    if ps:
                        f.write(f"{k}\t{tier[0]}\t{ps}\n")
    print("wrote", out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model():
    os.makedirs(model_dir(), exist_ok=True)
    for name, digest in MODEL_FILES.items():
        path = os.path.join(model_dir(), name)
        if os.path.exists(path) and sha256(path) == digest:
            continue
        print("downloading", name)
        urllib.request.urlretrieve(MODEL_URL + name, path + ".part")
        if sha256(path + ".part") != digest:
            raise SystemExit(f"sha256 mismatch for {name}")
        os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-model", action="store_true", help="also download Kokoro v1.0 ONNX + voices")
    args = ap.parse_args()
    os.makedirs(tts_dir(), exist_ok=True)
    build(clone())
    make_lexicon()
    if args.with_model:
        fetch_model()


if __name__ == "__main__":
    main()
