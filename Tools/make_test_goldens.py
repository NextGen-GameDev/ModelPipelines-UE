#!/usr/bin/env python
"""Generate goldens (schema pipelines-test-model-goldens/1) for the test ONNX models by
running them in onnxruntime (CPU) on seeded inputs."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

SCHEMA = "pipelines-test-model-goldens/1"
SCRIPT = "Tools/make_test_goldens.py"
FILES = ["affine_dynamic.onnx", "affine_static.onnx", "int_ops.onnx", "half_io.onnx"]
EDGE = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-3, -1e6, 1e6, 8.5e37]
ORT_TYPES = {"tensor(float)": "float32", "tensor(int64)": "int64",
             "tensor(int32)": "int32", "tensor(float16)": "float16"}
NP = {"float32": np.float32, "int64": np.int64, "int32": np.int32, "float16": np.float16}


def sha256_file(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            hh.update(c)
    return hh.hexdigest()


def _dims(shape):
    return [d if isinstance(d, int) else -1 for d in shape]


def _jsonable(arr, typ):
    flat = arr.reshape(-1)
    if typ == "float32":
        return [float(np.float32(v)) for v in flat]
    if typ == "float16":
        return [float(v) for v in flat]
    return [int(v) for v in flat]


def tensor(name, typ, arr):
    return {"name": name, "type": typ, "shape": [int(d) for d in arr.shape], "data": _jsonable(arr, typ)}


def gen_float(rng, n):
    edge = np.array(EDGE[:n], dtype=np.float32)
    rest = rng.uniform(-100, 100, size=n - len(edge)).astype(np.float32)
    return np.concatenate([edge, rest])


def gen_ids(rng, n):
    first = np.array([2 ** 40, -(2 ** 40)][:n], dtype=np.int64)
    rest = rng.integers(-1000, 1001, size=n - len(first)).astype(np.int64)
    return np.concatenate([first, rest])


def run_case(sess, cid, feeds):
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, {k: v for k, v in feeds})
    types = {o.name: ORT_TYPES[o.type] for o in sess.get_outputs()}
    return {"id": cid,
            "inputs": [tensor(k, ORT_TYPES[next(i.type for i in sess.get_inputs() if i.name == k)], v)
                       for k, v in feeds],
            "outputs": [tensor(n, types[n], o) for n, o in zip(names, outs)]}


def build(models_dir, seed, batches):
    rng = np.random.default_rng(seed)
    models = []
    for fn in FILES:
        path = os.path.join(models_dir, fn)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        cases = []
        if fn == "affine_dynamic.onnx":
            for b in batches:
                x = gen_float(rng, b * 4).reshape(b, 4)
                cases.append(run_case(sess, f"batch{b}", [("x", x)]))
        elif fn == "affine_static.onnx":
            x = gen_float(rng, 8).reshape(2, 4)
            cases.append(run_case(sess, "static2x4", [("x", x)]))
        elif fn == "int_ops.onnx":
            for cid, (b, s) in (("ids1x5", (1, 5)), ("ids2x3", (2, 3))):
                ids = gen_ids(rng, b * s).reshape(b, s)
                mask = rng.integers(0, 2, size=b * s).astype(np.int32).reshape(b, s)
                cases.append(run_case(sess, cid, [("ids", ids), ("mask", mask)]))
        models.append({
            "file": fn,
            "sha256": sha256_file(path),
            "inputs": [{"name": i.name, "type": ORT_TYPES[i.type], "shape": _dims(i.shape)}
                       for i in sess.get_inputs()],
            "outputs": [{"name": o.name, "type": ORT_TYPES[o.type], "shape": _dims(o.shape)}
                        for o in sess.get_outputs()],
            "cases": cases,
        })
    return {"schema": SCHEMA,
            "generator": {"script": SCRIPT, "onnxruntime": ort.__version__, "onnx": onnx.__version__,
                          "numpy": np.__version__, "seed": seed, "batches": list(batches)},
            "models": models}


def main():
    dev = _paths.dev_dir()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=os.path.join(dev, "models", "test"),
                    help="folder with the test models (default <dev>/models/test)")
    ap.add_argument("--out", default=os.path.join(dev, "goldens", "test_models", "test_models.json"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batches", default="1,3,8,256", help="comma-separated batch sizes")
    a = ap.parse_args()
    try:
        batches = [int(t) for t in a.batches.split(",") if t.strip()]
    except ValueError:
        sys.exit("error: --batches must be comma-separated integers")
    if not batches or any(b < 1 for b in batches):
        sys.exit("error: --batches needs positive integers")
    missing = [f for f in FILES if not os.path.isfile(os.path.join(a.models, f))]
    if missing:
        sys.exit(f"error: missing models in {a.models}: {', '.join(missing)}. Run make_test_models.py first.")
    doc = build(a.models, a.seed, batches)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n = sum(len(m["cases"]) for m in doc["models"])
    print(f"wrote {a.out} ({len(doc['models'])} models, {n} cases)")


if __name__ == "__main__":
    main()
