#!/usr/bin/env python
"""Build the 4 tiny test ONNX models (opset 17, IR 8) with the onnx helper API.
Output is deterministic (no timestamps)."""
import argparse
import hashlib
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as h, numpy_helper as nh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

PRODUCER = "ModelPipelines-Tools"


def _finish(graph):
    m = h.make_model(graph, producer_name=PRODUCER, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def _vi(name, elem, shape):
    return h.make_tensor_value_info(name, elem, shape)


def affine(name, shape):
    nodes = [
        h.make_node("Mul", ["x", "two"], ["x2"], name="mul"),
        h.make_node("Add", ["x2", "one"], ["y"], name="add"),
    ]
    inits = [nh.from_array(np.array(2.0, dtype=np.float32), "two"),
             nh.from_array(np.array(1.0, dtype=np.float32), "one")]
    g = h.make_graph(nodes, name, [_vi("x", TP.FLOAT, shape)], [_vi("y", TP.FLOAT, shape)], inits)
    return _finish(g)


def int_ops():
    nodes = [
        h.make_node("Add", ["ids", "one_i64"], ["ids_plus_one"], name="add_one"),
        h.make_node("Cast", ["mask"], ["mask_f"], name="cast", to=TP.FLOAT),
        h.make_node("Mul", ["mask_f", "half"], ["mask_half"], name="mul_half"),
    ]
    inits = [nh.from_array(np.array(1, dtype=np.int64), "one_i64"),
             nh.from_array(np.array(0.5, dtype=np.float32), "half")]
    shp = ["batch", "seq"]
    g = h.make_graph(nodes, "int_ops",
                     [_vi("ids", TP.INT64, shp), _vi("mask", TP.INT32, shp)],
                     [_vi("ids_plus_one", TP.INT64, shp), _vi("mask_half", TP.FLOAT, shp)], inits)
    return _finish(g)


def half_io():
    g = h.make_graph([h.make_node("Identity", ["h"], ["h_out"], name="identity")], "half_io",
                     [_vi("h", TP.FLOAT16, [2])], [_vi("h_out", TP.FLOAT16, [2])])
    return _finish(g)


def build_all():
    return {
        "affine_dynamic.onnx": affine("affine_dynamic", ["batch", 4]),
        "affine_static.onnx": affine("affine_static", [2, 4]),
        "int_ops.onnx": int_ops(),
        "half_io.onnx": half_io(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_paths.dev_dir(), "models", "test"),
                    help="output folder (default <dev>/models/test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn, m in build_all().items():
        data = m.SerializeToString(deterministic=True)
        p = os.path.join(a.out, fn)
        with open(p, "wb") as f:
            f.write(data)
        print(f"{fn}  {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
