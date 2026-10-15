#!/usr/bin/env python
"""Build the tiny test ONNX models (opset 17, IR 8) with the onnx helper API.
Output is deterministic (no timestamps).

Models: affine_dynamic, affine_static, int_ops, half_io (C10) and the tiny speech-to-text
graphs (C72c) that mimic the Phonon graph names with toy sizes:

  tiny_stt_encoder.onnx        input_features f32 [1,frames,8] -> encoder_projected f32 [1,frames,4]
                               encoder_projected = MatMul(input_features, W_enc)   (W_enc 8x4 fixed)
  tiny_stt_decoder_joint.onnx  inputs encoder_frame f32 [1,4], token i64 [1],
                               state_h f32 [2,1,4], state_c f32 [2,1,4]
                               token_logits    = MatMul(encoder_frame, W1) + Gather(E, token)
                                                 (W1 4x6, E 6x6; Gather picks row `token` of E,
                                                  giving [1,6]; token ids 0..5, 5 = blank)
                               duration_logits = MatMul(encoder_frame, W3)          (W3 4x3 -> [1,3])
                               state_h_out     = state_h + 1
                               state_c_out     = state_c * 0.5
  tiny_stt_bad_decoder.onnx    same math but `token` is float32 [1]; a Cast to int64 inside.
                               Valid ONNX, wrong input type for the C70 contract.

All weights are fixed formulas below (no randomness)."""
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


# ---- tiny speech-to-text graphs (C72c) -------------------------------------------------
W_ENC = np.array([[((i * 4 + j) % 7 - 3) * 0.25 for j in range(4)] for i in range(8)], dtype=np.float32)
W1 = np.array([[2, 0, 0, 0, 0, 0.5],
               [0, 2, 0, 0, 0, 0.5],
               [0, 0, 2, 0, 1, 0.5],
               [0, 0, 0, 2, 1, 0.5]], dtype=np.float32)
E = np.zeros((6, 6), dtype=np.float32)
E[np.arange(6), np.arange(6)] = 0.25   # repeat bias
E[0, 5] = 3.0                          # after token 0, blank wins (forces d 0 -> 1 on the same frame)
E[1, 2] = 0.5
E[2, 3] = 0.5
E[4, 0] = 0.5
W3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]], dtype=np.float32)


def tiny_stt_encoder():
    nodes = [h.make_node("MatMul", ["input_features", "W_enc"], ["encoder_projected"], name="proj")]
    g = h.make_graph(nodes, "tiny_stt_encoder",
                     [_vi("input_features", TP.FLOAT, [1, "frames", 8])],
                     [_vi("encoder_projected", TP.FLOAT, [1, "frames", 4])],
                     [nh.from_array(W_ENC, "W_enc")])
    return _finish(g)


def tiny_stt_decoder(float_token):
    nodes = []
    tok = "token"
    if float_token:
        nodes.append(h.make_node("Cast", ["token"], ["token_i64"], name="cast_token", to=TP.INT64))
        tok = "token_i64"
    nodes += [
        h.make_node("MatMul", ["encoder_frame", "W1"], ["enc_logits"], name="mm1"),
        h.make_node("Gather", ["E", tok], ["tok_row"], name="gather", axis=0),
        h.make_node("Add", ["enc_logits", "tok_row"], ["token_logits"], name="add_logits"),
        h.make_node("MatMul", ["encoder_frame", "W3"], ["duration_logits"], name="mm3"),
        h.make_node("Add", ["state_h", "one"], ["state_h_out"], name="inc_h"),
        h.make_node("Mul", ["state_c", "half"], ["state_c_out"], name="half_c"),
    ]
    inits = [nh.from_array(W1, "W1"), nh.from_array(E, "E"), nh.from_array(W3, "W3"),
             nh.from_array(np.array(1.0, dtype=np.float32), "one"),
             nh.from_array(np.array(0.5, dtype=np.float32), "half")]
    name = "tiny_stt_bad_decoder" if float_token else "tiny_stt_decoder_joint"
    g = h.make_graph(nodes, name,
                     [_vi("encoder_frame", TP.FLOAT, [1, 4]),
                      _vi("token", TP.FLOAT if float_token else TP.INT64, [1]),
                      _vi("state_h", TP.FLOAT, [2, 1, 4]), _vi("state_c", TP.FLOAT, [2, 1, 4])],
                     [_vi("token_logits", TP.FLOAT, [1, 6]), _vi("duration_logits", TP.FLOAT, [1, 3]),
                      _vi("state_h_out", TP.FLOAT, [2, 1, 4]), _vi("state_c_out", TP.FLOAT, [2, 1, 4])],
                     inits)
    return _finish(g)


def build_all():
    return {
        "affine_dynamic.onnx": affine("affine_dynamic", ["batch", 4]),
        "affine_static.onnx": affine("affine_static", [2, 4]),
        "int_ops.onnx": int_ops(),
        "half_io.onnx": half_io(),
        "tiny_stt_encoder.onnx": tiny_stt_encoder(),
        "tiny_stt_decoder_joint.onnx": tiny_stt_decoder(False),
        "tiny_stt_bad_decoder.onnx": tiny_stt_decoder(True),
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
