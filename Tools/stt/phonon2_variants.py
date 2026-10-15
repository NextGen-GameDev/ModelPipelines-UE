"""Phonon-2 encoder shipping variants (WP9S.1b): fp16w, int8, fp16. All build from the fp32 export folder
(fp16 from the torch model), single encoder.onnx < 2 GB with float32 I/O. decoder_joint.onnx is copied unchanged.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from stt import phonon2_export as ex

VARIANTS = ("fp16w", "int8", "fp16")
MIN_ELEMS = 1024
INT8_ARGS = {"weight_type": "QInt8", "per_channel": True, "op_types_to_quantize": ["MatMul"]}
LIMIT = 2_000_000_000


def convert_fp16_weights(model, min_elems=MIN_ELEMS):
    """In place: each float32 initializer with >= min_elems elements becomes float16 plus a Cast(to=FLOAT) node
    that produces the original name, so every consumer reads the cast output. Returns (converted, overflow)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    g = model.graph
    casts, keep, converted, overflow = [], [], 0, 0
    for init in g.initializer:
        n = int(np.prod(init.dims)) if len(init.dims) else 1
        if init.data_type == TensorProto.FLOAT and n >= min_elems:
            arr = numpy_helper.to_array(init)
            overflow += int((np.abs(arr) > 65504).sum())
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], name=init.name + "_cast_fp32",
                                          to=TensorProto.FLOAT))
            converted += 1
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)        # casts have only initializer inputs: valid at the front
    return converted, overflow


def build_fp16w(fp32_dir, out_dir):
    import onnx
    m = onnx.load(os.path.join(fp32_dir, "encoder.onnx"), load_external_data=True)
    conv, over = convert_fp16_weights(m)
    print("fp16w: converted %d initializers, %d values beyond fp16 range" % (conv, over))
    onnx.save_model(m, os.path.join(out_dir, "encoder.onnx"))
    return {"initializers_converted": conv, "values_over_fp16_max": over, "min_elements": MIN_ELEMS}


def build_int8(fp32_dir, out_dir):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src = os.path.join(fp32_dir, "encoder.onnx")
    quantize_dynamic(src, os.path.join(out_dir, "encoder.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])
    return {"function": "onnxruntime.quantization.quantize_dynamic", "args": INT8_ARGS}


def build_fp16(model, out_dir):
    """Torch model in half precision; float32 I/O via Cast at the graph edges."""
    import torch
    enc, _ = ex.make_wrappers(model)

    class Half(torch.nn.Module):
        def __init__(self, e):
            super().__init__()
            self.e = e.half()

        def forward(self, x):
            return self.e(x.half()).float()
    h = Half(enc).eval()
    x = torch.randn(1, 801, 128)
    with torch.no_grad():
        torch.onnx.export(h, (x,), os.path.join(out_dir, "encoder.onnx"), opset_version=ex.OPSET, dynamo=False,
                          input_names=["input_features"], output_names=["encoder_projected"],
                          dynamic_axes={"input_features": {1: "T"}, "encoder_projected": {1: "F"}})
    import onnx
    m = onnx.load(os.path.join(out_dir, "encoder.onnx"))
    for vi in list(m.graph.input) + list(m.graph.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.Clear()
        d0.dim_value = 1
    onnx.save_model(m, os.path.join(out_dir, "encoder.onnx"))
    return {"note": "model.half() export; float32 I/O via Cast"}


def build_variant(variant, fp32_dir, out_dir, model=None):
    """Write the variant folder into out_dir (must exist). Returns the build-notes dict."""
    if variant == "fp16w":
        notes = build_fp16w(fp32_dir, out_dir)
    elif variant == "int8":
        notes = build_int8(fp32_dir, out_dir)
    elif variant == "fp16":
        notes = build_fp16(model, out_dir)
    else:
        raise ValueError(variant)
    shutil.copyfile(os.path.join(fp32_dir, "decoder_joint.onnx"), os.path.join(out_dir, "decoder_joint.onnx"))
    for f in ex.BASE_COPY + ex.PHONON_COPY:
        shutil.copyfile(os.path.join(fp32_dir, f), os.path.join(out_dir, f))
    return notes


def check_s1(folder):
    """S1 pieces: returns dict of booleans/size. Raises on checker or C70 failure."""
    enc = os.path.join(folder, "encoder.onnx")
    size = os.path.getsize(enc)
    import onnx
    ext = [i.name for i in onnx.load(enc, load_external_data=False).graph.initializer
           if any(e.key == "location" for e in i.external_data)]
    ex.check_onnx(folder)
    ex.assert_c70(folder)
    return {"bytes": size, "under_2GB": size < LIMIT, "no_external_data": not ext
            and not os.path.exists(enc + ".data"), "s1": size < LIMIT and not ext}


def write_variant_source(folder, variant, fp32_dir, notes):
    src = ex.write_source_json(folder, variant)
    src["derived_from"] = {"variant": "fp32", "encoder_sha256": ex.sha256_file(os.path.join(fp32_dir, "encoder.onnx")),
                           "encoder_data_sha256": ex.sha256_file(os.path.join(fp32_dir, "encoder.onnx.data"))}
    src["build"] = notes
    with open(os.path.join(folder, "SOURCE.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(src, f, indent=1)
        f.write("\n")
    return src
