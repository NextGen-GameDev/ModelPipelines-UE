"""Phonon-2 container -> torch ParakeetForTDT -> two ONNX graphs (encoder, one-step decoder+joint). WP9S.1a.

The container is expanded to fp32 exactly like the vendor's reference_transformers.py and loaded into stock
`ParakeetForTDT` (strict). Contract: docs/contracts.md C70. Heavy imports (torch, transformers, onnx) are lazy so
the pure helpers can be imported without them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import _paths  # noqa: E402
from stt.fermion_container import read_container  # noqa: E402

MODEL_ID = "FermionResearch/Phonon-2"
REVISION = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
CONTAINER_SHA256 = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"
BASE_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
OPSET = 17
VOCAB = 8193       # tokens incl. blank (id 8192)
BLANK = 8192
NUM_DUR = 5
HIDDEN = 640
BASE_COPY = ["tokenizer.json", "tokenizer_config.json", "processor_config.json", "generation_config.json", "config.json"]
PHONON_COPY = ["NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt"]


_OVERRIDE = {}


def set_paths(container, base_dir, phonon_dir, out_root):
    """WP9S.6c: inject the input paths (user fetch tool). While set, dev_paths() never touches the dev folder.
    out_root holds the fp32 folder the shipping variants are built from (<out_root>/fp32)."""
    _OVERRIDE.clear()
    _OVERRIDE.update({"dev": None, "container": container, "phonon_dir": phonon_dir, "base_dir": base_dir,
                      "out_root": out_root})


def clear_paths():
    _OVERRIDE.clear()


def dev_paths():
    if _OVERRIDE:
        return dict(_OVERRIDE)
    dev = _paths.dev_dir()
    return {
        "dev": dev,
        "container": os.path.join(dev, "models", "stt", "phonon-2", "unpacked", "model.fermion"),
        "phonon_dir": os.path.join(dev, "models", "stt", "phonon-2"),
        "base_dir": os.path.join(dev, "models", "stt", "parakeet-tdt-0.6b-v3-base"),
        "out_root": os.path.join(dev, "models", "stt", "phonon-2-onnx"),
    }


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---- torch model ------------------------------------------------------------------------------------------------

def container_state_dict(container):
    """HF-named fp32 torch state dict (same rules as the vendor's reference_transformers.py)."""
    import torch
    tensors, index = read_container(container, with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v.astype(np.float32))
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                arr = arr[:, :, None]
            sd[k] = torch.from_numpy(arr)
    return sd, index


def load_model(container, base_dir):
    """-> (model, processor). Stock ParakeetForTDT, strict load, eval, fp32, base generation config."""
    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    cfg = ParakeetTDTConfig.from_pretrained(base_dir)
    model = ParakeetForTDT(cfg)
    sd, _ = container_state_dict(container)
    model.load_state_dict(sd, strict=True)
    del sd
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(base_dir)
    processor = AutoProcessor.from_pretrained(base_dir)
    return model, processor


def make_wrappers(model):
    """The two nn.Modules that get exported."""
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.encoder
            self.projector = m.encoder_projector

        def forward(self, input_features):
            return self.projector(self.encoder(input_features=input_features).last_hidden_state)

    class DecoderJoint(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.embedding = m.decoder.embedding
            self.lstm = m.decoder.lstm
            self.decoder_projector = m.decoder.decoder_projector
            self.joint = m.joint

        def forward(self, encoder_frame, token, state_h, state_c):
            emb = self.embedding(token.unsqueeze(1))                       # [1,1,640]
            out, (h, c) = self.lstm(emb, (state_h, state_c))
            g = self.decoder_projector(out)[:, 0, :]                       # [1,640]
            logits = self.joint.head(self.joint.activation(encoder_frame + g))
            return logits[:, :VOCAB], logits[:, VOCAB:], h, c

    return Encoder(model).eval(), DecoderJoint(model).eval()


# ---- export -----------------------------------------------------------------------------------------------------

def export_graphs(model, out_dir, dummy_frames=801):
    """Write encoder.onnx (+ encoder.onnx.data) and decoder_joint.onnx into out_dir."""
    import onnx
    import torch
    enc, dj = make_wrappers(model)
    os.makedirs(out_dir, exist_ok=True)

    # decoder_joint (small, single file)
    args = (torch.randn(1, HIDDEN), torch.tensor([BLANK], dtype=torch.int64),
            torch.zeros(2, 1, HIDDEN), torch.zeros(2, 1, HIDDEN))
    with torch.no_grad():
        torch.onnx.export(
            dj, args, os.path.join(out_dir, "decoder_joint.onnx"), opset_version=OPSET, dynamo=False,
            input_names=["encoder_frame", "token", "state_h", "state_c"],
            output_names=["token_logits", "duration_logits", "state_h_out", "state_c_out"])

    # encoder: legacy exporter writes one external file per tensor when > 2 GB; consolidate afterwards.
    tmp = os.path.join(out_dir, "_enc_raw")
    os.makedirs(tmp, exist_ok=True)
    x = torch.randn(1, dummy_frames, 128)
    with torch.no_grad():
        torch.onnx.export(
            enc, (x,), os.path.join(tmp, "encoder.onnx"), opset_version=OPSET, dynamo=False,
            input_names=["input_features"], output_names=["encoder_projected"],
            dynamic_axes={"input_features": {1: "T"}, "encoder_projected": {1: "F"}})
    m = onnx.load(os.path.join(tmp, "encoder.onnx"), load_external_data=True)
    for vi in list(m.graph.input) + list(m.graph.output):      # batch is fixed to 1 (shape inference leaves a symbol)
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.Clear()
        d0.dim_value = 1
    onnx.save_model(m, os.path.join(out_dir, "encoder.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="encoder.onnx.data", size_threshold=1024)
    del m
    shutil.rmtree(tmp, ignore_errors=True)


def check_graph_signature(path):
    """C70 names/dtypes/ranks. Returns {inputs:{name:(dtype,shape)}, outputs:{...}}; raises AssertionError."""
    import onnx
    m = onnx.load(path, load_external_data=False)

    def desc(vi):
        t = vi.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
        return onnx.TensorProto.DataType.Name(t.elem_type), dims
    return ({v.name: desc(v) for v in m.graph.input}, {v.name: desc(v) for v in m.graph.output})


C70 = {
    "encoder.onnx": ({"input_features": ("FLOAT", [1, "T", 128])},
                     {"encoder_projected": ("FLOAT", [1, "F", 640])}),
    "decoder_joint.onnx": ({"encoder_frame": ("FLOAT", [1, 640]), "token": ("INT64", [1]),
                            "state_h": ("FLOAT", [2, 1, 640]), "state_c": ("FLOAT", [2, 1, 640])},
                           {"token_logits": ("FLOAT", [1, 8193]), "duration_logits": ("FLOAT", [1, 5]),
                            "state_h_out": ("FLOAT", [2, 1, 640]), "state_c_out": ("FLOAT", [2, 1, 640])}),
}


def assert_c70(folder):
    """Names, dtypes, ranks exactly C70; fixed dims equal; dynamic dims (T, F) just need to be non-fixed."""
    for fname, (ins, outs) in C70.items():
        got_in, got_out = check_graph_signature(os.path.join(folder, fname))
        for want, got, kind in ((ins, got_in, "input"), (outs, got_out, "output")):
            assert list(want) == list(got), "%s %s names %s != %s" % (fname, kind, list(got), list(want))
            for n, (dt, shape) in want.items():
                gdt, gshape = got[n]
                assert gdt == dt, "%s %s dtype %s != %s" % (fname, n, gdt, dt)
                assert len(gshape) == len(shape), "%s %s rank %s != %s" % (fname, n, gshape, shape)
                for a, b in zip(gshape, shape):
                    if isinstance(b, int):
                        assert a == b, "%s %s dims %s != %s" % (fname, n, gshape, shape)
                    else:
                        assert not isinstance(a, int), "%s %s dim %s should be dynamic" % (fname, n, gshape)


def op_counts(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    counts = {}
    for n in m.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    return dict(sorted(counts.items())), opset, m.ir_version


def check_onnx(folder):
    import onnx
    for f in ("encoder.onnx", "decoder_joint.onnx"):
        onnx.checker.check_model(os.path.join(folder, f))   # path form: works with external data


def write_sidecar(folder):
    """C78: modelpipelines.json (task + CC-BY attribution) so the editor importer needs no manual settings."""
    from write_phonon2_sidecar import write_sidecar as _write
    return _write(folder)


def write_source_json(folder, variant="fp32"):
    write_sidecar(folder)  # before the file listing so SOURCE.json covers it; every finalised folder passes here
    import onnx
    import onnxruntime
    import torch
    import transformers
    files = {}
    for f in sorted(os.listdir(folder)):
        fp = os.path.join(folder, f)
        if f == "SOURCE.json" or not os.path.isfile(fp):
            continue
        files[f] = {"bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
    src = {"model_id": MODEL_ID, "revision": REVISION, "container_sha256": CONTAINER_SHA256,
           "base_model": BASE_MODEL_ID, "base_revision": BASE_REVISION, "license": "cc-by-4.0",
           "variant": variant, "opset": OPSET, "files": files, "torch": torch.__version__,
           "transformers": transformers.__version__, "onnx": onnx.__version__,
           "onnxruntime": onnxruntime.__version__}
    with open(os.path.join(folder, "SOURCE.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(src, f, indent=1)
        f.write("\n")
    return src


def up_to_date(folder):
    sj = os.path.join(folder, "SOURCE.json")
    if not os.path.isfile(sj):
        return False
    try:
        src = json.load(open(sj, encoding="utf-8"))
        if src.get("revision") != REVISION or src.get("container_sha256") != CONTAINER_SHA256:
            return False
        for name, e in src["files"].items():
            fp = os.path.join(folder, name)
            if not os.path.isfile(fp) or os.path.getsize(fp) != e["bytes"] or sha256_file(fp) != e["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def copy_support_files(folder):
    p = dev_paths()
    for f in BASE_COPY:
        shutil.copyfile(os.path.join(p["base_dir"], f), os.path.join(folder, f))
    for f in PHONON_COPY:
        shutil.copyfile(os.path.join(p["phonon_dir"], f), os.path.join(folder, f))
