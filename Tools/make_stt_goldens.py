"""Write the STT goldens (WP9S.2, contract C71): mel filters, per-clip audio / features / log-mel sidecars and
stt_goldens.json (Hugging Face fp32 decode + onnxruntime decode of the shipping variant).

  python make_stt_goldens.py [--clips <clips_visible.json>] [--out <dir>] [--variant fp16w]

Run with the pipelines-stt env through micromamba (numpy matmul crashes un-activated):
  MAMBA_ROOT_PREFIX=D:/Micromamba/micromambaenv D:/Micromamba/micromamba.exe run -n pipelines-stt python make_stt_goldens.py
Offline; deterministic (a rerun gives byte-identical files). Inputs: clips file, LibriSpeech FLACs, container model (fp32 HF
reference), <dev>/models/stt/phonon-2-onnx/<variant>/. Goldens come only from the Python stack, never from Unreal output.
"""
import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stt import clips as cl  # noqa: E402
from stt import goldens as g  # noqa: E402
from stt import phonon2_export as ex  # noqa: E402
from stt import tdt_greedy as tg  # noqa: E402


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_sidecar(out, rel, arr):
    b = g.f32_bytes(arr)
    path = os.path.join(out, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)
    return g.sha256_bytes(b)


def main(argv=None):
    p = ex.dev_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=os.path.join(p["dev"], "goldens", "stt", "clips_visible.json"))
    ap.add_argument("--out", default=os.path.join(p["dev"], "goldens", "stt"))
    ap.add_argument("--variant", default="fp16w")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import librosa
    import onnxruntime
    import torch
    import transformers
    import check_phonon2_onnx as chk

    rows = cl.load_clips(a.clips)
    folder = os.path.join(p["out_root"], a.variant)
    with open(os.path.join(folder, "SOURCE.json"), encoding="utf-8") as f:
        src = json.load(f)
    hf = chk.HfRef(p["container"], p["base_dir"])
    proc = hf.processor
    tok = proc.tokenizer
    ort = tg.OrtTdt(folder)
    fe = proc.feature_extractor
    mel_filters = fe.mel_filters.numpy()
    assert mel_filters.shape == (128, 257) and mel_filters.dtype == np.float32
    os.makedirs(a.out, exist_ok=True)
    mel_sha = write_sidecar(a.out, "mel_filters.f32", mel_filters)

    items = []   # (id, kind, wav, ref)
    for r in rows:
        items.append((r["id"], "librispeech", cl.read_audio(cl.clip_abs_path(r, p["dev"])), r["ref"]))
    for cid, wav in g.synthetic_clips(items[0][2]):
        items.append((cid, "synthetic", wav, None))

    def pieces(k):
        return tok.convert_ids_to_tokens([int(k)])[0]

    clips = []
    for i, (cid, kind, wav, ref) in enumerate(items):
        n = len(wav)
        entry = {"id": cid, "kind": kind,
                 "audio": {"file": f"audio/{cid}.f32", "num_samples": n,
                           "sha256": write_sidecar(a.out, f"audio/{cid}.f32", wav)},
                 "rms_dbfs": g.rms_dbfs(wav), "features": None, "logmel": None, "encoder_frames": None,
                 "hf": None, "onnx": None, "ref": ref}
        if g.valid_frames(n) < g.MIN_VALID_FRAMES:
            entry["note"] = "too short: fewer than 2 valid mel frames"
            clips.append(entry)
            print(f"[{i + 1}/{len(items)}] {cid}: too short", flush=True)
            continue
        inp = hf.features(wav)
        feats = chk.valid_features(inp)[0]
        assert feats.shape[0] == n // g.HOP, (feats.shape, n)
        entry["features"] = {"file": f"features/{cid}.f32", "num_frames": int(feats.shape[0]),
                             "sha256": write_sidecar(a.out, f"features/{cid}.f32", feats)}
        if i < 3 or kind == "synthetic":
            x = torch.from_numpy(wav)[None].to(torch.float32)
            x = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], dim=1)
            lm = fe._torch_extract_fbank_features(x)[0].numpy()
            entry["logmel"] = {"file": f"logmel/{cid}.f32", "num_frames": int(lm.shape[0]),
                               "sha256": write_sidecar(a.out, f"logmel/{cid}.f32", lm)}
        seq, dur, text = hf.generate(inp)
        entry["hf"] = {"sequences": seq, "durations": dur, "text": text}
        enc = ort.encode(feats[None])
        entry["encoder_frames"] = int(enc.shape[0])
        steps, ids = tg.greedy_tdt(ort.step, enc, max_steps=hf.full_cap(inp))
        entry["onnx"] = {"steps": [[k, d] for k, d in steps],
                         "text": hf.decode_text(ids), "token_ids": ids,
                         "word_frames": g.word_frames(steps, pieces)}
        clips.append(entry)
        print(f"[{i + 1}/{len(items)}] {cid}: F={enc.shape[0]} onnx==hf {entry['onnx']['text'] == text} | {text[:60]}",
              flush=True)

    doc = {
        "schema": "stt-goldens/1",
        "generator": {"script": "Tools/make_stt_goldens.py", "torch": torch.__version__,
                      "transformers": transformers.__version__, "onnxruntime": onnxruntime.__version__,
                      "numpy": np.__version__, "librosa": librosa.__version__},
        "model": {"container_sha256": src["container_sha256"], "revision": src["revision"], "variant": a.variant,
                  "encoder_sha256": sha256_file(os.path.join(folder, "encoder.onnx")),
                  "decoder_sha256": sha256_file(os.path.join(folder, "decoder_joint.onnx")),
                  "base_model": src.get("base_model"), "base_revision": src.get("base_revision")},
        "settings": {"sample_rate": 16000, "num_mel_bins": 128, "fft_size": 512, "window_length": 400, "hop_length": 160,
                     "preemphasis": 0.97, "subsampling_factor": 8, "blank_token_id": 8192,
                     "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10,
                     "log_zero_guard": g.LOG_GUARD, "norm_epsilon": g.NORM_EPS,
                     "window": "hann symmetric, 400 centred in 512", "stft_padding": "zero, 256 each side",
                     "mel_norm": "slaney (librosa), fmin 0, fmax 8000",
                     "exact_16000": "samples [16000:32000) of the first visible clip",
                     "word_frames": "start = t of first token, end = t + duration of last token; "
                                    "a word starts at a piece beginning with U+2581"},
        "mel_filters": {"file": "mel_filters.f32", "shape": [128, 257], "sha256": mel_sha},
        "clips": clips,
    }
    path = os.path.join(a.out, "stt_goldens.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
