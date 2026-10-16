"""Write the long-audio goldens (WP9S.5c1, contract C77): stt-long-goldens/1.

  python make_stt_long_goldens.py --make-spec                      # write <dev>/goldens/stt/long/long_spec_visible.json
  python make_stt_long_goldens.py [--spec <json>] [--out <dir>] [--variant fp16w]

Run with the pipelines-stt env through micromamba (numpy matmul crashes un-activated):
  MAMBA_ROOT_PREFIX=D:/Micromamba/micromambaenv D:/Micromamba/micromamba.exe run -n pipelines-stt python make_stt_long_goldens.py
Spec: list of {"id","speaker","chapter","target_seconds"} (LibriSpeech: the chapter's utterances in order, concatenated
until >= target_seconds) or {"id","synthetic":"silence_60s"|"gap_70s"}. Offline; byte-identical on rerun. Goldens come
only from the Python stack (HF feature extractor + onnxruntime fp16w greedy loop), never from Unreal output.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stt import clips as cl  # noqa: E402
from stt import goldens as g  # noqa: E402
from stt import long_audio as la  # noqa: E402
from stt import phonon2_export as ex  # noqa: E402
from stt import tdt_greedy as tg  # noqa: E402
from stt import wer as wr  # noqa: E402

TARGETS = (45, 60, 90, 120, 180, 300)
LIMIT_S = 600


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def libri_root(dev):
    return os.path.join(dev, "testdata", "stt", "librispeech", "LibriSpeech", "test-clean")


# Explicit visible set (lead, 2026-10-16): first 6 speakers' lowest chapters, except 908 (its chapters are < 300 s) -> 1188.
VISIBLE = [("61", "70968", 45), ("121", "121726", 60), ("237", "126133", 90), ("260", "123286", 120),
           ("672", "122797", 180), ("1188", "133604", 300)]


def make_visible_spec(dev):
    spec = [{"id": f"long_{spk}_{chap}_{tgt}s", "speaker": spk, "chapter": chap, "target_seconds": tgt}
            for spk, chap, tgt in VISIBLE]
    spec.append({"id": "silence_60s", "synthetic": "silence_60s"})
    spec.append({"id": "gap_70s", "synthetic": "gap_70s"})
    return spec


def libri_recording(dev, item):
    cdir = os.path.join(libri_root(dev), str(item["speaker"]), str(item["chapter"]))
    refs = cl._read_trans(cdir)
    uids = sorted(f[:-5] for f in os.listdir(cdir) if f.endswith(".flac"))
    parts, used, total = [], [], 0
    for uid in uids:
        w = cl.read_audio(os.path.join(cdir, uid + ".flac"))
        parts.append(w)
        used.append(uid)
        total += len(w)
        if total >= item["target_seconds"] * la.R:
            break
    if total < item["target_seconds"] * la.R:
        raise SystemExit(f"{item['id']}: chapter {item['speaker']}/{item['chapter']} has only {total / la.R:.1f} s "
                         f"< target {item['target_seconds']} s")
    return used, [parts[i] for i in range(len(parts))], " ".join(refs[u] for u in used)


def tile(wav, n):
    return np.tile(wav, -(-n // len(wav)))[:n].astype(np.float32)


def synthetic(name, first_clip):
    if name == "silence_60s":
        return np.zeros(60 * la.R, dtype=np.float32)
    if name == "gap_70s":
        a = tile(first_clip, 33 * la.R)
        z = np.zeros(3 * la.R, dtype=np.float32)
        b = tile(first_clip, 70 * la.R - len(a) - len(z))
        return np.concatenate([a, z, b])
    raise SystemExit(f"unknown synthetic recording {name}")


def main(argv=None):
    p = ex.dev_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--out", default=os.path.join(p["dev"], "goldens", "stt", "long"))
    ap.add_argument("--variant", default="fp16w")
    ap.add_argument("--clips", default=os.path.join(p["dev"], "goldens", "stt", "clips_visible.json"))
    ap.add_argument("--make-spec", action="store_true", help="write long_spec_visible.json into --out and exit")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec_path = a.spec or os.path.join(a.out, "long_spec_visible.json")
    if a.make_spec:
        with open(spec_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(make_visible_spec(p["dev"]), f, indent=1)
            f.write("\n")
        print("wrote", spec_path)
        return
    warnings.filterwarnings("ignore")
    import onnxruntime
    import transformers
    import check_phonon2_onnx as chk

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    folder = os.path.join(p["out_root"], a.variant)
    hf = chk.HfRef(p["container"], p["base_dir"])
    tok = hf.processor.tokenizer
    ort = tg.OrtTdt(folder)
    first_clip = cl.read_audio(cl.clip_abs_path(cl.load_clips(a.clips)[0], p["dev"]))
    timing = {"decode_s": 0.0, "audio_s": 0.0}

    def pieces(k):
        return tok.convert_ids_to_tokens([int(k)])[0]

    def decode(w):
        w = np.ascontiguousarray(w, dtype=np.float32)
        rms = g.rms_dbfs(w)
        gated = {"rms_dbfs": rms, "gated": True, "token_ids": [], "text": "", "words": []}
        if rms < la.GATE_DB or g.valid_frames(len(w)) < g.MIN_VALID_FRAMES:
            return gated
        t0 = time.perf_counter()
        inp = hf.features(w)
        feats = chk.valid_features(inp)[0]
        enc = ort.encode(feats[None])
        steps, ids = tg.greedy_tdt(ort.step, enc, max_steps=hf.full_cap(inp))
        timing["decode_s"] += time.perf_counter() - t0
        timing["audio_s"] += len(w) / la.R
        return {"rms_dbfs": rms, "gated": False, "token_ids": ids, "text": hf.decode_text(ids),
                "words": la.words_from_frames(g.word_frames(steps, pieces))}

    utt_cache = {}

    def decode_utt(uid, w):
        if uid not in utt_cache:
            utt_cache[uid] = decode(w)["text"]
        return utt_cache[uid]

    recs, refs, h_long, h_utt, h_fix = [], [], [], [], []
    summary = []
    for item in spec:
        rid = item["id"]
        if "synthetic" in item:
            x, kind, uids, ref, parts = synthetic(item["synthetic"], first_clip), "synthetic", [], None, None
        else:
            uids, parts, ref = libri_recording(p["dev"], item)
            x, kind = np.concatenate(parts).astype(np.float32), "librispeech"
        assert len(x) <= LIMIT_S * la.R
        wins = la.plan_windows(x)
        dw, text, words = la.decode_windows(x, wins, decode)
        if rid == "gap_70s":
            assert 33 * la.R <= wins[0][1] <= 36 * la.R, wins
        if rid == "silence_60s":
            assert text == "" and all(w["gated"] for w in dw), dw
        b = g.f32_bytes(x)
        rel = f"long_audio/{rid}.f32"
        os.makedirs(os.path.join(a.out, "long_audio"), exist_ok=True)
        with open(os.path.join(a.out, "long_audio", rid + ".f32"), "wb") as f:
            f.write(b)
        recs.append({"id": rid, "kind": kind, "utterances": uids,
                     "audio": {"file": rel, "num_samples": int(len(x)), "sha256": g.sha256_bytes(b)},
                     "ref": ref, "windows": dw, "text": text, "words": words})
        line = {"id": rid, "seconds": len(x) / la.R, "cuts_s": [w[1] / la.R for w in wins[:-1]]}
        if kind == "librispeech":
            per = " ".join(t for t in (decode_utt(u, w).strip() for u, w in zip(uids, parts)) if t)
            _, fixed, _ = la.decode_windows(x, la.plan_fixed(len(x)), decode)
            refs.append(ref)
            h_long.append(text)
            h_utt.append(per)
            h_fix.append(fixed)
            nz = wr.make_normalizer()
            line.update({k: 100 * wr.normalized_corpus_wer([ref], [h], nz)["wer"]
                         for k, h in (("long", text), ("per_utt", per), ("fixed30", fixed))})
        summary.append(line)
        print(json.dumps(line), flush=True)

    nz = wr.make_normalizer()
    wer = {"long": wr.normalized_corpus_wer(refs, h_long, nz)["wer"],
           "per_utterance": wr.normalized_corpus_wer(refs, h_utt, nz)["wer"],
           "fixed_30s": wr.normalized_corpus_wer(refs, h_fix, nz)["wer"],
           "normalizer": "whisper english"}
    with open(os.path.join(folder, "SOURCE.json"), encoding="utf-8") as f:
        json.load(f)
    doc = {"schema": "stt-long-goldens/1",
           "generator": {"script": "Tools/make_stt_long_goldens.py", "transformers": transformers.__version__,
                         "onnxruntime": onnxruntime.__version__, "numpy": np.__version__},
           "model": {"variant": a.variant, "encoder_sha256": sha256_file(os.path.join(folder, "encoder.onnx")),
                     "decoder_sha256": sha256_file(os.path.join(folder, "decoder_joint.onnx"))},
           "settings": {"hop": la.H, "min_window_s": la.MIN_S, "max_window_s": la.MAX_S, "tail_s": la.TAIL_S,
                        "half_frames": la.HALF, "limit_s": LIMIT_S, "silence_threshold_db": la.GATE_DB},
           "recordings": recs, "wer": wer}
    path = os.path.join(a.out, "long_goldens.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wer (fraction):", json.dumps(wer))
    print(f"decode time (long windows, excl. cache/per-utt/fixed runs counted too): {timing['decode_s']:.1f} s "
          f"for {timing['audio_s']:.0f} s audio")
    print("wrote", path)


if __name__ == "__main__":
    main()
