"""Check the Phonon-2 ONNX export against Hugging Face and write the WP9S.1a report (contract C70).

  python check_phonon2_onnx.py --make-clips visible            # writes <dev>/goldens/stt/clips_visible.json
  python check_phonon2_onnx.py --wer --variant fp16w --clips all   # WP9S.1b: 40-clip eval + test-clean WER
  python check_phonon2_onnx.py --variant-report                    # WP9S.1b: combine into wp9s1b_report
  python check_phonon2_onnx.py --variant fp32 --clips <dev>/goldens/stt/clips_visible.json [--report-dir DIR]

Steps: (1) graph parity on the first --parity-clips clips (encoder vs get_audio_features, decoder_joint on 50 real
tuples), (2) loop parity: HF ParakeetForTDT.generate vs tdt_greedy over ORT on every clip (sequences, durations, text),
(3) edge checks (step cap, silence, longest test-clean clip, non-ASCII, argmax ties), (4) report .json + .md.
Run with the pipelines-stt env. Offline.
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


def peak_rss_mb():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / 1048576, 1)


class HfRef:
    """HF reference: model + processor from the container (same call as reference_transformers.py)."""

    def __init__(self, container, base_dir):
        import torch
        from stt import phonon2_export as ex
        self.torch = torch
        self.model, self.processor = ex.load_model(container, base_dir)

    def features(self, wav):
        inp = self.processor([wav], sampling_rate=16000, return_tensors="pt", padding=True)
        return inp

    def generate(self, inp):
        torch = self.torch
        with torch.no_grad():
            out = self.model.generate(input_features=inp["input_features"], attention_mask=inp.get("attention_mask"))
        seq = [int(x) for x in out.sequences[0].tolist()]
        dur = [int(x) for x in out.durations[0].tolist()]
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        return seq, dur, text

    def decode_text(self, ids):
        return self.processor.batch_decode(self.torch.tensor([ids]), skip_special_tokens=True)[0].strip()

    def full_cap(self, inp):
        """HF output-buffer cap in steps: max_length = 10 * subsampled(ALL mel frames), minus the start token."""
        torch = self.torch
        n = int(inp["input_features"].shape[1])
        f = int(self.model.encoder._get_subsampling_output_length(torch.tensor([n])).item())
        return 10 * f - 1

    def audio_features(self, inp):
        torch = self.torch
        with torch.no_grad():
            o = self.model.get_audio_features(input_features=inp["input_features"],
                                              attention_mask=inp.get("attention_mask"))
        return o.pooler_output[0].numpy()


def valid_features(inp):
    """The mel frames the encoder sees: the first attention_mask.sum() frames (the extractor masks the last frame of
    n_samples // 160 + 1). C70 encoder.onnx has no mask input, so the caller passes only valid frames."""
    x = inp["input_features"].numpy()
    am = inp.get("attention_mask")
    return x[:, :int(am.sum())] if am is not None else x


def run_clip(hf, ort, wav):
    """-> dict with both decodes (sequences, durations, text) and timings."""
    from stt import tdt_greedy as tg
    inp = hf.features(wav)
    t0 = time.perf_counter()
    seq_h, dur_h, text_h = hf.generate(inp)
    hf_ms = (time.perf_counter() - t0) * 1000
    feats = valid_features(inp)
    t0 = time.perf_counter()
    enc = ort.encode(feats)
    enc_ms = (time.perf_counter() - t0) * 1000
    n0, c0 = ort.step_ns, ort.step_count
    trace = []
    t0 = time.perf_counter()
    steps, ids = tg.greedy_tdt(ort.step, enc, trace=trace, max_steps=hf.full_cap(inp))
    dec_ms = (time.perf_counter() - t0) * 1000
    seq_o, dur_o = tg.sequences_and_durations(steps)
    text_o = hf.decode_text(ids)
    return {"seq_h": seq_h, "dur_h": dur_h, "text_h": text_h, "seq_o": seq_o, "dur_o": dur_o, "text_o": text_o,
            "hf_ms": hf_ms, "enc_ms": enc_ms, "dec_ms": dec_ms, "frames": int(enc.shape[0]),
            "mel_frames": int(feats.shape[1]), "step_ns": ort.step_ns - n0, "step_count": ort.step_count - c0,
            "trace": trace, "match": seq_h == seq_o and dur_h == dur_o and text_h == text_o}


def describe_mismatch(r):
    sh, so, dh, do = r["seq_h"], r["seq_o"], r["dur_h"], r["dur_o"]
    n = min(len(sh), len(so))
    i = next((j for j in range(n) if sh[j] != so[j] or dh[j] != do[j]), n)
    m = {"first_diff_index": i, "hf_len": len(sh), "ort_len": len(so)}
    if i < n:
        m["hf_kd"] = [sh[i], dh[i]]
        m["ort_kd"] = [so[i], do[i]]
    if 1 <= i <= len(r["trace"]):
        t, top1, top2 = r["trace"][i - 1]
        m["ort_frame"] = t
        m["ort_top1_top2_margin"] = top1 - top2
    return m


def graph_parity(hf, ort, clips, dev, n_clips, n_tuples=50):
    from stt import clips as cl
    torch = hf.torch
    model = hf.model
    enc_res = []
    tuples = []   # (frame, token, h, c)
    for row in clips[:n_clips]:
        wav = cl.read_audio(cl.clip_abs_path(row, dev))
        inp = hf.features(wav)
        ref = hf.audio_features(inp)
        got = ort.encode(valid_features(inp))
        extra = int(ref.shape[0] - got.shape[0])      # HF's pooler output may hold one trailing masked frame
        assert extra in (0, 1), (ref.shape, got.shape)
        ref = ref[:got.shape[0]]
        diff = float(np.abs(ref - got).max())
        num = (ref * got).sum(-1)
        den = np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1)
        cos = float((num / den).min())
        enc_res.append({"id": row["id"], "frames": int(ref.shape[0]), "hf_extra_masked_frames": extra, "max_abs_diff": diff, "min_cosine": cos})
        # replay the HF decode with torch modules to harvest real (frame, token, state) tuples
        seq, dur, _ = hf.generate(inp)
        t, tok = 0, 8192
        h = torch.zeros(2, 1, 640)
        c = torch.zeros(2, 1, 640)
        clip_tuples = []
        for k, d in zip(seq[1:], dur[1:]):
            clip_tuples.append((torch.from_numpy(ref[t:t + 1].copy()), tok, h.clone(), c.clone(), k))
            with torch.no_grad():
                _, _, h2, c2 = torch_step(model, clip_tuples[-1][0], tok, h, c)
            if k != 8192:
                tok, h, c = k, h2, c2
            t += d
        tuples.append(clip_tuples)
    # pick 50 spread over all clips: always first step of every clip, then evenly spaced, forcing some after non-blank
    flat = [(ci, si) for ci, ct in enumerate(tuples) for si in range(len(ct))]
    pick = [(ci, 0) for ci in range(len(tuples))]
    nonblank_after = [(ci, si) for ci, si in flat if si > 0 and tuples[ci][si - 1][4] != 8192 and tuples[ci][si][1] != 8192]
    rest = n_tuples - len(pick)
    half = rest // 2
    step_nb = max(1, len(nonblank_after) // half)
    pick += nonblank_after[::step_nb][:half]
    others = [x for x in flat if x not in set(pick)]
    step_o = max(1, len(others) // (n_tuples - len(pick)))
    pick += others[::step_o][:n_tuples - len(pick)]
    pick = pick[:n_tuples]
    tl_d = dl_d = st_d = 0.0
    n_after_nb = 0
    for ci, si in pick:
        frame, tok, h, c, _k = tuples[ci][si]
        tl_r, dl_r, h_r, c_r = [x.numpy() for x in torch_step(model, frame, tok, h, c)]
        tl_o, dl_o, h_o, c_o = ort.step(frame.numpy(), np.array([tok], np.int64), h.numpy(), c.numpy())
        tl_d = max(tl_d, float(np.abs(tl_r - tl_o).max()))
        dl_d = max(dl_d, float(np.abs(dl_r - dl_o).max()))
        st_d = max(st_d, float(np.abs(h_r - h_o).max()), float(np.abs(c_r - c_o).max()))
        n_after_nb += int(tok != 8192)
    return {"encoder": enc_res,
            "decoder_joint": {"tuples": len(pick), "tuples_with_nonblank_input_token": n_after_nb,
                              "first_steps_from_zero_state": n_clips,
                              "token_logits_max_abs_diff": tl_d, "duration_logits_max_abs_diff": dl_d,
                              "states_max_abs_diff": st_d}}


def torch_step(model, frame, tok, h, c):
    """One decoder+joint step using the stock HF modules (independent of the exported wrapper)."""
    torch = model.encoder.parameters().__next__().new_tensor
    import torch as T
    with T.no_grad():
        emb = model.decoder.embedding(T.tensor([[tok]], dtype=T.int64))
        out, (h2, c2) = model.decoder.lstm(emb, (h, c))
        g = model.decoder.decoder_projector(out)[:, 0, :]
        logits = model.joint(encoder_hidden_states=frame, decoder_hidden_states=g)
    return logits[:, :8193], logits[:, 8193:], h2, c2


def op_report(folder):
    from stt import phonon2_export as ex
    out = {}
    for f in ("encoder.onnx", "decoder_joint.onnx"):
        counts, opset, ir = ex.op_counts(os.path.join(folder, f))
        out[f] = {"op_types": counts, "opset": opset, "ir_version": ir,
                  "bytes": os.path.getsize(os.path.join(folder, f))}
    return out


def edge_checks(hf, ort, dev, clips_rows, visible_results):
    from stt import clips as cl
    res = {}
    res["a_step_cap"] = ("HF max_length counts the start token: MaxLengthCriteria stops when input_ids.shape[1] >= max_length "
                         "(transformers/generation/stopping_criteria.py:82-83) and max_length = max_symbols_per_step * F "
                         "(models/parakeet/generation_parakeet.py:181), so sequences hold at most 10*F tokens = 10*F-1 steps; "
                         "tdt_greedy uses len(steps) < 10*F-1. HF sizes F from ALL mel frames (one more than the valid frames the encoder is fed), "
                         "so the check passes max_steps = 10*F_all-1; the cap is never reached on real speech.")
    # b: silence
    r = run_clip(hf, ort, np.zeros(8000, dtype=np.float32))
    res["b_silence_0p5s"] = {"match": r["match"], "hf_text": r["text_h"], "ort_text": r["text_o"], "frames": r["frames"],
                             "steps": len(r["seq_o"]) - 1}
    # c: longest test-clean clip
    root = os.path.join(dev, "testdata", "stt", "librispeech", "LibriSpeech", "test-clean")
    allrows = cl.scan_split(root)
    longest = max(allrows, key=lambda x: (x["num_samples"], x["id"]))
    r = run_clip(hf, ort, cl.read_audio(longest["abs_path"]))
    res["c_longest_clip"] = {"id": longest["id"], "seconds": round(longest["num_samples"] / 16000, 3),
                             "utterances_scanned": len(allrows), "match": r["match"], "steps": len(r["seq_o"]) - 1,
                             "enc_ms": r["enc_ms"], "dec_ms": r["dec_ms"], "hf_ms": r["hf_ms"],
                             "mismatch": None if r["match"] else describe_mismatch(r)}
    # d: non-ASCII
    na = [{"id": row["id"], "text": v["text_h"]} for row, v in visible_results if any(ord(ch) > 127 for ch in v["text_h"])]
    res["d_non_ascii"] = na if na else "none of the 40 decoded transcripts contains a non-ASCII character"
    # e: ties
    a = np.array([0.5, 1.0, 1.0, 0.2], dtype=np.float32)
    import torch
    res["e_argmax_tie"] = {"numpy": int(np.argmax(a)), "torch": int(torch.argmax(torch.from_numpy(a))),
                           "first_max_both": int(np.argmax(a)) == int(torch.argmax(torch.from_numpy(a))) == 1}
    assert res["e_argmax_tie"]["first_max_both"]
    return res


def versions():
    import importlib
    v = {}
    for n in ("torch", "transformers", "onnx", "onnxruntime", "numpy", "librosa", "soundfile", "tokenizers"):
        try:
            v[n] = importlib.import_module(n).__version__
        except Exception:
            v[n] = None
    v["python"] = sys.version.split()[0]
    return v


def write_md(rep, path):
    L = []
    a = L.append
    a("# WP9S.1a report: Phonon-2 fp32 ONNX + greedy TDT over onnxruntime\n")
    lp = rep["loop_parity"]
    a("Loop parity: **%d/%d clips** identical sequences, durations and text (variant %s).\n" % (lp["matched"], lp["clips"], rep["variant"]))
    a("## Files\n")
    a("| file | bytes | sha256 |\n|---|---|---|")
    for n, e in rep["files"].items():
        a("| %s | %d | %s |" % (n, e["bytes"], e["sha256"]))
    a("\n## ONNX ops\n")
    for g, e in rep["graphs"].items():
        a("%s: opset %s, IR %d, %d bytes. ops: %s\n" % (g, e["opset"], e["ir_version"], e["bytes"],
                                                        ", ".join("%s x%d" % kv for kv in e["op_types"].items())))
    gp = rep["graph_parity"]
    a("## Graph parity (ORT CPU vs torch)\n")
    a("encoder_projected: max abs diff %.3e, min per-frame cosine %.8f over %d clips" % (
        max(e["max_abs_diff"] for e in gp["encoder"]), min(e["min_cosine"] for e in gp["encoder"]), len(gp["encoder"])))
    a("\ndecoder_joint (%s)\n" % json.dumps(gp["decoder_joint"]))
    a("## Timings (ORT CPU, default threads)\n")
    t = rep["timings"]
    a(json.dumps(t, indent=1))
    a("\nPeak process RSS: %s MB\n" % rep["peak_rss_mb"])
    a("## Edge checks\n")
    a("```\n" + json.dumps(rep["edge_checks"], indent=1, ensure_ascii=False) + "\n```\n")
    a("## Clips\n")
    a("| id | seconds | steps | tokens | match | HF ms | ORT ms (enc+loop) |\n|---|---|---|---|---|---|---|")
    for r in rep["clips"]:
        a("| %s | %.2f | %d | %d | %s | %.0f | %.0f |" % (r["id"], r["seconds"], r["steps"], r["tokens"], r["match"],
                                                           r["hf_ms"], r["ort_ms"]))
    if rep["loop_parity"]["mismatches"]:
        a("\n## Mismatches\n```\n" + json.dumps(rep["loop_parity"]["mismatches"], indent=1) + "\n```")
    a("\n## Versions\n```\n" + json.dumps(rep["versions"], indent=1) + "\n```")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def ort_decode(proc, ort, wav):
    """Encoder + greedy TDT on one waveform with ORT only (no HF model). -> dict."""
    from stt import tdt_greedy as tg
    inp = proc([wav], sampling_rate=16000, return_tensors="pt", padding=True)
    feats = valid_features(inp)
    t0 = time.perf_counter()
    enc = ort.encode(feats)
    enc_ms = (time.perf_counter() - t0) * 1000
    n0, c0 = ort.step_ns, ort.step_count
    t0 = time.perf_counter()
    steps, ids = tg.greedy_tdt(ort.step, enc, max_steps=10 * (int(enc.shape[0]) + 1) - 1)
    dec_ms = (time.perf_counter() - t0) * 1000
    seq, dur = tg.sequences_and_durations(steps)
    text = proc.batch_decode([ids], skip_special_tokens=True)[0].strip()
    return {"seq": seq, "dur": dur, "text": text, "enc": enc, "enc_ms": enc_ms, "dec_ms": dec_ms,
            "step_ns": ort.step_ns - n0, "step_count": ort.step_count - c0}


def eval_variant(variant, clips_file, do_wer, report_dir, dev, p):
    """One process, one encoder session: 40 visible clips (parity data for the variant report, speed, RSS),
    then optionally the full test-clean WER run. Writes wp9s1b_eval_<variant>.json/.npz and the hyp jsonl."""
    from stt import clips as cl
    from stt import wer as wr
    from stt.tdt_greedy import OrtTdt
    from transformers import AutoProcessor
    folder = os.path.join(p["out_root"], variant)
    proc = AutoProcessor.from_pretrained(p["base_dir"])
    t0 = time.perf_counter()
    ort = OrtTdt(folder)
    create_s = time.perf_counter() - t0
    rows = cl.load_clips(clips_file)
    res, encs = [], {}
    for row in rows:
        wav = cl.read_audio(cl.clip_abs_path(row, dev))
        r = ort_decode(proc, ort, wav)
        encs[row["id"]] = r.pop("enc")
        r.update(id=row["id"], seconds=len(wav) / 16000)
        res.append(r)
    secs = sum(r["seconds"] for r in res)
    steps = sum(r["step_count"] for r in res)
    enc_ms = [r["enc_ms"] for r in res]
    out = {"variant": variant, "session_create_s": round(create_s, 2), "clips": res,
           "encoder_ms_mean": round(sum(enc_ms) / len(enc_ms), 1), "encoder_ms_max": round(max(enc_ms), 1),
           "decoder_joint_us_per_step": round(sum(r["step_ns"] for r in res) / max(1, steps) / 1000, 1),
           "rtf": round(sum(r["enc_ms"] + r["dec_ms"] for r in res) / 1000 / secs, 4),
           "audio_seconds": round(secs, 2), "peak_rss_mb_40clips": peak_rss_mb(), "ops": op_report(folder)}
    os.makedirs(report_dir, exist_ok=True)
    np.savez(os.path.join(report_dir, "wp9s1b_eval_%s.npz" % variant), **encs)
    if do_wer:
        root = os.path.join(dev, "testdata", "stt", "librispeech", "LibriSpeech", "test-clean")
        allrows = cl.scan_split(root)
        nz = wr.make_normalizer()
        hp = os.path.join(report_dir, "wp9s1b_hyp_%s.jsonl" % variant)
        refs, hyps, tsec, tcomp = [], [], 0.0, 0.0
        with open(hp, "w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(allrows):
                wav = cl.read_audio(row["abs_path"])
                r = ort_decode(proc, ort, wav)
                tsec += len(wav) / 16000
                tcomp += (r["enc_ms"] + r["dec_ms"]) / 1000
                f.write(json.dumps({"id": row["id"], "ref": row["ref"], "hyp": r["text"]}, ensure_ascii=False) + "\n")
                refs.append(nz(row["ref"]))
                hyps.append(nz(r["text"]))
                if i % 100 == 0:
                    print("%d/%d wer so far %.3f%%" % (i, len(allrows), 100 * wr.corpus_wer(refs, hyps)["wer"]), flush=True)
        w = wr.corpus_wer(refs, hyps)
        out["wer"] = dict(w, utterances=len(allrows), audio_seconds=round(tsec, 1), rtf=round(tcomp / tsec, 4))
        print("WER %s: %.4f%% over %d utterances" % (variant, 100 * w["wer"], len(allrows)), flush=True)
    out["peak_rss_mb"] = peak_rss_mb()
    with open(os.path.join(report_dir, "wp9s1b_eval_%s.json" % variant), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return out


S2_DELTA_PP = 0.10
S3_MIN = 38
FERMION_CLEAN_WER_PCT = 1.72


def variant_report(report_dir, dev, p):
    """Combine wp9s1b_eval_*.json into wp9s1b_report.json/.md (variants vs the fp32 ORT path)."""
    from stt import phonon2_variants as pv
    ev = {}
    for v in ("fp32", "fp16w", "int8", "fp16"):
        jp = os.path.join(report_dir, "wp9s1b_eval_%s.json" % v)
        if os.path.isfile(jp):
            ev[v] = json.load(open(jp, encoding="utf-8"))
    base = ev["fp32"]
    base_enc = np.load(os.path.join(report_dir, "wp9s1b_eval_fp32.npz"))
    rows = {}
    for v, e in ev.items():
        folder = os.path.join(p["out_root"], v)
        enc_bytes = os.path.getsize(os.path.join(folder, "encoder.onnx"))
        entry = {"encoder_bytes": enc_bytes, "rtf": e["rtf"], "peak_rss_mb": e["peak_rss_mb_40clips"],
                 "peak_rss_mb_after_wer": e["peak_rss_mb"], "session_create_s": e["session_create_s"],
                 "encoder_ms_mean": e["encoder_ms_mean"], "encoder_ms_max": e["encoder_ms_max"],
                 "decoder_joint_us_per_step": e["decoder_joint_us_per_step"],
                 "op_types_encoder": e["ops"]["encoder.onnx"]["op_types"]}
        if v != "fp32":
            s1 = pv.check_s1(folder)
            entry["s1"] = s1
            entry["decoder_joint_identical"] = (
                open(os.path.join(folder, "decoder_joint.onnx"), "rb").read() ==
                open(os.path.join(p["out_root"], "fp32", "decoder_joint.onnx"), "rb").read())
            same_seq = same_text = 0
            diffs, cosm, mism = [], [], []
            vz = np.load(os.path.join(report_dir, "wp9s1b_eval_%s.npz" % v))
            for a, b in zip(base["clips"], e["clips"]):
                assert a["id"] == b["id"]
                ra, rb = base_enc[a["id"]], vz[b["id"]]
                diffs.append(float(np.abs(ra - rb).max()))
                cosm.append(float(((ra * rb).sum(-1) / (np.linalg.norm(ra, axis=-1) * np.linalg.norm(rb, axis=-1))).min()))
                if a["seq"] == b["seq"] and a["dur"] == b["dur"]:
                    same_seq += 1
                if a["text"] == b["text"]:
                    same_text += 1
                if a["seq"] != b["seq"] or a["dur"] != b["dur"]:
                    sa, sb, da, db = a["seq"], b["seq"], a["dur"], b["dur"]
                    n = min(len(sa), len(sb))
                    i = next((j for j in range(n) if sa[j] != sb[j] or da[j] != db[j]), n)
                    mism.append({"id": a["id"], "first_diff_step": i, "text_identical": a["text"] == b["text"]})
            entry.update(identical_sequences=same_seq, identical_text=same_text, clips=len(base["clips"]),
                         enc_max_abs_diff=max(diffs), enc_min_cosine=min(cosm), mismatches=mism,
                         s3=same_text >= S3_MIN)
        if "wer" in e:
            entry["wer_pct"] = round(100 * e["wer"]["wer"], 4)
            entry["wer_detail"] = e["wer"]
        rows[v] = entry
    bw = rows["fp32"].get("wer_pct")
    for v, r in rows.items():
        if "wer_pct" in r and bw is not None:
            r["delta_wer_pp"] = round(r["wer_pct"] - bw, 4)
            if v != "fp32":
                r["s2"] = r["wer_pct"] <= bw + S2_DELTA_PP
    if bw is not None:
        rows["fp32"]["gap_to_fermion_1.72_pp"] = round(bw - FERMION_CLEAN_WER_PCT, 4)
    rep = {"bar": {"S1": "single encoder.onnx < 2e9 bytes, no external data, checker + C70",
                   "S2": "corpus WER <= fp32 ORT WER + 0.10 pp on test-clean (2620)",
                   "S3": "text identical to fp32 ORT on >= 38 of 40 visible clips"},
           "int8_args": pv.INT8_ARGS, "variants": rows, "versions": versions()}
    with open(os.path.join(report_dir, "wp9s1b_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    L = ["# WP9S.1b report: Phonon-2 encoder shipping variants\n",
         "Bar: " + "; ".join("%s: %s" % kv for kv in rep["bar"].items()) + "\n",
         "| variant | encoder bytes | S1 | WER % | dWER pp | S2 | text-identical/40 | S3 | RTF | peak RSS MB | notes |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def yn(x):
        return "-" if x is None else ("pass" if x else "FAIL")
    for v, r in rows.items():
        notes = "enc max abs diff %.3g, min cos %.6f; enc ms mean %s max %s; dj %s us/step; create %s s" % (
            r.get("enc_max_abs_diff", 0), r.get("enc_min_cosine", 1), r["encoder_ms_mean"], r["encoder_ms_max"],
            r["decoder_joint_us_per_step"], r["session_create_s"])
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            v, r["encoder_bytes"], "n/a (reference)" if v == "fp32" else yn(r["s1"]["s1"]), r.get("wer_pct", "-"),
            r.get("delta_wer_pp", "-"), yn(r.get("s2")), r.get("identical_text", 40), yn(r.get("s3")),
            r["rtf"], r["peak_rss_mb"], notes))
    L.append("\nfp32 gap to Fermion's published 1.72 percent: %s pp\n" % rows["fp32"].get("gap_to_fermion_1.72_pp"))
    for v, r in rows.items():
        L.append("## %s encoder ops\n%s\n" % (v, ", ".join("%s x%d" % kv for kv in r["op_types_encoder"].items())))
        if r.get("mismatches"):
            L.append("mismatching clips (first differing step): %s\n" % json.dumps(r["mismatches"]))
    with open(os.path.join(report_dir, "wp9s1b_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-clips", choices=["visible"], help="write the clip list and exit")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--model-dir", default=None, help="ONNX folder (default <dev>/models/stt/phonon-2-onnx/<variant>)")
    ap.add_argument("--clips", help="clip list JSON (rows: id, path relative to <dev>, ref)")
    ap.add_argument("--parity-clips", type=int, default=5)
    ap.add_argument("--report-dir", default=None, help="default <dev>/logs/m9")
    ap.add_argument("--report-name", default="wp9s1a_report")
    ap.add_argument("--no-edge", action="store_true", help="skip the edge checks")
    ap.add_argument("--wer", action="store_true",
                    help="WP9S.1b: eval --variant on the 40 clips, then (with --clips all) full test-clean WER")
    ap.add_argument("--variant-report", action="store_true", help="WP9S.1b: combine wp9s1b_eval_*.json into wp9s1b_report")
    a = ap.parse_args(argv)

    from stt import clips as cl
    from stt import phonon2_export as ex
    p = ex.dev_paths()
    dev = p["dev"]
    if a.make_clips:
        rows, out = cl.make_visible(dev)
        print("wrote %d clips to %s" % (len(rows), out))
        return 0
    if a.variant_report:
        variant_report(a.report_dir or os.path.join(dev, "logs", "m9"), dev, p)
        return 0
    if a.wer:
        warnings.filterwarnings("ignore")
        vis = os.path.join(dev, "goldens", "stt", "clips_visible.json")
        eval_variant(a.variant, vis if a.clips in (None, "all") else a.clips, a.clips == "all",
                     a.report_dir or os.path.join(dev, "logs", "m9"), dev, p)
        return 0
    if not a.clips:
        ap.error("--clips is required (or --make-clips visible)")
    folder = a.model_dir or os.path.join(p["out_root"], a.variant)
    rows = cl.load_clips(a.clips)
    warnings.filterwarnings("ignore")

    from stt.tdt_greedy import OrtTdt
    ex.assert_c70(folder)
    print("loading HF reference from the container ...", flush=True)
    hf = HfRef(p["container"], p["base_dir"])
    ort = OrtTdt(folder)

    print("graph parity ...", flush=True)
    gp = graph_parity(hf, ort, rows, dev, a.parity_clips)
    print(json.dumps(gp["decoder_joint"]), flush=True)

    print("loop parity on %d clips ..." % len(rows), flush=True)
    results = []
    mism = []
    out_rows = []
    for row in rows:
        wav = cl.read_audio(cl.clip_abs_path(row, dev))
        r = run_clip(hf, ort, wav)
        results.append((row, r))
        out_rows.append({"id": row["id"], "seconds": round(len(wav) / 16000, 3), "steps": len(r["seq_o"]) - 1,
                         "tokens": sum(1 for k in r["seq_o"][1:] if k != 8192), "match": r["match"],
                         "hf_ms": r["hf_ms"], "ort_ms": r["enc_ms"] + r["dec_ms"], "enc_ms": r["enc_ms"],
                         "dec_ms": r["dec_ms"], "frames": r["frames"], "step_count": r["step_count"],
                         "step_ns": r["step_ns"], "text": r["text_o"]})
        if not r["match"]:
            mism.append(dict(id=row["id"], **describe_mismatch(r)))
        print("%s %s %.0f ms hf / %.0f ms ort  %s" % (row["id"], "OK " if r["match"] else "BAD", r["hf_ms"],
                                                    r["enc_ms"] + r["dec_ms"], r["text_o"][:60]), flush=True)
    matched = sum(1 for _, r in results if r["match"])

    edge = None if a.no_edge else edge_checks(hf, ort, dev, rows, results)

    secs = sum(r["seconds"] for r in out_rows)
    steps = sum(r["step_count"] for r in out_rows)
    rep = {
        "variant": a.variant, "folder": folder, "clips_file": os.path.abspath(a.clips),
        "files": json.load(open(os.path.join(folder, "SOURCE.json"), encoding="utf-8"))["files"],
        "graphs": op_report(folder), "graph_parity": gp,
        "loop_parity": {"clips": len(rows), "matched": matched, "mismatches": mism},
        "clips": out_rows,
        "timings": {"threads": "ORT default", "encoder_ms_per_clip": [round(r["enc_ms"], 1) for r in out_rows],
                    "encoder_ms_mean": round(sum(r["enc_ms"] for r in out_rows) / len(out_rows), 1),
                    "decoder_joint_mean_us_per_step": round(sum(r["step_ns"] for r in out_rows) / max(1, steps) / 1000, 1),
                    "audio_seconds": round(secs, 2),
                    "real_time_factor_ort": round(sum(r["enc_ms"] + r["dec_ms"] for r in out_rows) / 1000 / secs, 4),
                    "real_time_factor_hf": round(sum(r["hf_ms"] for r in out_rows) / 1000 / secs, 4)},
        "edge_checks": edge, "peak_rss_mb": peak_rss_mb(), "versions": versions(),
    }
    rdir = a.report_dir or os.path.join(dev, "logs", "m9")
    os.makedirs(rdir, exist_ok=True)
    jp = os.path.join(rdir, a.report_name + ".json")
    with open(jp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
        f.write("\n")
    write_md(rep, os.path.join(rdir, a.report_name + ".md"))
    print("%d/%d clips identical. report: %s" % (matched, len(rows), jp))
    return 0 if matched == len(rows) else 3


if __name__ == "__main__":
    sys.exit(main())
