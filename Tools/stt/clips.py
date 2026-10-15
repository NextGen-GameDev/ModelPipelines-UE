"""Clip lists for the STT checks. LibriSpeech layout: <root>/<spk>/<chapter>/<spk>-<chapter>-NNNN.flac + trans.txt.

visible set: one clip per speaker, the utterance whose duration is closest to TARGET_SECONDS (ties -> smaller id).
Row: {"id","path" (relative to <dev>, forward slashes),"seconds","num_samples","ref"}.
"""
from __future__ import annotations

import json
import os

TARGET_SECONDS = 8.0
SAMPLE_RATE = 16000


def _read_trans(chapter_dir):
    refs = {}
    for f in os.listdir(chapter_dir):
        if f.endswith(".trans.txt"):
            with open(os.path.join(chapter_dir, f), encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        uid, _, text = line.partition(" ")
                        refs[uid] = text
    return refs


def scan_split(root, durations=None):
    """All utterances under root -> list of dicts (speaker, id, abs_path, num_samples, ref).
    `durations` (optional) maps utterance id -> num_samples instead of reading audio (used by tests)."""
    import soundfile as sf
    rows = []
    for spk in sorted(os.listdir(root)):
        sdir = os.path.join(root, spk)
        if not os.path.isdir(sdir):
            continue
        for ch in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, ch)
            if not os.path.isdir(cdir):
                continue
            refs = _read_trans(cdir)
            for f in sorted(os.listdir(cdir)):
                if not f.endswith(".flac"):
                    continue
                uid = f[:-5]
                p = os.path.join(cdir, f)
                n = durations[uid] if durations is not None else sf.info(p).frames
                rows.append({"speaker": spk, "id": uid, "abs_path": p, "num_samples": int(n), "ref": refs.get(uid, "")})
    return rows


def select_visible(rows, target=TARGET_SECONDS, sr=SAMPLE_RATE):
    """One row per speaker: min |seconds - target|, ties -> smaller id. Sorted by id."""
    best = {}
    for r in rows:
        key = (abs(r["num_samples"] / sr - target), r["id"])
        cur = best.get(r["speaker"])
        if cur is None or key < cur[0]:
            best[r["speaker"]] = (key, r)
    return sorted((v[1] for v in best.values()), key=lambda r: r["id"])


def to_golden_rows(selected, dev, sr=SAMPLE_RATE):
    out = []
    for r in selected:
        rel = os.path.relpath(r["abs_path"], dev).replace("\\", "/")
        out.append({"id": r["id"], "path": rel, "seconds": round(r["num_samples"] / sr, 3),
                    "num_samples": r["num_samples"], "ref": r["ref"]})
    return out


def make_visible(dev, out_path=None, root=None):
    root = root or os.path.join(dev, "testdata", "stt", "librispeech", "LibriSpeech", "test-clean")
    rows = to_golden_rows(select_visible(scan_split(root)), dev)
    out_path = out_path or os.path.join(dev, "goldens", "stt", "clips_visible.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return rows, out_path


def load_clips(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clip_abs_path(row, dev):
    return os.path.join(dev, *row["path"].split("/"))


def read_audio(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, sr
    return w.mean(axis=1) if w.shape[1] > 1 else w[:, 0]
