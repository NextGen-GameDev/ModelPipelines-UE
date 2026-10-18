#!/usr/bin/env python
"""Compares English G2P candidates against misaki + espeak-ng, Kokoro v1.0's en-us front end, on corpus.tsv.

Systems: larroy/kokoro.cpp as-is (en_dump); the C++ misaki port (misaki_dump) fed spaCy's tokens and tags, the
ceiling for a perfect tagger; the port standalone, with our tokenizer and tagger (the one that could ship); and
Python misaki with no espeak fallback. Both port runs use data/game_words.tsv and fallback.h for unknown words.
Prints exact-line rate, phoneme error rate (Levenshtein over code points, spaces and punctuation removed; with
and without stress marks) and word error rate, overall and per corpus category. Word differences listed in
data/known_diffs.tsv (intentional, where the espeak reference is wrong) are counted separately.
Writes every line's output and word differences to <dev>/tts/report.json (synth.py reads it).
Needs setup_g2p_bench.py and train_pos_tagger.py first. espeak-ng (GPL) is only the evaluation reference here;
nothing ships it.
"""
import collections
import difflib
import json
import os
import re
import subprocess
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_g2p_bench as setup  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set(';:,.!?—…"“”()-\'')


def load_corpus(path=os.path.join(HERE, "corpus.tsv")):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]


def phones(s, stress=True):
    s = "".join(c for c in unicodedata.normalize("NFC", s) if c not in PUNCT and not c.isspace())
    return s if stress else s.replace("ˈ", "").replace("ˌ", "")


def words(s):
    return [w for w in re.split(r'[\s;:,.!?—…"“”()]+', s) if w]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_diffs(ref, hyp):
    rw, hw = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
    diffs = [(" ".join(rw[i1:i2]), " ".join(hw[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    return diffs, len(rw)


def run_dump(cmd, texts):
    out = subprocess.run(cmd, input="\n".join(texts) + "\n", capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    return out.rstrip("\n").split("\n")


def load_known_diffs(path=os.path.join(HERE, "data", "known_diffs.tsv")):
    with open(path, encoding="utf-8") as f:
        return {tuple(line.rstrip("\n").split("\t")[:2]) for line in f if line.strip() and not line.startswith("#")}


def spacy_tokens(nlp, text):
    return "\x1e".join(f"{tk.text}\x1f{tk.tag_}\x1f{1 if tk.whitespace_ else 0}" for tk in nlp(text))


def main():
    from misaki import en, espeak

    cats, texts = zip(*load_corpus())
    ref_g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
    bare_g2p = en.G2P(trf=False, british=False, fallback=None, unk="❓")
    ref = [ref_g2p(t)[0] for t in texts]
    larroy_dict = os.path.join(setup.tts_dir(), "kokoro.cpp", "dict")
    neural = os.path.join(larroy_dict, "g2p_en.weights")
    port = [setup.exe("misaki_dump"), os.path.join(setup.tts_dir(), "misaki_us.tsv"), neural,
            "--game", os.path.join(HERE, "data", "game_words.tsv")]
    systems = {
        "larroy (CMUdict + g2p_en)": run_dump(
            [setup.exe("en_dump"), os.path.join(larroy_dict, "cmudict-0.7b", "cmudict.dict"), neural], texts),
        "port, spaCy tags": run_dump(port + ["--oracle"], [spacy_tokens(ref_g2p.nlp, t) for t in texts]),
        "port, standalone": run_dump(port + ["--tagger", os.path.join(setup.tts_dir(), "pos_tagger.tsv")], texts),
        "misaki (Python, no espeak)": [bare_g2p(t)[0] for t in texts],
    }
    known = load_known_diffs()

    report = {"texts": texts, "cats": cats, "ref": ref, "systems": {}}
    order = list(dict.fromkeys(cats))
    for name, hyp in systems.items():
        assert len(hyp) == len(ref), (name, len(hyp))
        # lines, exact, edits, ref len, word err, ref words, known (intentional) word diffs
        per_cat = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
        tot = [0, 0, 0, 0, 0, 0, 0]
        no_stress = [0, 0]
        lines = []
        for c, t, r, h in zip(cats, texts, ref, hyp):
            e, n = lev(phones(r), phones(h)), len(phones(r))
            no_stress[0] += lev(phones(r, False), phones(h, False))
            no_stress[1] += len(phones(r, False))
            diffs, nw = word_diffs(r, h)
            kn = sum(max(len(a.split()), len(b.split())) for a, b in diffs if (a, b) in known)
            we = sum(max(len(a.split()), len(b.split())) for a, b in diffs if (a, b) not in known)
            for acc in (per_cat[c], tot):
                acc[0] += 1
                acc[1] += e == 0
                acc[2] += e
                acc[3] += n
                acc[4] += we
                acc[5] += nw
                acc[6] += kn
            lines.append({"cat": c, "text": t, "hyp": h, "per": e / max(n, 1), "diffs": diffs})
        report["systems"][name] = {"lines": lines, "total": tot, "cats": dict(per_cat)}
        print(f"\n== {name}")
        print(f"   overall: exact lines {tot[1]}/{tot[0]}  PER {100 * tot[2] / tot[3]:.1f}%  "
              f"PER-no-stress {100 * no_stress[0] / no_stress[1]:.1f}%  word err {100 * tot[4] / tot[5]:.1f}%"
              f"  (+{tot[6]} known diffs)")
        for c in order:
            a = per_cat[c]
            print(f"   {c:13s} exact {a[1]:3d}/{a[0]:3d}  PER {100 * a[2] / a[3]:5.1f}%  word err {100 * a[4] / a[5]:5.1f}%"
                  + (f"  (+{a[6]} known)" if a[6] else ""))
    out = os.path.join(setup.tts_dir(), "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
