#!/usr/bin/env python
"""Trains the C++ port's POS tagger (cpp/pos_tagger.h) and writes <dev>/tts/pos_tagger.tsv.

misaki picks homograph readings ("read" VBD vs VBP, "wind" NOUN vs VERB) and function-word stress from spaCy's
Penn Treebank tags. This script distils those tags into an averaged perceptron (after Matthew Honnibal's
"A good part-of-speech tagger in about 200 lines of Python"):

1. Text: NLTK's Project Gutenberg selection (public domain; zip from the nltk_data GitHub repo, sha256 pinned),
   prose books only (no Bible, Shakespeare, Milton or poetry). carroll-alice.txt is held out for evaluation.
2. Tokens: our C++ tokenizer (misaki_dump --tokens), so training sees exactly the runtime tokens.
3. Labels: spaCy en_core_web_sm (MIT) tags those tokens, as misaki would.
4. Training: 5 passes, seeded shuffle, averaged weights; weights below 0.001 dropped, rounded to 4 places.
5. Checks: held-out tag accuracy against spaCy, and parity: the C++ tagger (misaki_dump --tags) must return
   the same tags as this script for every held-out sentence.

Needs setup_g2p_bench.py first. Takes a few minutes; rerun with --force to retrain.
"""
import argparse
import collections
import hashlib
import os
import random
import re
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_g2p_bench as setup  # noqa: E402

GUTENBERG_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
GUTENBERG_SHA256 = "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611"
TRAIN_BOOKS = ["austen-emma", "austen-persuasion", "austen-sense", "bryant-stories", "burgess-busterbrown",
               "chesterton-ball", "chesterton-brown", "chesterton-thursday", "edgeworth-parents",
               "melville-moby_dick"]
HELDOUT_BOOK = "carroll-alice"
ITERATIONS = 5
SEED = 1234
START, END = ["-START-", "-START2-"], ["-END-", "-END2-"]


# Character classes and lowercase as in cpp/misaki_g2p.h (not Python's full Unicode rules).
def is_upper(c):
    o = ord(c)
    return 65 <= o <= 90 or (0xC0 <= o <= 0xDE and o != 0xD7)


def is_lower(c):
    o = ord(c)
    return 97 <= o <= 122 or (0xDF <= o <= 0xFF and o != 0xF7)


def is_letter(c):
    return is_upper(c) or is_lower(c) or 0x100 <= ord(c) <= 0x24F


def is_digit(c):
    return "0" <= c <= "9"


def lower(w):
    return "".join(chr(ord(c) + 32) if is_upper(c) else c for c in w)


def normalize(w):
    if "-" in w and w[0] != "-":
        return "!HYPHEN"
    if len(w) == 4 and all(is_digit(c) for c in w):
        return "!YEAR"
    if w and is_digit(w[0]):
        return "!DIGITS"
    return lower(w)


def shape(w):
    if not w:
        return "p"
    if len(w) > 1 and all(is_upper(c) for c in w):
        return "X"
    if is_upper(w[0]):
        return "Xx"
    if is_lower(w[0]) or is_letter(w[0]):
        return "x"
    if is_digit(w[0]):
        return "d"
    return "p"


def FEATURES(i, word, context, prev, prev2):
    """Must match PosTagger::features in cpp/pos_tagger.h."""
    return [
        "bias",
        "i suffix " + word[-3:],
        "i pref1 " + word[:1],
        "i-1 tag " + prev,
        "i-2 tag " + prev2,
        "i tag+i-2 tag " + prev + " " + prev2,
        "i word " + context[i],
        "i-1 tag+i word " + prev + " " + context[i],
        "i-1 word " + context[i - 1],
        "i-1 suffix " + context[i - 1][-3:],
        "i-2 word " + context[i - 2],
        "i+1 word " + context[i + 1],
        "i+1 suffix " + context[i + 1][-3:],
        "i+2 word " + context[i + 2],
        "i shape " + shape(word),
    ]


class Perceptron:
    def __init__(self, classes):
        self.classes = sorted(classes)
        self.weights = {}
        self._totals = collections.defaultdict(float)
        self._tstamps = collections.defaultdict(int)
        self.i = 0

    def predict(self, feats):
        scores = dict.fromkeys(self.classes, 0.0)
        for f in feats:
            ws = self.weights.get(f)
            if ws:
                for c, w in ws.items():
                    scores[c] += w
        return max(self.classes, key=lambda c: (scores[c], c))

    def update(self, truth, guess, feats):
        self.i += 1
        if truth == guess:
            return
        for f in feats:
            ws = self.weights.setdefault(f, {})
            for c, delta in ((truth, 1.0), (guess, -1.0)):
                w = ws.get(c, 0.0)
                key = (f, c)
                self._totals[key] += (self.i - self._tstamps[key]) * w
                self._tstamps[key] = self.i
                ws[c] = w + delta

    def average(self):
        for f, ws in self.weights.items():
            avg = {}
            for c, w in ws.items():
                key = (f, c)
                total = self._totals[key] + (self.i - self._tstamps[key]) * w
                a = round(total / self.i, 4)
                if abs(a) >= 0.001:
                    avg[c] = a
            self.weights[f] = avg


class Tagger:
    def __init__(self, classes, tagdict, weights):
        self.classes, self.tagdict, self.weights = sorted(classes), tagdict, weights

    def tag(self, words):
        context = START + [normalize(w) for w in words] + END
        prev, prev2 = START
        out = []
        for i, w in enumerate(words):
            guess = self.tagdict.get(w)
            if guess is None:
                scores = dict.fromkeys(self.classes, 0.0)
                for f in FEATURES(i + 2, w, context, prev, prev2):
                    for c, wt in self.weights.get(f, {}).items():
                        scores[c] += wt
                guess = max(self.classes, key=lambda c: (scores[c], c))
            out.append(guess)
            prev2, prev = prev, guess
        return out

    def save(self, path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("C\t" + "\t".join(self.classes) + "\n")
            for w, t in sorted(self.tagdict.items()):
                f.write(f"D\t{w}\t{t}\n")
            for feat, ws in sorted(self.weights.items()):
                if ws:
                    f.write(f"W\t{feat}\t" + "\t".join(f"{c}={w!r}" for c, w in sorted(ws.items())) + "\n")

    @staticmethod
    def load(path):
        classes, tagdict, weights = [], {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if cols[0] == "C":
                    classes = cols[1:]
                elif cols[0] == "D":
                    tagdict[cols[1]] = cols[2]
                elif cols[0] == "W":
                    weights[cols[1]] = {c: float(w) for c, w in (x.rsplit("=", 1) for x in cols[2:])}
        return Tagger(classes, tagdict, weights)


def fetch_books():
    path = os.path.join(_cache_dir(), "gutenberg.zip")
    if not (os.path.exists(path) and setup.sha256(path) == GUTENBERG_SHA256):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", GUTENBERG_URL)
        urllib.request.urlretrieve(GUTENBERG_URL, path + ".part")
        if setup.sha256(path + ".part") != GUTENBERG_SHA256:
            raise SystemExit("sha256 mismatch for gutenberg.zip")
        os.replace(path + ".part", path)
    with zipfile.ZipFile(path) as z:
        return {name: z.read(f"gutenberg/{name}.txt").decode("latin-1") for name in TRAIN_BOOKS + [HELDOUT_BOOK]}


def _cache_dir():
    return os.path.join(setup._paths.dev_dir(), "cache", "tts")


def sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        out += [s for s in re.split(r"(?<=[.!?])\s+(?=[\"'A-Z])", para) if 3 <= len(s) <= 400]
    return out


def encode(lines, mode, tagger_path=None):
    cmd = [setup.exe("misaki_dump"), "-", "-", f"--{mode}"]
    if tagger_path:
        cmd += ["--tagger", tagger_path]
    out = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    return out.rstrip("\n").split("\n")


def decode(line):
    return [tuple(t.split("\x1f")) for t in line.split("\x1e")] if line else []


def spacy_tags(nlp, token_lines):
    from spacy.tokens import Doc
    docs = []
    for line in token_lines:
        toks = decode(line)
        docs.append(Doc(nlp.vocab, words=[t[0] for t in toks], spaces=[t[2] == "1" for t in toks]))
    return [[tk.tag_ for tk in doc] for doc in nlp.pipe(docs, batch_size=256)]


def labelled(nlp, texts):
    token_lines = encode(texts, "tokens")
    keep = [(line, i) for i, line in enumerate(token_lines) if line]
    tags = spacy_tags(nlp, [line for line, _ in keep])
    return [([t[0] for t in decode(line)], tg, line) for (line, _), tg in zip(keep, tags)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="retrain even if pos_tagger.tsv exists")
    args = ap.parse_args()
    out = os.path.join(setup.tts_dir(), "pos_tagger.tsv")
    import spacy
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])

    books = fetch_books()
    heldout = labelled(nlp, sentences(books[HELDOUT_BOOK]))
    if args.force or not os.path.exists(out):
        train = labelled(nlp, [s for name in TRAIN_BOOKS for s in sentences(books[name])])
        print(f"train: {len(train)} sentences, {sum(len(w) for w, _, _ in train)} tokens")
        counts = collections.defaultdict(collections.Counter)
        for words, tags, _ in train:
            for w, t in zip(words, tags):
                counts[w][t] += 1
        tagdict = {w: c.most_common(1)[0][0] for w, c in counts.items()
                   if sum(c.values()) >= 20 and c.most_common(1)[0][1] / sum(c.values()) >= 0.97}
        model = Perceptron({t for _, tags, _ in train for t in tags})
        rng = random.Random(SEED)
        for it in range(ITERATIONS):
            correct = total = 0
            for words, tags, _ in train:
                context = START + [normalize(w) for w in words] + END
                prev, prev2 = START
                for i, (w, truth) in enumerate(zip(words, tags)):
                    guess = tagdict.get(w)
                    if guess is None:
                        feats = FEATURES(i + 2, w, context, prev, prev2)
                        guess = model.predict(feats)
                        model.update(truth, guess, feats)
                    prev2, prev = prev, guess
                    correct += guess == truth
                    total += 1
            rng.shuffle(train)
            print(f"  pass {it + 1}: {100 * correct / total:.2f}% on training data")
        model.average()
        Tagger(model.classes, tagdict, model.weights).save(out)
        print("wrote", out, f"({os.path.getsize(out) // 1024} KB)")

    tagger = Tagger.load(out)
    predicted = [tagger.tag(words) for words, _, _ in heldout]
    correct = sum(p == t for (_, tags, _), pred in zip(heldout, predicted) for p, t in zip(pred, tags))
    total = sum(len(tags) for _, tags, _ in heldout)
    print(f"held-out ({HELDOUT_BOOK}, {len(heldout)} sentences): {100 * correct / total:.2f}% tags agree with spaCy")

    cpp = [[t[1] for t in decode(line)] for line in encode([line for _, _, line in heldout], "tags", out)]
    mismatches = sum(c != p for c, p in zip(cpp, predicted))
    print(f"C++ parity: {len(heldout) - mismatches}/{len(heldout)} sentences tagged identically")
    if mismatches:
        raise SystemExit("C++ tagger differs from the Python tagger")


if __name__ == "__main__":
    main()
