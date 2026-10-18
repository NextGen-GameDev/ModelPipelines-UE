#!/usr/bin/env python
"""Checks that the C++ misaki port (cpp/misaki_g2p.h) reproduces Python misaki exactly when both see spaCy's
tokens and tags: misaki_dump --oracle with no fallback against misaki G2P(fallback=None, unk='').

Lines: corpus.tsv, NUMBER_LINES (number reading stress test) and, with --prose N, N sentences of the held-out
public-domain book from train_pos_tagger.py (needs its download). Exits 1 on any difference.
Needs setup_g2p_bench.py first.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import setup_g2p_bench as setup  # noqa: E402

NUMBER_LINES = [
    "I have 3 potions.", "That will be 1,250 gold coins.", "The war ended in 1999.", "You are the 3rd person.",
    "We lost 50% of it.", "It costs $5 to cross.", "It costs $5.50 to cross.", "He paid $1 and 5 cents.",
    "£20 please.", "It's €3.99.", "Only 0.5 seconds left.", "The year is 2077.", "Population: 12,000.",
    "Combo x3!", "Code 4 8 15 16 23 42.", "Room 101.", "Room 105.", "Agent 007.", "Call 555-0199 now.",
    "Meet at 10:30.", "Kepler-442b has gone dark.", "It was the 1990s.", "He was 21st.", "The 100th time.",
    "It's 1,000,000 strong.", "Year 2005 and 1905 and 2000 and 1800.", "Take -5 damage.", "-3 degrees.",
    "He scored 3.14159 points.", "Version 2.0.1 is out.", "It is .5 percent.", "Chapter 12.", "Floor 2B.",
    "A 2-for-1 deal.", "R2D2 beeps.", "Area 51.", "B2B deals.", "The 49ers won.", "5th of May.", "Wave 7 of 10.",
    "1st, 2nd, 3rd and 4th.", "12th 13th 22nd 101st 111th.", "I'm 2 tired.", "He got 2nd.", "It's 10x better.",
    "$100 million.", "Pay $0.99 now.", "That's 99¢.", "3D printing.", "A 4K screen.", "COVID-19.",
    "The 80s were wild.", "Over 9000!",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prose", type=int, default=0, help="also check N held-out prose sentences")
    args = ap.parse_args()
    from misaki import en

    lines = [text for _, text in compare.load_corpus()] + NUMBER_LINES
    if args.prose:
        import train_pos_tagger
        prose = train_pos_tagger.sentences(train_pos_tagger.fetch_books()[train_pos_tagger.HELDOUT_BOOK])
        random.Random(1).shuffle(prose)
        lines += prose[:args.prose]
    g2p = en.G2P(trf=False, british=False, fallback=None, unk="")
    ref = [g2p(t)[0] for t in lines]
    tokens = [compare.spacy_tokens(g2p.nlp, t) for t in lines]
    out = compare.run_dump([setup.exe("misaki_dump"), os.path.join(setup.tts_dir(), "misaki_us.tsv"), "-", "--oracle"],
                           tokens)
    bad = [(t, r, o) for t, r, o in zip(lines, ref, out) if r.strip() != o.strip()]
    print(f"oracle parity: {len(lines) - len(bad)}/{len(lines)} lines identical")
    for t, r, o in bad[:20]:
        print(f"  {t}\n    misaki {r}\n    port   {o}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
