# TTS exploration: English G2P bench

Kokoro-82M (Apache-2.0) needs text turned into misaki phonemes first. Kokoro's own front end falls back to
espeak-ng, which is GPL, so it cannot ship in the plugin. This bench measures C++ front ends without espeak
against misaki + espeak-ng, the front end Kokoro v1.0 was trained with. Exploration only: nothing here is
packaged (`Config/FilterPlugin.ini` lists the Tools files that ship, and none of these).

## Systems

- **larroy** — [larroy/kokoro.cpp](https://github.com/larroy/kokoro.cpp) English G2P as-is (MIT): CMUdict
  (BSD-style) + g2p_en neural guesser (Apache-2.0), ARPAbet mapped to misaki symbols.
- **misaki lexicon port** — `cpp/misaki_lex.h`: a context-free C++ port of misaki's English lexicon lookup
  (misaki's `us_gold`/`us_silver` dictionaries, Apache-2.0) with stemming, capitalization stress and the
  the/a/an/to/in/I cases, plus clitic and compound splitting. Unknown words go to larroy's g2p_en guesser.
- **misaki (Python, no espeak)** — the ceiling for a perfect lexicon port: unknown words come out as ❓.

## Run

Python 3.12 (`pip install -r Tools/tts/requirements.txt`), CMake and a C++17 compiler. Output goes to
`<dev>/tts/` (`MODELPIPELINES_DEV_DIR` or `../.pipelines-dev`).

1. `setup_g2p_bench.py [--with-model]` clones larroy/kokoro.cpp at a pinned commit, builds `en_dump` and
   `misaki_dump` (G2P sources only, no ONNX Runtime), converts misaki's lexicon to `misaki_us.tsv`, and with
   `--with-model` downloads Kokoro v1.0 ONNX + voices (sha256-checked) into `<dev>/models/kokoro/`.
2. `compare.py` runs all systems on `corpus.tsv` (170 game-style lines in 11 categories) and prints exact-line,
   phoneme error (PER) and word error rates per category; details go to `<dev>/tts/report.json`.
3. `synth.py` writes 7 WAVs to `<dev>/tts/samples/`: each line spoken from the reference, larroy and the port.

## Results (2026-10-18)

Word error against misaki + espeak-ng:

| Category | larroy | misaki port | misaki, no espeak |
|---|---|---|---|
| dialogue | 49.1% | 0.4% | 0.0% |
| contraction | 55.8% | 3.2% | 3.2% |
| homograph | 48.8% | 10.6% | 0.0% |
| number | 45.1% | 13.4% | 0.7% |
| jargon | 51.8% | 7.0% | 5.3% |
| name | 49.2% | 17.4% | 28.0% |
| caps | 23.7% | 5.3% | 0.0% |
| interjection | 52.9% | 15.7% | 9.8% |
| punct | 39.3% | 3.6% | 0.0% |
| scifi | 44.6% | 3.1% | 0.0% |
| hyphen | 56.2% | 0.0% | 0.0% |
| **overall** | **48.3%** (8/170 lines exact) | **7.5%** (107/170) | **4.5%** (137/170) |

- **larroy is off in a systematic way.** CMUdict citation forms give every function word primary stress
  ("tˈu", "jˈu", "ˈɪf") and nothing is reduced (no ᵊ, flap T, "the" before vowels). The same line comes out
  about 10% longer from Kokoro.
- **Homographs and "that"/"there" stress** need part-of-speech tags; the port takes misaki's DEFAULT reading.
- **Numbers are broken in larroy's normalizer** (used by both C++ systems): "1,250" and "12,000" split at the
  comma, "1999" is not read as a year, "3rd" reads "three r-d", % and $ are dropped, "Kepler-442b" reads
  "minus four hundred…".
- **Fantasy names are mostly compounds** (Oak+hollow, Silver+brook, Thistle+wood). Splitting them into
  dictionary words fixed the worst g2p_en guesses. Names have no ground truth; espeak's guesses are not
  better, so the name score measures disagreement only.
- **Interjections**: the reference is wrong too (espeak spells "Shh" as "S-H-H"); they need a small dictionary.
- Kokoro v1.0 fp32 on 4 vCPUs (Linux, onnxruntime 1.31): about 2x faster than real time.

## Next

1. Number normalizer rewrite (misaki's `get_number` as the reference).
2. A small POS tagger for homographs and function-word stress.
3. A game dictionary for interjections and jargon (PvP, AoE, Lv.).
4. Load the same Kokoro ONNX file through NNE (operators, phoneme durations for visemes).
