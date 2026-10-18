# TTS exploration: English G2P bench

Kokoro-82M (Apache-2.0) needs text turned into misaki phonemes first. Kokoro's own front end (misaki) needs
spaCy and falls back to espeak-ng, which is GPL, so it cannot ship in the plugin. This bench builds a C++ front
end without espeak and measures it against misaki + espeak-ng, the front end Kokoro v1.0 was trained with.
Exploration only: nothing here is packaged (`Config/FilterPlugin.ini` lists the Tools files that ship, and none
of these).

## The C++ front end (`cpp/`)

- `misaki_g2p.h`: port of misaki's English G2P after spaCy: sub-tokenizing, merging tokens written without
  spaces, next-word context ("the" before vowels, "to"), POS-aware lexicon lookup in misaki's `us_gold` and
  `us_silver` dictionaries (Apache-2.0), stemming, number reading with num2words' English rules (years,
  ordinals, currency, decimals, digits inside words), stress resolution. Given spaCy's tokens and tags it
  produces exactly misaki's output (`check_parity.py`).
- `tokenizer.h`: spaCy-like tokenizer (punctuation, clitics, hyphens, abbreviations). Same tokens as spaCy on
  95% of test lines; most differences are Gutenberg's "word--" dashes.
- `pos_tagger.h`: averaged-perceptron tagger (after Matthew Honnibal's design), 7 MB of weights as text.
  `train_pos_tagger.py` distils spaCy en_core_web_sm's tags (MIT) on NLTK's Project Gutenberg selection
  (public domain prose; 5% of sentences repeated in capitals for shouted lines). Agrees with spaCy on 96.9%
  of tags in a held-out book; the C++ and Python taggers give identical tags.
- `fallback.h`: where misaki would call espeak: trailing apostrophes ("movin'"), short mixed-case words spelled
  ("PvP"), clitics ("should've"), compounds of known words ("Oakhollow" = oak + hollow), then larroy's g2p_en
  neural guesser (Apache-2.0).
- `data/game_words.tsv`: 15 interjections and game terms misaki spells out or misreads (Shh, Mmhm, Lv., AoE).

## Run

Python 3.12 (`pip install -r Tools/tts/requirements.txt`), CMake and a C++17 compiler. Output goes to
`<dev>/tts/` (`MODELPIPELINES_DEV_DIR` or `../.pipelines-dev`).

1. `setup_g2p_bench.py [--with-model]` clones larroy/kokoro.cpp at a pinned commit, builds `en_dump` (larroy
   as-is) and `misaki_dump` (the port), converts misaki's lexicon to `misaki_us.tsv`, and with `--with-model`
   downloads Kokoro v1.0 ONNX + voices (sha256-checked) into `<dev>/models/kokoro/`.
2. `train_pos_tagger.py` writes `<dev>/tts/pos_tagger.tsv` (about 25 minutes on 4 cores), then checks held-out
   accuracy and C++/Python parity.
3. `check_parity.py [--prose N]` checks the port against Python misaki with spaCy's tags (corpus, 53 number
   lines, optionally N held-out prose sentences).
4. `compare.py` runs every system on `corpus.tsv` (170 game-style lines in 11 categories); details go to
   `<dev>/tts/report.json`.
5. `synth.py` writes 7 WAVs to `<dev>/tts/samples/`: each line spoken from the reference, larroy and the port.

## Results (2026-10-18)

Word error against misaki + espeak-ng. "Known" differences (`data/known_diffs.tsv`) are intentional: the espeak
reference spells "Shh" as S-H-H, "Lv." as L-V, and so on; they are counted separately.

| Category | larroy as-is | port, spaCy tags | **port, standalone** | misaki, no espeak |
|---|---|---|---|---|
| dialogue | 49.1% | 0.0% | **0.0%** | 0.0% |
| contraction | 55.8% | 1.1% | **1.1%** | 3.2% |
| homograph | 48.8% | 0.0% | **1.8%** | 0.0% |
| number | 45.1% | 0.0% | **0.7%** | 0.7% |
| jargon | 51.8% | 0.9% (+5 known) | **0.9%** (+5 known) | 5.3% |
| name | 49.2% | 17.4% | **17.4%** | 28.0% |
| caps | 23.7% | 0.0% | **0.0%** | 0.0% |
| interjection | 52.9% | 0.0% (+4 known) | **0.0%** (+4 known) | 9.8% |
| punct | 39.3% | 0.0% | **0.0%** | 0.0% |
| scifi | 44.6% | 0.0% | **0.0%** | 0.0% |
| hyphen | 56.2% | 0.0% | **0.0%** | 0.0% |
| **overall** | **48.3%** (8/170 exact) | **2.2%** (143/170) | **2.5%** (139/170) | **4.5%** (137/170) |

- **larroy is off in a systematic way.** CMUdict citation forms give every function word primary stress
  ("tˈu", "jˈu", "ˈɪf") and nothing is reduced; the same line comes out about 10% longer from Kokoro.
- **Outside names, 6 words in 150 lines differ** from the reference: 4 tagger misses ("wind" and "dove" as
  verbs, stressed "that" twice) and 2 stress-only differences ("should've", "cooldown").
- **Names have no ground truth.** Most are compounds (Oak+hollow, Silver+brook) and `fallback.h` splits them;
  espeak's guesses are not better, so the name score measures disagreement only.
- Kokoro v1.0 fp32 on 4 vCPUs (Linux, onnxruntime 1.31): about 2x faster than real time.

## Next

1. Load the same Kokoro ONNX file through NNE (operators, phoneme durations for visemes).
2. Binary tagger weights (the 7 MB text file would shrink to about 2 MB) and a lexicon format the plugin can
   load quickly.
3. More tagger training text closer to game dialogue, if the remaining homograph misses matter in practice.
