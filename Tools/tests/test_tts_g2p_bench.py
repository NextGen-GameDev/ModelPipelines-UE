"""Offline checks for the TTS G2P bench (Tools/tts): corpus format, metrics, lexicon growth."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts"))
import compare  # noqa: E402
import setup_g2p_bench  # noqa: E402

CATEGORIES = {"dialogue", "contraction", "homograph", "number", "jargon", "name", "caps", "interjection", "punct",
              "scifi", "hyphen"}


def test_corpus_rows_are_category_tab_text_without_duplicates():
    rows = compare.load_corpus()
    assert len(rows) == 170
    assert all(len(r) == 2 and r[0] in CATEGORIES and r[1].strip() == r[1] and r[1] for r in rows)
    texts = [r[1] for r in rows]
    assert len(set(texts)) == len(texts)


def test_synth_picks_are_corpus_lines():
    import synth
    texts = {r[1] for r in compare.load_corpus()}
    assert set(synth.PICKS.values()) <= texts


def test_phones_drops_spaces_punctuation_and_optionally_stress():
    assert compare.phones("həlˈO ðˈɛɹ, tɹˈævᵊləɹ!") == "həlˈOðˈɛɹtɹˈævᵊləɹ"
    assert compare.phones("həlˈO ˌI", stress=False) == "həlOI"


def test_lev_counts_code_point_edits():
    assert compare.lev("ðə", "ði") == 1
    assert compare.lev("", "abc") == 3
    assert compare.lev("ˈæʃ", "ˈæʃ") == 0


def test_word_diffs_lists_changed_words_and_ref_length():
    diffs, n = compare.word_diffs("ˌI ɹˈɛd ðə bˈʊk.", "ˈI ɹˈɛd ðə bˈʊk.")
    assert diffs == [("ˌI", "ˈI")]
    assert n == 4


def test_grow_adds_capitalized_and_lowercase_forms_like_misaki():
    grown = setup_g2p_bench.grow({"dragon": "dɹˈæɡən", "Bellini": "bəlˈini", "a": "ɐ", "GPU": "x"})
    assert grown["Dragon"] == "dɹˈæɡən"
    assert grown["bellini"] == "bəlˈini"
    assert "A" not in grown and "gpu" not in grown


# misaki's US_VOCAB plus T (the flap after misaki's ɾ -> T rewrite).
US_PHONEMES = set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔT")


def _data_rows(name):
    path = os.path.join(os.path.dirname(compare.__file__), "data", name)
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]


def test_game_words_use_only_kokoro_us_phonemes_and_give_a_reason():
    rows = _data_rows("game_words.tsv")
    assert rows
    for word, ps, why in rows:
        assert word and why
        assert set(ps) <= US_PHONEMES, (word, set(ps) - US_PHONEMES)
    assert len({r[0] for r in rows}) == len(rows)


def test_known_diffs_have_reference_ours_and_reason():
    assert all(len(r) == 3 and r[0] != r[1] and r[2] for r in _data_rows("known_diffs.tsv"))
    assert compare.load_known_diffs() == {(r[0], r[1]) for r in _data_rows("known_diffs.tsv")}


def test_tagger_features_normalize_and_shape():
    import train_pos_tagger as t
    assert t.normalize("half-elf") == "!HYPHEN"
    assert t.normalize("1999") == "!YEAR"
    assert t.normalize("42") == "!DIGITS"
    assert t.normalize("Dragon") == "dragon"
    assert [t.shape(w) for w in ("NPC", "Elf", "elf", "3rd", "!", "I")] == ["X", "Xx", "x", "d", "p", "Xx"]
    context = t.START + ["the", "dragon", "sleeps"] + t.END
    feats = t.FEATURES(3, "dragon", context, "DT", "-START-")
    assert len(feats) == 15 and "i suffix gon" in feats and "i-1 word the" in feats and "i+2 word -END-" in feats
