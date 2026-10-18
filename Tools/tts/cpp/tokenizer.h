#pragma once
// spaCy-like English tokenizer for the G2P bench: whitespace split, then prefix/suffix punctuation, clitics
// ("do n't", "I 'm", "captain 's", "dogs '"), infix hyphens/dashes/ellipses between words, currency and percent,
// abbreviations kept with their period ("Mr.", "U.S."). An approximation of spaCy's rules: the tagger is trained
// on this tokenizer's output (train_pos_tagger.py), so training and runtime see the same tokens.
#include "misaki_g2p.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mg2p {

inline std::vector<Token> tokenize(const U& text) {
    static const std::unordered_set<U> ABBREVIATIONS = {
        U"mr.", U"mrs.", U"ms.", U"dr.", U"st.", U"jr.", U"sr.", U"vs.", U"etc.", U"e.g.", U"i.e.", U"prof.",
        U"gen.", U"col.", U"lt.", U"sgt.", U"capt.", U"gov.", U"rev.", U"vol.", U"approx.", U"dept.",
        U"inc.", U"ltd.", U"co.", U"mt.", U"ft.", U"jan.", U"feb.", U"mar.", U"apr.", U"aug.", U"sep.", U"sept.",
        U"oct.", U"nov.", U"dec.", U"a.m.", U"p.m."};
    static const std::vector<U> CLITICS = {U"n't", U"N'T", U"'s", U"'S", U"'m", U"'M", U"'re", U"'RE", U"'ve",
                                           U"'VE", U"'ll", U"'LL", U"'d", U"'D"};
    // Words spaCy keeps whole although they end like a clitic stem.
    static const std::unordered_set<U> SPECIAL = {U"'s", U"'S", U"y'all", U"Y'all", U"ma'am", U"Ma'am", U"o'clock",
                                                  U"O'clock", U"'em", U"'tis", U"'twas"};
    // spaCy's tokenizer exceptions that split one word in two (lowercase key; case is kept from the text).
    static const std::unordered_map<U, size_t> SPLITS = {{U"gonna", 3}, {U"gotta", 3}, {U"wanna", 3},
                                                          {U"cannot", 3}, {U"c'mon", 3}, {U"c’mon", 3}};
    const U PREFIX = U"\"“‘'([{«¿¡$£€¥#*`";
    const U SUFFIX = U"\"”’'.,;:!?)]}»%…*";
    auto apos = [](char32_t c) { return c == U'\'' || c == U'’'; };
    auto norm = [&](U s) { for (auto& c : s) if (c == U'’') c = U'\''; return s; };

    std::vector<Token> out;
    auto emit = [&](const U& t) { if (!t.empty()) { Token tk; tk.text = t; out.push_back(tk); } };

    // Splits one whitespace-free chunk; trailing whitespace is set by the caller on the last token.
    std::function<void(U)> split_chunk = [&](U s) {
        std::vector<U> suffixes;
        while (!s.empty()) {
            const U n = norm(s);
            if (SPECIAL.count(n) || ABBREVIATIONS.count(lower(n))) break;
            // prefixes
            if (in(PREFIX, s[0]) && s.size() > 1) {
                emit(s.substr(0, 1));
                s = s.substr(1);
                continue;
            }
            // "..." and "…" at the end
            if (ends_with(s, U"...") && s.size() > 3) { suffixes.push_back(U"..."); s = s.substr(0, s.size() - 3); continue; }
            // single-letter abbreviation "J." and dotted letters "U.S." keep their final period
            if (s.back() == U'.' && s.size() >= 2) {
                bool dotted = true;
                for (size_t i = 0; i + 1 < s.size(); ++i) {
                    const bool letter_slot = (i % 2 == 0);
                    if (letter_slot ? !is_letter(s[i]) : s[i] != U'.') { dotted = false; break; }
                }
                if (dotted && s.size() % 2 == 0) break;
            }
            // suffix punctuation (not a decimal point inside a number)
            if (in(SUFFIX, s.back()) && s.size() > 1) { suffixes.push_back(s.substr(s.size() - 1)); s.pop_back(); continue; }
            // clitics (can't -> ca n't, won't -> wo n't, as spaCy)
            bool clitic = false;
            for (const U& c : CLITICS) {
                if (s.size() > c.size() && ends_with(norm(s), norm(c))) {
                    suffixes.push_back(s.substr(s.size() - c.size()));
                    s = s.substr(0, s.size() - c.size());
                    clitic = true;
                    break;
                }
            }
            if (clitic) continue;
            // plural possessive "dogs'"
            if (s.size() > 2 && apos(s.back()) && (s[s.size() - 2] == U's' || s[s.size() - 2] == U'S')) {
                suffixes.push_back(s.substr(s.size() - 1));
                s.pop_back();
                continue;
            }
            break;
        }
        // infixes: hyphen or slash between letters, em/en dash, "--", ellipsis between words
        U cur;
        for (size_t i = 0; i < s.size(); ++i) {
            const char32_t c = s[i];
            const bool prev_letter = i > 0 && is_letter(s[i - 1]);
            const bool next_letter = i + 1 < s.size() && is_letter(s[i + 1]);
            if ((c == U'-' || c == U'/') && prev_letter && next_letter) { emit(cur); cur.clear(); emit(U(1, c)); continue; }
            if (c == U'—' || c == U'–' || c == U'…') { emit(cur); cur.clear(); emit(U(1, c)); continue; }
            if (c == U'-' && i + 1 < s.size() && s[i + 1] == U'-') { emit(cur); cur.clear(); emit(U"--"); ++i; continue; }
            cur += c;
        }
        if (auto sp = SPLITS.find(lower(cur)); sp != SPLITS.end()) {
            emit(cur.substr(0, sp->second));
            emit(cur.substr(sp->second));
        } else {
            emit(cur);
        }
        for (auto it = suffixes.rbegin(); it != suffixes.rend(); ++it) emit(*it);
    };

    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == U' ' || text[i] == U'\t' || text[i] == U'\n')) ++i;
        size_t j = i;
        while (j < text.size() && !(text[j] == U' ' || text[j] == U'\t' || text[j] == U'\n')) ++j;
        if (j == i) break;
        split_chunk(text.substr(i, j - i));
        if (!out.empty() && j < text.size()) out.back().whitespace = U" ";
        i = j;
    }
    return out;
}

}  // namespace mg2p
