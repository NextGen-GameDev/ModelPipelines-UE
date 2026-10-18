#pragma once
// Fallback for words misaki's lexicon cannot resolve (misaki calls espeak-ng here): a trailing apostrophe is
// dropped ("movin'"), short words with capitals inside are spelled ("PvP"), clitics are split off a known host
// ("should've"), unknown compounds of known words are split ("Oakhollow" = oak + hollow), and anything else goes
// to the neural guesser (larroy's g2p_en port).
#include "misaki_g2p.h"

#include <functional>
#include <optional>
#include <vector>

namespace mg2p {

using Guesser = std::function<std::optional<U>(const U& word)>;

// Best split of an unknown word into 2-3 gold words of 3+ letters, longest first part first; primary stress on the
// first part, secondary on the rest (English compound stress).
inline std::optional<U> compound(const Lexicon& lex, const U& w) {
    if (w.size() < 6 || !all_letters(w)) return std::nullopt;
    const U wl = lower(w);
    std::function<std::optional<std::vector<U>>(size_t, int)> split = [&](size_t from, int parts) -> std::optional<std::vector<U>> {
        if (from == wl.size()) return std::vector<U>{};
        if (parts == 0) return std::nullopt;
        for (size_t end = wl.size(); end >= from + 3; --end) {
            const U part = wl.substr(from, end - from);
            if (!lex.in_gold(part)) continue;
            if (auto rest = split(end, parts - 1)) { rest->insert(rest->begin(), part); return rest; }
        }
        return std::nullopt;
    };
    auto parts = split(0, 3);
    if (!parts || parts->size() < 2) return std::nullopt;
    U out;
    for (size_t i = 0; i < parts->size(); ++i) {
        auto ps = lex.lookup_plain((*parts)[i]);
        if (!ps) return std::nullopt;
        out += i == 0 ? apply_stress(*ps, 2) : replace_char(*ps, PRI, SEC);
    }
    return out;
}

inline G2P::Fallback make_fallback(const Lexicon& lex, Guesser guess) {
    return [&lex, guess](const Token& tk) -> std::optional<U> {
        U w = tk.text;
        while (w.size() > 1 && (w.back() == U'\'' || w.back() == U'’')) w.pop_back();
        if (w != tk.text) {
            Token t = tk;
            t.text = w;
            if (auto ps = lex(t, Ctx{})) return ps;
        }
        if (w.size() <= 5 && all_letters(w) && std::any_of(w.begin() + 1, w.end(), is_upper)) {
            if (auto ps = lex.spell(w)) return ps;
        }
        static const std::vector<std::pair<U, U>> CLITICS = {
            {U"'ve", U"əv"}, {U"'d", U"d"}, {U"'ll", U"l"}, {U"'re", U"əɹ"}, {U"'m", U"m"}};
        const U wn = replace_char(w, U'’', U'\'');
        for (auto& [suffix, ps] : CLITICS) {
            if (wn.size() <= suffix.size() + 1 || !ends_with(lower(wn), suffix)) continue;
            Token host = tk;
            host.text = wn.substr(0, wn.size() - suffix.size());
            if (auto h = lex(host, Ctx{})) return *h + ps;
            if (auto h = guess ? guess(host.text) : std::nullopt) return *h + ps;
        }
        if (auto ps = compound(lex, w)) return ps;
        if (guess && std::all_of(wn.begin(), wn.end(), [](char32_t c) { return is_letter(c) || c == U'\''; })) return guess(wn);
        return std::nullopt;
    };
}

}  // namespace mg2p
