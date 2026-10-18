#pragma once
// Context-free C++ port of misaki's English Lexicon (misaki/en.py, Apache-2.0) for the G2P bench.
// Ported: gold/silver lookup, capitalization stress, -s/-ed/-ing stemming, letter spelling (get_NNP),
// the/a/an/to/in/am/I special cases that need only the next word, hyphen sub-token stress, ɾ->T, ʔ->t.
// Added (not in misaki): clitic split ("should've") and compound split of unknown words into 2-3 gold
// words ("Oakhollow" = oak + hollow) before the neural guess.
// Not ported: POS tags (homographs take DEFAULT), misaki's number reader (digits go through larroy's
// normalize_numbers first), currency. Words the lexicon cannot resolve go to `oov`.
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <functional>
#include <locale>
#include <optional>
#include <tuple>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlex {

using U = std::u32string;

inline U to_u32(const std::string& s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.from_bytes(s); }
inline std::string to_u8(const U& s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.to_bytes(s); }

const U VOWELS = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const U CONSONANTS = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const U DIPHTHONGS = U"AIOQWYʤʧ";
const U US_TAUS = U"AIOWYiuæɑəɛɪɹʊʌ";
const U NON_QUOTE_PUNCTS = U";:,.!?—…";
constexpr char32_t PRI = U'ˈ', SEC = U'ˌ';

inline bool in(const U& set, char32_t c) { return set.find(c) != U::npos; }
inline bool has(const U& s, char32_t c) { return s.find(c) != U::npos; }
inline bool is_upper(char32_t c) { return c >= 'A' && c <= 'Z'; }
inline bool is_lower(char32_t c) { return c >= 'a' && c <= 'z'; }
inline bool is_alpha(char32_t c) { return is_upper(c) || is_lower(c); }
inline U lower(U s) { for (auto& c : s) if (is_upper(c)) c += 32; return s; }
inline U upper(U s) { for (auto& c : s) if (is_lower(c)) c -= 32; return s; }
inline bool ends_with(const U& s, const U& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; }
inline U erase_all(U s, char32_t c) { s.erase(std::remove(s.begin(), s.end(), c), s.end()); return s; }
inline U replace_char(U s, char32_t a, char32_t b) { std::replace(s.begin(), s.end(), a, b); return s; }

inline int stress_weight(const U& ps) { int w = 0; for (char32_t c : ps) w += in(DIPHTHONGS, c) ? 2 : 1; return w; }

// Moves each stress mark to just before the next vowel (misaki apply_stress.restress).
inline U restress(const U& ps) {
    std::vector<std::pair<double, char32_t>> ips;
    for (size_t i = 0; i < ps.size(); ++i) {
        double key = static_cast<double>(i);
        if (ps[i] == PRI || ps[i] == SEC) {
            for (size_t j = i; j < ps.size(); ++j) if (in(VOWELS, ps[j])) { key = j - 0.5; break; }
        }
        ips.emplace_back(key, ps[i]);
    }
    std::stable_sort(ips.begin(), ips.end(), [](auto& a, auto& b) { return a.first < b.first; });
    U out; for (auto& p : ips) out += p.second; return out;
}

inline U apply_stress(const U& ps, std::optional<double> stress) {
    if (!stress) return ps;
    const double s = *stress;
    const bool any_stress = has(ps, PRI) || has(ps, SEC);
    const bool any_vowel = std::any_of(ps.begin(), ps.end(), [](char32_t c) { return in(VOWELS, c); });
    if (s < -1) return erase_all(erase_all(ps, PRI), SEC);
    if (s == -1 || ((s == 0 || s == -0.5) && has(ps, PRI))) return replace_char(erase_all(ps, SEC), PRI, SEC);
    if ((s == 0 || s == 0.5 || s == 1) && !any_stress) return any_vowel ? restress(U(1, SEC) + ps) : ps;
    if (s >= 1 && !has(ps, PRI) && has(ps, SEC)) return replace_char(ps, SEC, PRI);
    if (s > 1 && !any_stress) return any_vowel ? restress(U(1, PRI) + ps) : ps;
    return ps;
}

struct Ctx { std::optional<bool> future_vowel; };

class Lexicon {
public:
    using Oov = std::function<std::optional<U>(const U& word)>;

    Lexicon(const std::string& tsv, Oov oov) : oov_(std::move(oov)) {
        std::ifstream f(tsv);
        std::string line;
        while (std::getline(f, line)) {
            const size_t a = line.find('\t'), b = line.find('\t', a + 1);
            auto& d = line[a + 1] == 'g' ? gold_ : silver_;
            d.emplace(to_u32(line.substr(0, a)), to_u32(line.substr(b + 1)));
        }
    }

    // One word (letters, apostrophes, hyphens). Empty optional: unresolved even by `oov`.
    std::optional<U> word(const U& w, const Ctx& ctx) const {
        const bool all_lower = w == lower(w);
        std::optional<double> stress;
        if (!all_lower) stress = (w == upper(w)) ? 2.0 : 0.5;
        if (auto ps = get_word(w, stress, ctx)) return ps;
        // Hyphenated: resolve each part, joined with no space; demote weaker primaries (resolve_tokens).
        if (w.find(U'-') != U::npos) {
            std::vector<U> parts; U cur;
            for (char32_t c : w) { if (c == U'-') { if (!cur.empty()) parts.push_back(cur); cur.clear(); } else cur += c; }
            if (!cur.empty()) parts.push_back(cur);
            std::vector<U> pss;
            for (auto& p : parts) { auto ps = word(p, ctx); if (!ps) return std::nullopt; pss.push_back(*ps); }
            demote(pss);
            U out; for (auto& p : pss) out += p; return out;
        }
        if (w.size() > 1 && w.back() == U'\'') return word(w.substr(0, w.size() - 1), ctx);  // "dogs'", "movin'"
        if (auto ps = clitic(w, ctx)) return ps;
        if (auto ps = compound(w)) return ps;
        if (std::all_of(w.begin(), w.end(), [](char32_t c) { return is_alpha(c) || c == U'\''; })) return oov_(w);
        return std::nullopt;
    }

private:
    std::unordered_map<U, U> gold_, silver_;
    Oov oov_;

    // "should've", "we'd've": host word + clitic, as misaki's sub-tokenizer splits them.
    std::optional<U> clitic(const U& w, const Ctx& ctx) const {
        static const std::vector<std::pair<U, U>> CLITICS = {
            {U"'ve", U"əv"}, {U"'d", U"d"}, {U"'ll", U"l"}, {U"'re", U"əɹ"}, {U"'m", U"m"}};
        for (auto& [suffix, ps] : CLITICS) {
            if (w.size() > suffix.size() + 1 && ends_with(lower(w), suffix)) {
                auto host = word(w.substr(0, w.size() - suffix.size()), ctx);
                if (host) return *host + ps;
            }
        }
        return std::nullopt;
    }

    // Unknown compounds of known words ("Oakhollow", "Silverbrook"): best split into 2-3 gold words
    // of 3+ letters, longest first part wins; primary stress on the first part, secondary after.
    std::optional<U> compound(const U& w) const {
        if (w.size() < 6 || !std::all_of(w.begin(), w.end(), is_alpha)) return std::nullopt;
        const U wl = lower(w);
        std::function<std::optional<std::vector<U>>(size_t, int)> split = [&](size_t from, int parts) -> std::optional<std::vector<U>> {
            if (from == wl.size()) return std::vector<U>{};
            if (parts == 0) return std::nullopt;
            for (size_t end = wl.size(); end >= from + 3; --end) {
                const U part = wl.substr(from, end - from);
                if (!gold(part)) continue;
                if (auto rest = split(end, parts - 1)) { rest->insert(rest->begin(), part); return rest; }
            }
            return std::nullopt;
        };
        auto parts = split(0, 3);
        if (!parts || parts->size() < 2) return std::nullopt;
        U out;
        for (size_t i = 0; i < parts->size(); ++i) {
            U ps = *lookup((*parts)[i], std::nullopt);
            out += i == 0 ? apply_stress(ps, 2) : replace_char(ps, PRI, SEC);
        }
        return out;
    }

    static void demote(std::vector<U>& pss) {
        struct I { bool pri; int w; size_t i; };
        std::vector<I> idx;
        for (size_t i = 0; i < pss.size(); ++i) if (!pss[i].empty()) idx.push_back({has(pss[i], PRI), stress_weight(pss[i]), i});
        int n_pri = 0; for (auto& x : idx) n_pri += x.pri;
        if (idx.size() < 2 || n_pri <= static_cast<int>(idx.size() + 1) / 2) return;
        std::sort(idx.begin(), idx.end(), [](const I& a, const I& b) { return std::tie(a.pri, a.w, a.i) < std::tie(b.pri, b.w, b.i); });
        for (size_t k = 0; k < idx.size() / 2; ++k) pss[idx[k].i] = apply_stress(pss[idx[k].i], -0.5);
    }

    const U* gold(const U& w) const { auto it = gold_.find(w); return it == gold_.end() ? nullptr : &it->second; }
    const U* silver(const U& w) const { auto it = silver_.find(w); return it == silver_.end() ? nullptr : &it->second; }

    std::optional<U> get_nnp(const U& w) const {
        U ps;
        for (char32_t c : w) {
            if (!is_alpha(c)) continue;
            const U* p = gold(U(1, is_lower(c) ? c - 32 : c));
            if (!p) return std::nullopt;
            ps += *p;
        }
        ps = apply_stress(ps, 0);
        const size_t last = ps.rfind(SEC);
        if (last != U::npos) ps[last] = PRI;
        return ps;
    }

    // Special cases that misaki decides from the next word only; POS-dependent ones use the common tag.
    std::optional<U> special(const U& w, const Ctx& ctx) const {
        static const std::unordered_map<U, U> SYMBOLS = {{U"%", U"percent"}, {U"&", U"and"}, {U"+", U"plus"}, {U"@", U"at"}};
        if (auto it = SYMBOLS.find(w); it != SYMBOLS.end()) return lookup(it->second, std::nullopt);
        if (w == U"a" || w == U"A") return U"ɐ";  // determiner reading (tag DT)
        if (w == U"am" || w == U"Am" || w == U"AM") {
            if (!ctx.future_vowel || w != U"am") return *gold(U"am");
            return U"ɐm";
        }
        if (w == U"an" || w == U"An" || w == U"AN") return U"ɐn";
        if (w == U"I") return U"ˌI";
        if (w == U"to" || w == U"To" || w == U"TO") {
            if (!ctx.future_vowel) return *gold(U"to");
            return *ctx.future_vowel ? U"tʊ" : U"tə";
        }
        if (w == U"in" || w == U"In" || w == U"IN") return ctx.future_vowel ? U"ɪn" : U"ˈɪn";  // tag IN
        if (w == U"the" || w == U"The" || w == U"THE") return (ctx.future_vowel && *ctx.future_vowel) ? U"ði" : U"ðə";
        return std::nullopt;
    }

    bool is_known(const U& w) const {
        if (gold(w) || silver(w)) return true;
        if (!std::all_of(w.begin(), w.end(), is_alpha)) return false;
        if (w.size() == 1) return true;
        if (w == upper(w) && gold(lower(w))) return true;
        const U tail = w.substr(1);
        return tail == upper(tail);
    }

    std::optional<U> lookup(U w, std::optional<double> stress) const {
        if (w == upper(w) && !gold(w)) w = lower(w);
        const U* ps = gold(w);
        if (!ps) ps = silver(w);
        if (!ps) return get_nnp(w);
        return apply_stress(*ps, stress);
    }

    std::optional<U> stem_s(const U& w, std::optional<double> st) const {
        if (w.size() < 3 || w.back() != U's') return std::nullopt;
        U stem;
        if (!ends_with(w, U"ss") && is_known(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
        else if ((ends_with(w, U"'s") || (w.size() > 4 && ends_with(w, U"es") && !ends_with(w, U"ies"))) && is_known(w.substr(0, w.size() - 2))) stem = w.substr(0, w.size() - 2);
        else if (w.size() > 4 && ends_with(w, U"ies") && is_known(w.substr(0, w.size() - 3) + U"y")) stem = w.substr(0, w.size() - 3) + U"y";
        else return std::nullopt;
        auto ps = lookup(stem, st);
        if (!ps || ps->empty()) return std::nullopt;
        const char32_t last = ps->back();
        if (in(U"ptkfθ", last)) return *ps + U"s";
        if (in(U"szʃʒʧʤ", last)) return *ps + U"ᵻz";
        return *ps + U"z";
    }

    std::optional<U> stem_ed(const U& w, std::optional<double> st) const {
        if (w.size() < 4 || w.back() != U'd') return std::nullopt;
        U stem;
        if (!ends_with(w, U"dd") && is_known(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
        else if (w.size() > 4 && ends_with(w, U"ed") && !ends_with(w, U"eed") && is_known(w.substr(0, w.size() - 2))) stem = w.substr(0, w.size() - 2);
        else return std::nullopt;
        auto ps = lookup(stem, st);
        if (!ps || ps->empty()) return std::nullopt;
        const U& s = *ps;
        const char32_t last = s.back();
        if (in(U"pkfθʃsʧ", last)) return s + U"t";
        if (last == U'd') return s + U"ᵻd";
        if (last != U't') return s + U"d";
        if (s.size() < 2) return s + U"ɪd";
        if (in(US_TAUS, s[s.size() - 2])) return s.substr(0, s.size() - 1) + U"ɾᵻd";
        return s + U"ᵻd";
    }

    std::optional<U> stem_ing(const U& w, std::optional<double> st) const {
        if (w.size() < 5 || !ends_with(w, U"ing")) return std::nullopt;
        const U base = w.substr(0, w.size() - 3);
        U stem;
        auto doubled = [&] {
            if (ends_with(w, U"cking")) return true;
            const char32_t a = w[w.size() - 5], b = w[w.size() - 4];
            return a == b && in(U"bcdgklmnprstvxz", a);
        };
        if (w.size() > 5 && is_known(base)) stem = base;
        else if (is_known(base + U"e")) stem = base + U"e";
        else if (w.size() > 5 && doubled() && is_known(w.substr(0, w.size() - 4))) stem = w.substr(0, w.size() - 4);
        else return std::nullopt;
        auto ps = lookup(stem, st);
        if (!ps || ps->empty()) return std::nullopt;
        const U& s = *ps;
        if (s.size() > 1 && s.back() == U't' && in(US_TAUS, s[s.size() - 2])) return s.substr(0, s.size() - 1) + U"ɾɪŋ";
        return s + U"ɪŋ";
    }

    std::optional<U> get_word(U w, std::optional<double> stress, const Ctx& ctx) const {
        if (auto ps = special(w, ctx)) return ps;
        const U wl = lower(w);
        const bool alpha = std::all_of(w.begin(), w.end(), [](char32_t c) { return is_alpha(c) || c == U'\''; });
        if (w.size() > 1 && alpha && w != wl && !gold(w) && !silver(w) &&
            (w == upper(w) || w.substr(1) == lower(w.substr(1))) &&
            (gold(wl) || silver(wl) || stem_s(wl, stress) || stem_ed(wl, stress) || stem_ing(wl, stress))) {
            w = wl;
        }
        if (is_known(w)) return lookup(w, stress);
        if (ends_with(w, U"s'") && is_known(w.substr(0, w.size() - 2) + U"'s")) return lookup(w.substr(0, w.size() - 2) + U"'s", stress);
        if (ends_with(w, U"'") && is_known(w.substr(0, w.size() - 1))) return lookup(w.substr(0, w.size() - 1), stress);
        if (auto ps = stem_s(w, stress)) return ps;
        if (auto ps = stem_ed(w, stress)) return ps;
        if (auto ps = stem_ing(w, stress ? stress : std::optional<double>(0.5))) return ps;
        return std::nullopt;
    }
};

// Sentence -> misaki-style phoneme string. Words are runs of letters/apostrophes/hyphens; everything
// else passes through. Processed right to left so the/to/in/am can see the next word.
inline std::string phonemize(const Lexicon& lex, const std::string& text_u8) {
    U text = replace_char(replace_char(to_u32(text_u8), U'’', U'\''), U'‘', U'\'');
    struct Tok { U text; bool is_word; };
    std::vector<Tok> toks;
    for (size_t i = 0; i < text.size();) {
        auto word_char = [&](size_t k) {
            const char32_t c = text[k];
            if (is_alpha(c)) return true;
            // apostrophe/hyphen only inside or right after a word ("dogs'", "half-elf")
            return (c == U'\'' || c == U'-') && k > 0 && is_alpha(text[k - 1]);
        };
        if (word_char(i) && is_alpha(text[i])) {
            size_t j = i;
            while (j < text.size() && word_char(j)) ++j;
            while (j > i + 1 && text[j - 1] == U'-') --j;
            if (j > i + 1 && text[j - 1] == U'\'' && !(text[j - 2] == U's')) --j;
            toks.push_back({text.substr(i, j - i), true});
            i = j;
        } else {
            toks.push_back({U(1, text[i]), false});
            ++i;
        }
    }
    Ctx ctx;
    std::vector<U> out(toks.size());
    for (size_t k = toks.size(); k-- > 0;) {
        const Tok& t = toks[k];
        if (!t.is_word) {
            out[k] = t.text == U"'" ? U"" : t.text;
            if (in(NON_QUOTE_PUNCTS, t.text[0])) ctx.future_vowel.reset();
            continue;
        }
        auto ps = lex.word(t.text, ctx);
        out[k] = ps ? *ps : U"❓";
        if (ps) {
            for (char32_t c : *ps) {
                if (in(VOWELS, c)) { ctx.future_vowel = true; break; }
                if (in(CONSONANTS, c)) { ctx.future_vowel = false; break; }
            }
        }
    }
    U res; for (auto& o : out) res += o;
    res = replace_char(replace_char(res, U'ɾ', U'T'), U'ʔ', U't');
    return to_u8(res);
}

}  // namespace mlex
