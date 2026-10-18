#pragma once
// C++ port of misaki's English G2P (misaki/en.py 0.9.4, Apache-2.0) for the G2P bench: sub-tokenizing, token
// merging, next-word context, POS-aware lexicon lookup (us_gold/us_silver), stemming, number reading (num2words
// English rules) and stress resolution. Input is tokens with Penn Treebank tags (spaCy's in oracle mode, our
// tokenizer + tagger otherwise). Words the lexicon cannot resolve go to `fallback`, where misaki would call espeak.
// Not ported: markdown features ([word](/phonemes/) links), British English, the transformer model.
#include <algorithm>
#include <codecvt>
#include <cstdint>
#include <fstream>
#include <functional>
#include <locale>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mg2p {

using U = std::u32string;

inline U to_u32(const std::string& s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.from_bytes(s); }
inline std::string to_u8(const U& s) { return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>{}.to_bytes(s); }

const U VOWELS = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const U CONSONANTS = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const U DIPHTHONGS = U"AIOQWYʤʧ";
const U US_TAUS = U"AIOWYiuæɑəɛɪɹʊʌ";
const U PUNCTS = U";:,.!?—…\"“”";
const U NON_QUOTE_PUNCTS = U";:,.!?—…";
const U SUBTOKEN_JUNKS = U"',-._‘’/";
constexpr char32_t PRI = U'ˈ', SEC = U'ˌ';

inline bool in(const U& set, char32_t c) { return set.find(c) != U::npos; }
inline bool has(const U& s, char32_t c) { return s.find(c) != U::npos; }
// Letters: ASCII plus Latin-1/Latin Extended letters (enough for game text; misaki uses Unicode \p{L}).
inline bool is_upper(char32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7); }
inline bool is_lower(char32_t c) { return (c >= 'a' && c <= 'z') || (c >= 0xDF && c <= 0xFF && c != 0xF7); }
inline bool is_letter(char32_t c) { return is_upper(c) || is_lower(c) || (c >= 0x100 && c <= 0x24F); }
inline bool is_ascii_alpha(char32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
inline bool all_digits(const U& s) { return !s.empty() && std::all_of(s.begin(), s.end(), is_digit); }
inline bool all_letters(const U& s) { return !s.empty() && std::all_of(s.begin(), s.end(), is_letter); }
inline U lower(U s) { for (auto& c : s) if (is_upper(c)) c += 32; return s; }
inline U upper(U s) { for (auto& c : s) if (is_lower(c) && c != 0xDF && c != 0xFF) c -= 32; return s; }
inline U capitalize(const U& s) { if (s.empty()) return s; U r = lower(s); r[0] = upper(U(1, r[0]))[0]; return r; }
inline bool ends_with(const U& s, const U& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; }
inline bool starts_with(const std::string& s, const char* t) { return s.rfind(t, 0) == 0; }
inline U erase_all(U s, char32_t c) { s.erase(std::remove(s.begin(), s.end(), c), s.end()); return s; }
inline U replace_char(U s, char32_t a, char32_t b) { std::replace(s.begin(), s.end(), a, b); return s; }
inline U lstrip(const U& s, char32_t c) { size_t i = 0; while (i < s.size() && s[i] == c) ++i; return s.substr(i); }

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

// ---- num2words (English) -------------------------------------------------------------------------------------
namespace num {

inline std::string small(int n) {
    static const char* LOW[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                "eighteen", "nineteen"};
    static const char* TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    if (n < 20) return LOW[n];
    return std::string(TENS[n / 10]) + (n % 10 ? std::string("-") + LOW[n % 10] : "");
}

// num2words(n): "one thousand, two hundred and fifty", "twenty-one", "one hundred and one".
inline std::string cardinal(uint64_t n) {
    static const std::pair<uint64_t, const char*> SCALES[] = {
        {1000000000000000000ull, "quintillion"}, {1000000000000000ull, "quadrillion"},
        {1000000000000ull, "trillion"}, {1000000000ull, "billion"}, {1000000ull, "million"}, {1000ull, "thousand"}};
    if (n < 100) return small(static_cast<int>(n));
    if (n < 1000) {
        std::string s = small(static_cast<int>(n / 100)) + " hundred";
        return n % 100 ? s + " and " + small(static_cast<int>(n % 100)) : s;
    }
    for (auto& [scale, name] : SCALES) {
        if (n < scale) continue;
        std::string s = cardinal(n / scale) + " " + name;
        const uint64_t rest = n % scale;
        if (!rest) return s;
        return s + (rest < 100 ? " and " : ", ") + cardinal(rest);
    }
    return "";
}

inline std::string ordinal(uint64_t n) {
    static const std::unordered_map<std::string, std::string> ORDS = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"}, {"five", "fifth"},
        {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"}, {"nine", "ninth"}, {"ten", "tenth"},
        {"eleven", "eleventh"}, {"twelve", "twelfth"}};
    std::string s = cardinal(n);
    size_t cut = s.find_last_of(" -");
    cut = cut == std::string::npos ? 0 : cut + 1;
    std::string last = s.substr(cut);
    if (auto it = ORDS.find(last); it != ORDS.end()) last = it->second;
    else { if (last.back() == 'y') last = last.substr(0, last.size() - 1) + "ie"; last += "th"; }
    return s.substr(0, cut) + last;
}

inline std::string year(uint64_t n) {
    const uint64_t high = n / 100, low = n % 100;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return cardinal(n);
    const std::string lowtext = low == 0 ? "hundred" : low < 10 ? "oh-" + cardinal(low) : cardinal(low);
    return cardinal(high) + " " + lowtext;
}

// num2words(float(s)) for "12.5" -> "twelve point five" (precision from the shortest repr, as Python's).
inline std::string decimal(const std::string& s) {
    const size_t dot = s.find('.');
    std::string ip = s.substr(0, dot), fp = s.substr(dot + 1);
    while (fp.size() > 1 && fp.back() == '0') fp.pop_back();
    if (fp.empty()) fp = "0";
    std::string out = cardinal(ip.empty() ? 0 : std::stoull(ip)) + " point";
    for (char c : fp) out += " " + small(c - '0');
    return out;
}

}  // namespace num

// ---- tokens ---------------------------------------------------------------------------------------------------

struct Token {
    U text;
    std::string tag;
    U whitespace;                 // "" or " "
    std::optional<U> phonemes;
    bool is_head = true;
    std::optional<U> alias;
    std::optional<double> stress;
    std::optional<char32_t> currency;
    bool prespace = false;
};

struct Ctx { std::optional<bool> future_vowel; bool future_to = false; };

// misaki's SUBTOKEN_REGEX, hand-written (re.findall semantics: first alternative that matches wins, unmatched
// characters are skipped).
inline std::vector<U> subtokenize(const U& w) {
    std::vector<U> out;
    const size_t n = w.size();
    auto apos = [](char32_t c) { return c == U'\'' || c == U'‘' || c == U'’'; };
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        // ^['‘’]+
        if (i == 0 && apos(w[0])) { while (j < n && apos(w[j])) ++j; out.push_back(w.substr(i, j - i)); i = j; continue; }
        // \p{Lu}(?=\p{Lu}\p{Ll})
        if (is_upper(w[i]) && i + 2 < n && is_upper(w[i + 1]) && is_lower(w[i + 2])) { out.push_back(w.substr(i, 1)); ++i; continue; }
        // (?:^-)?(?:\d?[,.]?\d)+
        {
            size_t k = i;
            if (i == 0 && w[0] == U'-') ++k;
            size_t end = 0;
            while (true) {
                // one repetition of \d?[,.]?\d, greedy with backtracking over the optional parts
                size_t best = 0;
                for (int a = 1; a >= 0 && !best; --a)
                    for (int b = 1; b >= 0 && !best; --b) {
                        size_t p = k;
                        if (a) { if (p < n && is_digit(w[p])) ++p; else continue; }
                        if (b) { if (p < n && (w[p] == U',' || w[p] == U'.')) ++p; else continue; }
                        if (p < n && is_digit(w[p])) best = p + 1;
                    }
                if (!best) break;
                k = end = best;
            }
            if (end) { out.push_back(w.substr(i, end - i)); i = end; continue; }
        }
        // [-_]+
        if (w[i] == U'-' || w[i] == U'_') { while (j < n && (w[j] == U'-' || w[j] == U'_')) ++j; out.push_back(w.substr(i, j - i)); i = j; continue; }
        // ['‘’]{2,}
        if (apos(w[i]) && i + 1 < n && apos(w[i + 1])) { while (j < n && apos(w[j])) ++j; out.push_back(w.substr(i, j - i)); i = j; continue; }
        // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}): shortest letter run ending lowercase before an uppercase letter
        if (is_letter(w[i])) {
            size_t k = i, found = 0;
            while (k < n && (is_letter(w[k]) || (apos(w[k]) && k + 1 < n && is_letter(w[k + 1])))) {
                if (is_lower(w[k]) && k + 1 < n && is_upper(w[k + 1])) { found = k + 1; break; }
                ++k;
            }
            if (found) { out.push_back(w.substr(i, found - i)); i = found; continue; }
            // \p{L}+(?:['‘’]\p{L})*
            while (j < n && is_letter(w[j])) ++j;
            while (j + 1 < n && apos(w[j]) && is_letter(w[j + 1])) j += 2;
            out.push_back(w.substr(i, j - i)); i = j; continue;
        }
        // [^-_\p{L}'‘’\d]
        if (!apos(w[i]) && !is_digit(w[i])) { out.push_back(w.substr(i, 1)); ++i; continue; }
        // ['‘’]+$
        if (apos(w[i])) {
            while (j < n && apos(w[j])) ++j;
            if (j == n) { out.push_back(w.substr(i, j - i)); i = j; continue; }
        }
        ++i;  // no alternative matched
    }
    return out;
}

// ---- lexicon ----------------------------------------------------------------------------------------------------

class Lexicon {
public:
    // tsv lines: word<TAB>tier(g|s)<TAB>tag(- for a plain entry)<TAB>phonemes (empty: no reading for that tag).
    // extra: optional game dictionary, word<TAB>phonemes, checked before gold.
    explicit Lexicon(const std::string& tsv, const std::string& extra = "") {
        std::ifstream f(tsv);
        std::string line;
        while (std::getline(f, line)) {
            size_t a = line.find('\t'), b = line.find('\t', a + 1), c = line.find('\t', b + 1);
            if (c == std::string::npos) continue;
            const U word = to_u32(line.substr(0, a));
            const std::string tag = line.substr(b + 1, c - b - 1);
            Entry& e = (line[a + 1] == 'g' ? gold_ : silver_)[word];
            const U ps = to_u32(line.substr(c + 1));
            if (tag == "-") e.plain = ps;
            else e.tagged[tag] = ps.empty() ? std::nullopt : std::optional<U>(ps);
        }
        if (!extra.empty()) {
            std::ifstream g(extra);
            while (std::getline(g, line)) {
                if (line.empty() || line[0] == '#') continue;
                const size_t a = line.find('\t');
                const size_t b = line.find('\t', a + 1);
                game_[to_u32(line.substr(0, a))] = to_u32(line.substr(a + 1, b == std::string::npos ? b : b - a - 1));
            }
        }
    }

    // Exact case first, then lowercase.
    const U* game(const U& w) const {
        if (auto it = game_.find(w); it != game_.end()) return &it->second;
        if (auto it = game_.find(lower(w)); it != game_.end()) return &it->second;
        return nullptr;
    }
    bool in_gold(const U& w) const { return gold_.count(w) > 0; }

    // Lexicon.__call__
    std::optional<U> operator()(const Token& tk, const Ctx& ctx) const {
        U word = replace_char(replace_char(tk.alias ? *tk.alias : tk.text, U'‘', U'\''), U'’', U'\'');
        if (const U* g = game(word)) return apply_stress(*g, tk.stress);
        std::optional<double> stress;
        if (word != lower(word)) stress = word == upper(word) ? 2.0 : 0.5;
        if (auto ps = get_word(word, tk.tag, stress, ctx)) return apply_stress(append_currency(*ps, tk.currency), tk.stress);
        if (is_number(word, tk.is_head)) {
            auto ps = get_number(word, tk.currency, tk.is_head);
            if (ps) return apply_stress(*ps, tk.stress);
        }
        return std::nullopt;
    }

    std::optional<U> lookup_plain(const U& w) const { return lookup(w, "", std::nullopt, nullptr); }

private:
    struct Entry { std::optional<U> plain; std::unordered_map<std::string, std::optional<U>> tagged; };
    std::unordered_map<U, Entry> gold_, silver_;
    std::unordered_map<U, U> game_;

    const Entry* gold(const U& w) const { auto it = gold_.find(w); return it == gold_.end() ? nullptr : &it->second; }
    const Entry* silver(const U& w) const { auto it = silver_.find(w); return it == silver_.end() ? nullptr : &it->second; }
    const U& gold_plain(const U& w) const { return *gold(w)->plain; }

    static std::string parent_tag(const std::string& tag) {
        if (starts_with(tag, "VB")) return "VERB";
        if (starts_with(tag, "NN")) return "NOUN";
        if (starts_with(tag, "ADV") || starts_with(tag, "RB")) return "ADV";
        if (starts_with(tag, "ADJ") || starts_with(tag, "JJ")) return "ADJ";
        return tag;
    }

    std::optional<U> get_nnp(const U& w) const {
        U ps;
        for (char32_t c : w) {
            if (!is_letter(c)) continue;
            const Entry* e = gold(upper(U(1, c)));
            if (!e || !e->plain) return std::nullopt;
            ps += *e->plain;
        }
        ps = apply_stress(ps, 0);
        const size_t last = ps.rfind(SEC);
        if (last != U::npos) ps[last] = PRI;
        return ps;
    }

    std::optional<U> special(const U& w, const std::string& tag, std::optional<double> stress, const Ctx& ctx) const {
        static const std::unordered_map<U, U> ADD_SYMBOLS = {{U".", U"dot"}, {U"/", U"slash"}};
        static const std::unordered_map<U, U> SYMBOLS = {{U"%", U"percent"}, {U"&", U"and"}, {U"+", U"plus"}, {U"@", U"at"}};
        if (tag == "ADD" && ADD_SYMBOLS.count(w)) return lookup(ADD_SYMBOLS.at(w), "", -0.5, &ctx);
        if (auto it = SYMBOLS.find(w); it != SYMBOLS.end()) return lookup(it->second, "", std::nullopt, &ctx);
        {   // "U.S.": dotted letters, every part shorter than 3
            U stripped = w;
            while (!stripped.empty() && stripped.front() == U'.') stripped.erase(0, 1);
            while (!stripped.empty() && stripped.back() == U'.') stripped.pop_back();
            const U letters = erase_all(w, U'.');
            if (has(stripped, U'.') && all_letters(letters)) {
                size_t longest = 0, cur = 0;
                for (char32_t c : w) { if (c == U'.') cur = 0; else longest = std::max(longest, ++cur); }
                if (longest < 3) return get_nnp(w);
            }
        }
        if (w == U"a" || w == U"A") return tag == "DT" ? U"ɐ" : U"ˈA";
        if (w == U"am" || w == U"Am" || w == U"AM") {
            if (starts_with(tag, "NN")) return get_nnp(w);
            if (!ctx.future_vowel || w != U"am" || (stress && *stress > 0)) return gold_plain(U"am");
            return U"ɐm";
        }
        if (w == U"an" || w == U"An" || w == U"AN") {
            if (w == U"AN" && starts_with(tag, "NN")) return get_nnp(w);
            return U"ɐn";
        }
        if (w == U"I" && tag == "PRP") return U"ˌI";
        if ((w == U"by" || w == U"By" || w == U"BY") && parent_tag(tag) == "ADV") return U"bˈI";
        if (w == U"to" || w == U"To" || (w == U"TO" && (tag == "TO" || tag == "IN"))) {
            if (!ctx.future_vowel) return gold_plain(U"to");
            return *ctx.future_vowel ? U"tʊ" : U"tə";
        }
        if (w == U"in" || w == U"In" || (w == U"IN" && tag != "NNP")) {
            return (!ctx.future_vowel || tag != "IN") ? U"ˈɪn" : U"ɪn";
        }
        if (w == U"the" || w == U"The" || (w == U"THE" && tag == "DT")) {
            return (ctx.future_vowel && *ctx.future_vowel) ? U"ði" : U"ðə";
        }
        if (tag == "IN" && (lower(w) == U"vs" || lower(w) == U"vs.")) return lookup(U"versus", "", std::nullopt, &ctx);
        if (w == U"used" || w == U"Used" || w == U"USED") {
            const Entry* e = gold(U"used");
            if ((tag == "VBD" || tag == "JJ") && ctx.future_to) return *e->tagged.at("VBD");
            return *e->plain;
        }
        return std::nullopt;
    }

    bool is_known(const U& w) const {
        if (gold(w) || silver(w)) return true;
        static const std::unordered_set<U> SYMBOLS = {U"%", U"&", U"+", U"@"};
        if (SYMBOLS.count(w)) return true;
        if (!std::all_of(w.begin(), w.end(), [](char32_t c) { return is_ascii_alpha(c) || c == U'\'' || c == U'-'; })) return false;
        if (!all_letters(w)) return false;
        if (w.size() == 1) return true;
        if (w == upper(w) && gold(lower(w))) return true;
        const U tail = w.substr(1);
        return tail == upper(tail);
    }

    std::optional<U> lookup(U w, std::string tag, std::optional<double> stress, const Ctx* ctx) const {
        bool is_nnp = false;
        if (w == upper(w) && !gold(w)) { w = lower(w); is_nnp = tag == "NNP"; }
        const Entry* e = gold(w);
        if (!e && !is_nnp) e = silver(w);
        std::optional<U> ps;
        if (e) {
            if (!e->tagged.empty()) {
                if (ctx && !ctx->future_vowel && e->tagged.count("None")) tag = "None";
                else if (!e->tagged.count(tag)) tag = parent_tag(tag);
                auto it = e->tagged.find(tag);
                ps = it != e->tagged.end() ? it->second : e->plain;
            } else {
                ps = e->plain;
            }
        }
        if (!ps || (is_nnp && !has(*ps, PRI))) {
            if (auto nnp = get_nnp(w)) return nnp;
        }
        if (!ps) return std::nullopt;
        return apply_stress(*ps, stress);
    }

    static std::optional<U> suffix_s(const std::optional<U>& stem) {
        if (!stem || stem->empty()) return std::nullopt;
        const char32_t last = stem->back();
        if (in(U"ptkfθ", last)) return *stem + U"s";
        if (in(U"szʃʒʧʤ", last)) return *stem + U"ᵻz";
        return *stem + U"z";
    }

    static std::optional<U> suffix_ed(const std::optional<U>& stem) {
        if (!stem || stem->empty()) return std::nullopt;
        const U& s = *stem;
        const char32_t last = s.back();
        if (in(U"pkfθʃsʧ", last)) return s + U"t";
        if (last == U'd') return s + U"ᵻd";
        if (last != U't') return s + U"d";
        if (s.size() < 2) return s + U"ɪd";
        if (in(US_TAUS, s[s.size() - 2])) return s.substr(0, s.size() - 1) + U"ɾᵻd";
        return s + U"ᵻd";
    }

    static std::optional<U> suffix_ing(const std::optional<U>& stem) {
        if (!stem || stem->empty()) return std::nullopt;
        const U& s = *stem;
        if (s.size() > 1 && s.back() == U't' && in(US_TAUS, s[s.size() - 2])) return s.substr(0, s.size() - 1) + U"ɾɪŋ";
        return s + U"ɪŋ";
    }

    std::optional<U> stem_s(const U& w, const std::string& tag, std::optional<double> st, const Ctx* ctx) const {
        if (w.size() < 3 || w.back() != U's') return std::nullopt;
        U stem;
        if (!ends_with(w, U"ss") && is_known(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
        else if ((ends_with(w, U"'s") || (w.size() > 4 && ends_with(w, U"es") && !ends_with(w, U"ies"))) && is_known(w.substr(0, w.size() - 2))) stem = w.substr(0, w.size() - 2);
        else if (w.size() > 4 && ends_with(w, U"ies") && is_known(w.substr(0, w.size() - 3) + U"y")) stem = w.substr(0, w.size() - 3) + U"y";
        else return std::nullopt;
        return suffix_s(lookup(stem, tag, st, ctx));
    }

    std::optional<U> stem_ed(const U& w, const std::string& tag, std::optional<double> st, const Ctx* ctx) const {
        if (w.size() < 4 || w.back() != U'd') return std::nullopt;
        U stem;
        if (!ends_with(w, U"dd") && is_known(w.substr(0, w.size() - 1))) stem = w.substr(0, w.size() - 1);
        else if (w.size() > 4 && ends_with(w, U"ed") && !ends_with(w, U"eed") && is_known(w.substr(0, w.size() - 2))) stem = w.substr(0, w.size() - 2);
        else return std::nullopt;
        return suffix_ed(lookup(stem, tag, st, ctx));
    }

    std::optional<U> stem_ing(const U& w, const std::string& tag, std::optional<double> st, const Ctx* ctx) const {
        if (w.size() < 5 || !ends_with(w, U"ing")) return std::nullopt;
        const U base = w.substr(0, w.size() - 3);
        auto doubled = [&] {
            if (ends_with(w, U"cking")) return true;
            const char32_t a = w[w.size() - 5], b = w[w.size() - 4];
            return a == b && in(U"bcdgklmnprstvxz", a);
        };
        U stem;
        if (w.size() > 5 && is_known(base)) stem = base;
        else if (is_known(base + U"e")) stem = base + U"e";
        else if (w.size() > 5 && doubled() && is_known(w.substr(0, w.size() - 4))) stem = w.substr(0, w.size() - 4);
        else return std::nullopt;
        return suffix_ing(lookup(stem, tag, st, ctx));
    }

    std::optional<U> get_word(U w, const std::string& tag, std::optional<double> stress, const Ctx& ctx) const {
        if (auto ps = special(w, tag, stress, ctx)) return ps;
        const U wl = lower(w);
        const bool alpha = std::all_of(w.begin(), w.end(), [](char32_t c) { return is_letter(c) || c == U'\''; }) &&
                           std::any_of(w.begin(), w.end(), is_letter);
        if (w.size() > 1 && alpha && w != wl && (tag != "NNP" || w.size() > 7) && !gold(w) && !silver(w) &&
            (w == upper(w) || w.substr(1) == lower(w.substr(1))) &&
            (gold(wl) || silver(wl) || stem_s(wl, tag, stress, &ctx) || stem_ed(wl, tag, stress, &ctx) ||
             stem_ing(wl, tag, stress, &ctx))) {
            w = wl;
        }
        if (is_known(w)) return lookup(w, tag, stress, &ctx);
        if (ends_with(w, U"s'") && is_known(w.substr(0, w.size() - 2) + U"'s")) return lookup(w.substr(0, w.size() - 2) + U"'s", tag, stress, &ctx);
        if (ends_with(w, U"'") && is_known(w.substr(0, w.size() - 1))) return lookup(w.substr(0, w.size() - 1), tag, stress, &ctx);
        if (auto ps = stem_s(w, tag, stress, &ctx)) return ps;
        if (auto ps = stem_ed(w, tag, stress, &ctx)) return ps;
        if (auto ps = stem_ing(w, tag, stress ? stress : std::optional<double>(0.5), &ctx)) return ps;
        return std::nullopt;
    }

    // ---- numbers ----

    static std::optional<std::pair<U, U>> currency_units(std::optional<char32_t> c) {
        if (!c) return std::nullopt;
        if (*c == U'$') return std::make_pair(U"dollar", U"cent");
        if (*c == U'£') return std::make_pair(U"pound", U"pence");
        if (*c == U'€') return std::make_pair(U"euro", U"cent");
        return std::nullopt;
    }

    U append_currency(const U& ps, std::optional<char32_t> currency) const {
        auto units = currency_units(currency);
        if (!units) return ps;
        auto c = stem_s(units->first + U"s", "", std::nullopt, nullptr);
        return c ? ps + U" " + *c : ps;
    }

    static bool is_number(const U& w, bool is_head) {
        if (std::none_of(w.begin(), w.end(), is_digit)) return false;
        U word = w;
        for (const U s : {U"ing", U"'d", U"ed", U"'s", U"st", U"nd", U"rd", U"th", U"s"}) {
            if (ends_with(word, s)) { word = word.substr(0, word.size() - s.size()); break; }
        }
        for (size_t i = 0; i < word.size(); ++i) {
            const char32_t c = word[i];
            if (!(is_digit(c) || c == U',' || c == U'.' || (is_head && i == 0 && c == U'-'))) return false;
        }
        return true;
    }

    static bool is_currency_amount(const U& w) {
        const size_t dots = std::count(w.begin(), w.end(), U'.');
        if (dots == 0) return true;
        if (dots > 1) return false;
        return w.size() - w.find(U'.') - 1 < 3;
    }

    std::optional<U> get_number(U word, std::optional<char32_t> currency, bool is_head) const {
        U suffix;
        {
            size_t k = word.size();
            while (k > 0 && ((word[k - 1] >= 'a' && word[k - 1] <= 'z') || word[k - 1] == U'\'')) --k;
            suffix = word.substr(k);
            word = word.substr(0, k);
        }
        std::vector<U> result;
        if (!word.empty() && word[0] == U'-') { result.push_back(*lookup(U"minus", "", std::nullopt, nullptr)); word = word.substr(1); }
        // extend_num: split num2words text on non-letters, drop "and", look each word up.
        auto extend_words = [&](const std::string& text) {
            const U t = to_u32(text);
            std::vector<U> splits; U cur;
            for (char32_t c : t) { if (c >= 'a' && c <= 'z') cur += c; else { splits.push_back(cur); cur.clear(); } }
            splits.push_back(cur);
            // re.split keeps empty strings only at the edges; adjacent separators are one split.
            std::vector<U> merged;
            for (size_t i = 0; i < splits.size(); ++i) {
                if (splits[i].empty() && i != 0 && i + 1 != splits.size()) continue;
                merged.push_back(splits[i]);
            }
            for (auto& w : merged) {
                if (w == U"and") continue;
                auto ps = lookup(w, "", w == U"point" ? std::optional<double>(-2) : std::nullopt, nullptr);
                result.push_back(ps ? *ps : U"");
            }
        };
        auto extend_num = [&](const U& digits) { extend_words(num::cardinal(std::stoull(to_u8(digits)))); };
        const U plain = erase_all(word, U',');
        const bool ordinal = suffix == U"st" || suffix == U"nd" || suffix == U"rd" || suffix == U"th";
        const bool known_currency = currency_units(currency).has_value();
        if (all_digits(word) && ordinal) {
            extend_words(num::ordinal(std::stoull(to_u8(word))));
        } else if (result.empty() && word.size() == 4 && !known_currency && all_digits(word)) {
            extend_words(num::year(std::stoull(to_u8(word))));
        } else if (!is_head && !has(word, U'.')) {
            const U& n = plain;
            if (n.empty()) return std::nullopt;
            if (n[0] == U'0' || n.size() > 3) {
                for (char32_t c : n) extend_num(U(1, c));
            } else if (n.size() == 3 && !ends_with(n, U"00")) {
                extend_num(n.substr(0, 1));
                if (n[1] == U'0') { result.push_back(*lookup(U"O", "", -2, nullptr)); extend_num(n.substr(2)); }
                else extend_num(n.substr(1));
            } else {
                extend_num(n);
            }
        } else if (std::count(word.begin(), word.end(), U'.') > 1 || !is_head) {
            size_t start = 0;
            while (start <= plain.size()) {
                size_t dot = plain.find(U'.', start);
                if (dot == U::npos) dot = plain.size();
                const U part = plain.substr(start, dot - start);
                if (!part.empty()) {
                    const bool nonzero_tail = std::any_of(part.begin() + 1, part.end(), [](char32_t c) { return c != U'0'; });
                    if (part[0] == U'0' || (part.size() != 2 && nonzero_tail)) for (char32_t c : part) extend_num(U(1, c));
                    else extend_num(part);
                }
                start = dot + 1;
            }
        } else if (known_currency && is_currency_amount(word)) {
            const auto units = *currency_units(currency);
            std::vector<std::pair<uint64_t, U>> pairs;
            const size_t dot = plain.find(U'.');
            const U whole = plain.substr(0, dot);
            pairs.emplace_back(whole.empty() ? 0 : std::stoull(to_u8(whole)), units.first);
            if (dot != U::npos) {
                const U cents = plain.substr(dot + 1);
                pairs.emplace_back(cents.empty() ? 0 : std::stoull(to_u8(cents)), units.second);
            }
            if (pairs.size() > 1) {
                if (pairs[1].first == 0) pairs.resize(1);
                else if (pairs[0].first == 0) pairs.erase(pairs.begin());
            }
            for (size_t i = 0; i < pairs.size(); ++i) {
                if (i > 0) result.push_back(*lookup(U"and", "", std::nullopt, nullptr));
                extend_words(num::cardinal(pairs[i].first));
                const U& unit = pairs[i].second;
                if (pairs[i].first != 1 && unit != U"pence") {
                    auto s = stem_s(unit + U"s", "", std::nullopt, nullptr);
                    result.push_back(s ? *s : U"");
                } else {
                    result.push_back(*lookup(unit, "", std::nullopt, nullptr));
                }
            }
        } else {
            if (all_digits(word)) extend_words(num::cardinal(std::stoull(to_u8(word))));
            else if (!has(word, U'.')) extend_words(ordinal ? num::ordinal(std::stoull(to_u8(plain))) : num::cardinal(std::stoull(to_u8(plain))));
            else if (plain[0] == U'.') {
                std::string text = "point";
                for (size_t i = 1; i < plain.size(); ++i) text += " " + num::cardinal(plain[i] - U'0');
                extend_words(text);
            } else {
                extend_words(num::decimal(to_u8(plain)));
            }
        }
        if (result.empty()) return std::nullopt;
        U joined;
        for (size_t i = 0; i < result.size(); ++i) { if (i) joined += U' '; joined += result[i]; }
        if (suffix == U"s" || suffix == U"'s") return suffix_s(joined);
        if (suffix == U"ed" || suffix == U"'d") return suffix_ed(joined);
        if (suffix == U"ing") return suffix_ing(joined);
        return joined;
    }
};

// ---- pipeline (misaki G2P.__call__ after spaCy) ----------------------------------------------------------------

class G2P {
public:
    // fallback: phonemes for a word the lexicon cannot resolve (misaki: espeak); nullopt keeps it unknown.
    // An empty fallback is misaki's fallback=None: unresolved parts of a word stay empty, the rest is kept.
    using Fallback = std::function<std::optional<U>(const Token&)>;
    G2P(const Lexicon& lex, Fallback fallback) : lex_(lex), fallback_(std::move(fallback)) {}

    std::string operator()(std::vector<Token> tokens) const {
        std::vector<Item> items = retokenize(tokens);
        Ctx ctx;
        for (size_t k = items.size(); k-- > 0;) {
            Item& it = items[k];
            if (!it.is_list) {
                Token& w = it.toks[0];
                if (!w.phonemes) w.phonemes = lex_(w, ctx);
                if (!w.phonemes && fallback_) w.phonemes = fallback_(w);
                ctx = token_context(ctx, w.phonemes, w);
                continue;
            }
            auto& w = it.toks;
            size_t left = 0, right = w.size();
            bool should_fallback = false;
            while (left < right) {
                std::optional<Token> tk;
                bool blocked = false;
                for (size_t i = left; i < right; ++i) if (w[i].alias || w[i].phonemes) blocked = true;
                if (!blocked) tk = merge(w, left, right, false);
                std::optional<U> ps = tk ? lex_(*tk, ctx) : std::nullopt;
                if (ps) {
                    w[left].phonemes = ps;
                    for (size_t i = left + 1; i < right; ++i) w[i].phonemes = U"";
                    ctx = token_context(ctx, ps, *tk);
                    right = left;
                    left = 0;
                } else if (left + 1 < right) {
                    ++left;
                } else {
                    --right;
                    Token& t = w[right];
                    if (!t.phonemes) {
                        if (std::all_of(t.text.begin(), t.text.end(), [](char32_t c) { return in(SUBTOKEN_JUNKS, c); })) t.phonemes = U"";
                        else if (fallback_) { should_fallback = true; break; }
                    }
                    left = 0;
                }
            }
            if (should_fallback) {
                const Token merged = merge(w, 0, w.size(), false);
                w[0].phonemes = fallback_(merged);
                for (size_t i = 1; i < w.size(); ++i) w[i].phonemes = U"";
            } else {
                resolve_tokens(w);
            }
        }
        U result;
        for (auto& it : items) {
            const Token t = it.is_list ? merge(it.toks, 0, it.toks.size(), true) : it.toks[0];
            if (t.phonemes) result += *t.phonemes;
            result += t.whitespace;
        }
        return to_u8(replace_char(replace_char(result, U'ɾ', U'T'), U'ʔ', U't'));
    }

private:
    struct Item { bool is_list; std::vector<Token> toks; };
    const Lexicon& lex_;
    Fallback fallback_;

    // merge_tokens; with_unk: also join phonemes ('' for missing ones), as the final merge does.
    static Token merge(const std::vector<Token>& w, size_t a, size_t b, bool with_unk) {
        Token t;
        for (size_t i = a; i < b; ++i) t.text += w[i].text + (i + 1 < b ? w[i].whitespace : U"");
        int best = -1;
        for (size_t i = a; i < b; ++i) {
            int score = 0;
            for (char32_t c : w[i].text) score += (c == lower(U(1, c))[0]) ? 1 : 2;
            if (score > best) { best = score; t.tag = w[i].tag; }
        }
        t.whitespace = w[b - 1].whitespace;
        t.is_head = w[a].is_head;
        t.prespace = w[a].prespace;
        std::optional<double> stress; bool stress_conflict = false;
        for (size_t i = a; i < b; ++i) {
            if (!w[i].stress) continue;
            if (stress && *stress != *w[i].stress) stress_conflict = true;
            stress = w[i].stress;
        }
        if (!stress_conflict) t.stress = stress;
        for (size_t i = a; i < b; ++i) if (w[i].currency && (!t.currency || *w[i].currency > *t.currency)) t.currency = w[i].currency;
        if (with_unk) {
            U ps;
            for (size_t i = a; i < b; ++i) {
                if (w[i].prespace && !ps.empty() && ps.back() != U' ' && w[i].phonemes && !w[i].phonemes->empty()) ps += U' ';
                if (w[i].phonemes) ps += *w[i].phonemes;
            }
            t.phonemes = ps;
        }
        return t;
    }

    static Ctx token_context(const Ctx& ctx, const std::optional<U>& ps, const Token& tk) {
        Ctx out;
        out.future_vowel = ctx.future_vowel;
        if (ps && !ps->empty()) {
            for (char32_t c : *ps) {
                if (in(NON_QUOTE_PUNCTS, c)) { out.future_vowel.reset(); break; }
                if (in(VOWELS, c)) { out.future_vowel = true; break; }
                if (in(CONSONANTS, c)) { out.future_vowel = false; break; }
            }
        }
        out.future_to = tk.text == U"to" || tk.text == U"To" || (tk.text == U"TO" && (tk.tag == "TO" || tk.tag == "IN"));
        return out;
    }

    static void resolve_tokens(std::vector<Token>& tokens) {
        U text;
        for (size_t i = 0; i < tokens.size(); ++i) text += tokens[i].text + (i + 1 < tokens.size() ? tokens[i].whitespace : U"");
        std::unordered_set<int> kinds;
        for (char32_t c : text) {
            if (in(SUBTOKEN_JUNKS, c)) continue;
            kinds.insert(is_letter(c) ? 0 : is_digit(c) ? 1 : 2);
        }
        const bool prespace = has(text, U' ') || has(text, U'/') || kinds.size() > 1;
        for (size_t i = 0; i < tokens.size(); ++i) {
            Token& tk = tokens[i];
            if (!tk.phonemes) {
                if (i + 1 == tokens.size() && tk.text.size() == 1 && in(NON_QUOTE_PUNCTS, tk.text[0])) tk.phonemes = tk.text;
                else if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return in(SUBTOKEN_JUNKS, c); })) tk.phonemes = U"";
            } else if (i > 0) {
                tk.prespace = prespace;
            }
        }
        if (prespace) return;
        struct I { bool pri; int w; size_t i; };
        std::vector<I> idx;
        for (size_t i = 0; i < tokens.size(); ++i)
            if (tokens[i].phonemes && !tokens[i].phonemes->empty())
                idx.push_back({has(*tokens[i].phonemes, PRI), stress_weight(*tokens[i].phonemes), i});
        if (idx.size() == 2 && tokens[idx[0].i].text.size() == 1) {
            tokens[idx[1].i].phonemes = apply_stress(*tokens[idx[1].i].phonemes, -0.5);
            return;
        }
        int n_pri = 0; for (auto& x : idx) n_pri += x.pri;
        if (idx.size() < 2 || n_pri <= static_cast<int>(idx.size() + 1) / 2) return;
        std::sort(idx.begin(), idx.end(), [](const I& a, const I& b) { return std::tie(a.pri, a.w, a.i) < std::tie(b.pri, b.w, b.i); });
        for (size_t k = 0; k < idx.size() / 2; ++k) tokens[idx[k].i].phonemes = apply_stress(*tokens[idx[k].i].phonemes, -0.5);
    }

    static std::vector<Item> retokenize(const std::vector<Token>& tokens) {
        static const std::unordered_set<std::string> PUNCT_TAGS = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"};
        std::vector<Item> words;
        std::optional<char32_t> currency;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const Token& token = tokens[i];
            std::vector<Token> tks;
            if (!token.alias && !token.phonemes) {
                for (const U& t : subtokenize(token.text)) {
                    Token s;
                    s.text = t;
                    s.tag = token.tag;
                    s.stress = token.stress;
                    tks.push_back(s);
                }
                if (tks.empty()) { Token s = token; s.whitespace = U""; tks.push_back(s); }
            } else {
                tks.push_back(token);
            }
            tks.back().whitespace = token.whitespace;
            for (size_t j = 0; j < tks.size(); ++j) {
                Token& tk = tks[j];
                const bool letters_only = std::all_of(tk.text.begin(), tk.text.end(), is_ascii_alpha);
                if (tk.alias || tk.phonemes) {
                } else if (tk.tag == "$" && tk.text.size() == 1 && in(U"$£€", tk.text[0])) {
                    currency = tk.text[0];
                    tk.phonemes = U"";
                } else if (tk.tag == ":" && (tk.text == U"-" || tk.text == U"–")) {
                    tk.phonemes = U"—";
                } else if (PUNCT_TAGS.count(tk.tag) && !letters_only) {
                    if (tk.tag == "-LRB-") tk.phonemes = U"(";
                    else if (tk.tag == "-RRB-") tk.phonemes = U")";
                    else if (tk.tag == "``") tk.phonemes = U"“";
                    else if (tk.tag == "\"\"" || tk.tag == "''") tk.phonemes = U"”";
                    else { U p; for (char32_t c : tk.text) if (in(PUNCTS, c)) p += c; tk.phonemes = p; }
                } else if (currency) {
                    if (tk.tag != "CD") currency.reset();
                    else if (j + 1 == tks.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) tk.currency = currency;
                } else if (j > 0 && j + 1 < tks.size() && tk.text == U"2" && !tks[j - 1].text.empty() && !tks[j + 1].text.empty() &&
                           is_letter(tks[j - 1].text.back()) && is_letter(tks[j + 1].text[0])) {
                    tk.alias = U"to";
                }
                if (tk.alias || tk.phonemes) {
                    words.push_back({false, {tk}});
                } else if (!words.empty() && words.back().is_list && words.back().toks.back().whitespace.empty()) {
                    tk.is_head = false;
                    words.back().toks.push_back(tk);
                } else if (!tk.whitespace.empty()) {
                    words.push_back({false, {tk}});
                } else {
                    words.push_back({true, {tk}});
                }
            }
        }
        for (auto& w : words) if (w.is_list && w.toks.size() == 1) w.is_list = false;
        return words;
    }
};

}  // namespace mg2p
