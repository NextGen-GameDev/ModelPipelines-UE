#pragma once
// Averaged-perceptron POS tagger (after Matthew Honnibal's "A good part-of-speech tagger in about 200 lines of
// Python"), inference only. Weights come from train_pos_tagger.py, which distils spaCy en_core_web_sm's Penn
// Treebank tags on public-domain prose; the features below must match FEATURES in train_pos_tagger.py exactly.
// File format (UTF-8, tab-separated): "C" then every class; "D word tag" for unambiguous words;
// "W feature class=weight ..." for each feature.
#include "misaki_g2p.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mg2p {

class PosTagger {
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            std::vector<std::string> cols;
            size_t start = 0;
            for (size_t i = 0; i <= line.size(); ++i)
                if (i == line.size() || line[i] == '\t') { cols.push_back(line.substr(start, i - start)); start = i + 1; }
            if (cols[0] == "C") {
                classes_.assign(cols.begin() + 1, cols.end());
                for (size_t i = 0; i < classes_.size(); ++i) class_ids_[classes_[i]] = static_cast<int>(i);
            } else if (cols[0] == "D") {
                tagdict_[cols[1]] = cols[2];
            } else if (cols[0] == "W") {
                auto& ws = weights_[cols[1]];
                for (size_t i = 2; i < cols.size(); ++i) {
                    const size_t eq = cols[i].rfind('=');
                    ws.emplace_back(class_ids_.at(cols[i].substr(0, eq)), std::stod(cols[i].substr(eq + 1)));
                }
            }
        }
        return !classes_.empty();
    }

    // Tags every token's text; writes Token::tag.
    void tag(std::vector<Token>& tokens) const {
        std::vector<std::string> words, context = {"-START-", "-START2-"};
        for (const Token& t : tokens) { words.push_back(to_u8(t.text)); context.push_back(normalize(t.text)); }
        context.push_back("-END-");
        context.push_back("-END2-");
        std::string prev = "-START-", prev2 = "-START2-";
        std::vector<double> scores(classes_.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string guess;
            if (auto it = tagdict_.find(words[i]); it != tagdict_.end()) {
                guess = it->second;
            } else {
                std::fill(scores.begin(), scores.end(), 0.0);
                for (const std::string& feat : features(i + 2, tokens[i].text, context, prev, prev2)) {
                    auto it2 = weights_.find(feat);
                    if (it2 == weights_.end()) continue;
                    for (auto& [cls, w] : it2->second) scores[cls] += w;
                }
                // max by (score, label), as Python's max(classes, key=lambda c: (scores[c], c))
                size_t best = 0;
                for (size_t c = 1; c < classes_.size(); ++c)
                    if (scores[c] > scores[best] || (scores[c] == scores[best] && classes_[c] > classes_[best])) best = c;
                guess = classes_[best];
            }
            tokens[i].tag = guess;
            prev2 = prev;
            prev = guess;
        }
    }

    static std::string normalize(const U& w) {
        if (w.find(U'-') != U::npos && w[0] != U'-') return "!HYPHEN";
        if (w.size() == 4 && all_digits(w)) return "!YEAR";
        if (!w.empty() && is_digit(w[0])) return "!DIGITS";
        return to_u8(lower(w));
    }

    static std::string shape(const U& w) {
        if (w.empty()) return "p";
        if (w.size() > 1 && std::all_of(w.begin(), w.end(), [](char32_t c) { return is_upper(c); })) return "X";
        if (is_upper(w[0])) return "Xx";
        if (is_lower(w[0]) || is_letter(w[0])) return "x";
        if (is_digit(w[0])) return "d";
        return "p";
    }

    // Must match train_pos_tagger.py FEATURES. i: index into context (offset by the two start symbols).
    static std::vector<std::string> features(size_t i, const U& word, const std::vector<std::string>& context,
                                             const std::string& prev, const std::string& prev2) {
        auto suffix3 = [](const std::string& u8) {
            const U u = to_u32(u8);
            return to_u8(u.size() > 3 ? u.substr(u.size() - 3) : u);
        };
        const U suffix = word.size() > 3 ? word.substr(word.size() - 3) : word;
        return {
            "bias",
            "i suffix " + to_u8(suffix),
            "i pref1 " + to_u8(word.substr(0, 1)),
            "i-1 tag " + prev,
            "i-2 tag " + prev2,
            "i tag+i-2 tag " + prev + " " + prev2,
            "i word " + context[i],
            "i-1 tag+i word " + prev + " " + context[i],
            "i-1 word " + context[i - 1],
            "i-1 suffix " + suffix3(context[i - 1]),
            "i-2 word " + context[i - 2],
            "i+1 word " + context[i + 1],
            "i+1 suffix " + suffix3(context[i + 1]),
            "i+2 word " + context[i + 2],
            "i shape " + shape(word),
        };
    }

private:
    std::vector<std::string> classes_;
    std::unordered_map<std::string, int> class_ids_;
    std::unordered_map<std::string, std::string> tagdict_;
    std::unordered_map<std::string, std::vector<std::pair<int, double>>> weights_;
};

}  // namespace mg2p
