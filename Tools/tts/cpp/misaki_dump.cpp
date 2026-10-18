// Prints the misaki port's phonemes, one line per stdin line.
// usage: misaki_dump <misaki_us.tsv> <g2p_en.weights|-> [--tagger pos_tagger.tsv] [--game game_words.tsv]
//                    [--oracle | --tokens | --tags]
//   default: stdin lines are text, tokenized by tokenizer.h and tagged by --tagger (required).
//   --oracle: stdin lines are pre-tagged tokens (spaCy's, from compare.py), so the pipeline can be checked
//             without our tokenizer and tagger. Token encoding: text \x1f tag \x1f whitespace(0|1), joined by \x1e.
//   --tokens: prints our tokenization of each text line in the token encoding (tag empty), for training.
//   --tags:   prints our tokens with our tagger's tags, for the training-script parity check.
//   g2p_en.weights '-': no fallback, unresolved words come out empty (misaki with fallback=None, unk='').
#include "fallback.h"
#include "misaki_g2p.h"
#include "pos_tagger.h"
#include "tokenizer.h"
#include "english/EnG2P.h"
#include <cstring>
#include <iostream>

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) { out.push_back(s.substr(start, i - start)); start = i + 1; }
    }
    return out;
}

std::vector<mg2p::Token> parse_tokens(const std::string& line) {
    std::vector<mg2p::Token> tokens;
    if (line.empty()) return tokens;
    for (const std::string& t : split(line, '\x1e')) {
        const auto f = split(t, '\x1f');
        mg2p::Token tk;
        tk.text = mg2p::to_u32(f.at(0));
        tk.tag = f.at(1);
        tk.whitespace = f.at(2) == "1" ? U" " : U"";
        tokens.push_back(tk);
    }
    return tokens;
}

std::string encode_tokens(const std::vector<mg2p::Token>& tokens) {
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out += '\x1e';
        out += mg2p::to_u8(tokens[i].text) + '\x1f' + tokens[i].tag + '\x1f' + (tokens[i].whitespace.empty() ? "0" : "1");
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: misaki_dump <lexicon.tsv> <g2p_en.weights|-> [--tagger FILE] [--game FILE] "
                     "[--oracle | --tokens | --tags]\n";
        return 2;
    }
    std::string mode = "text", game, tagger_path;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--oracle")) mode = "oracle";
        else if (!std::strcmp(argv[i], "--tokens")) mode = "tokens";
        else if (!std::strcmp(argv[i], "--tags")) mode = "tags";
        else if (!std::strcmp(argv[i], "--game") && i + 1 < argc) game = argv[++i];
        else if (!std::strcmp(argv[i], "--tagger") && i + 1 < argc) tagger_path = argv[++i];
    }
    std::ios::sync_with_stdio(false);
    std::string line;
    if (mode == "tokens") {
        while (std::getline(std::cin, line)) std::cout << encode_tokens(mg2p::tokenize(mg2p::to_u32(line))) << '\n';
        return 0;
    }
    mg2p::PosTagger tagger;
    if (mode != "oracle" && !tagger.load(tagger_path)) {
        std::cerr << "cannot load tagger '" << tagger_path << "' (run train_pos_tagger.py)\n";
        return 2;
    }
    if (mode == "tags") {
        while (std::getline(std::cin, line)) {
            auto tokens = parse_tokens(line);
            tagger.tag(tokens);
            std::cout << encode_tokens(tokens) << '\n';
        }
        return 0;
    }
    const mg2p::Lexicon lex(argv[1], game);
    const std::string neural_path = std::string(argv[2]) == "-" ? "" : argv[2];
    const EnG2P neural("", "", neural_path);  // no CMU dict: neural guesses only
    mg2p::G2P::Fallback fallback;
    if (!neural_path.empty()) fallback = mg2p::make_fallback(lex, [&](const mg2p::U& word) -> std::optional<mg2p::U> {
        const std::string u8 = mg2p::to_u8(word), ps = neural.convert(u8);
        if (ps == u8) return std::nullopt;
        return mg2p::to_u32(ps);
    });
    const mg2p::G2P g2p(lex, fallback);
    while (std::getline(std::cin, line)) {
        std::vector<mg2p::Token> tokens;
        if (mode == "oracle") {
            tokens = parse_tokens(line);
        } else {
            tokens = mg2p::tokenize(mg2p::to_u32(line));
            tagger.tag(tokens);
        }
        std::cout << g2p(tokens) << '\n';
    }
}
