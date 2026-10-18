// Prints the misaki-lexicon port's phonemes, one line per stdin line.
// usage: misaki_dump <misaki_us.tsv> <g2p_en.weights>
#include "misaki_lex.h"
#include "english/EnG2P.h"
#include "NumberNormalizer.h"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: misaki_dump <lexicon.tsv> <g2p_en.weights>\n"; return 2; }
    const EnG2P neural("", "", argv[2]);  // no CMU dict: neural guesses only
    const mlex::Lexicon lex(argv[1], [&](const mlex::U& w) -> std::optional<mlex::U> {
        const std::string u8 = mlex::to_u8(w), ps = neural.convert(u8);
        if (ps == u8) return std::nullopt;
        return mlex::to_u32(ps);
    });
    std::string line;
    while (std::getline(std::cin, line))
        std::cout << mlex::phonemize(lex, normalize_numbers(line, NumberLanguage::English)) << '\n';
}
