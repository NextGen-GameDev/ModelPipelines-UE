// Prints larroy's English phonemes, one line per stdin line.
// usage: en_dump <cmudict> <g2p_en.weights|-> [user_dict]
#include "english/EnglishPhonemizer.h"
#include <iostream>
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: en_dump <dict> <neural|-> [user_dict]\n"; return 2; }
    const std::string neural = std::string(argv[2]) == "-" ? "" : argv[2];
    auto g2p = std::make_shared<const EnG2P>(argv[1], argc > 3 ? argv[3] : "", neural);
    EnglishPhonemizer ph(g2p);
    std::string line;
    while (std::getline(std::cin, line)) std::cout << ph.phonemize(line, NumberLanguage::Auto) << '\n';
}
