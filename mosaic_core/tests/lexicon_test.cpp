#include "mosaic/lexicon.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const std::string path = "mosaic_lexicon_test.tsv";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "MCR_WORDNET_2025_PLUS_V1\n";
        out << "apple\tn\ta fruit produced by an apple tree\torchard apple;malus pumila\tfruit;edible fruit\n";
        out << "apple\tn\tthe tree bearing apples\tapple tree\tfruit tree\n";
        out << "dog\tn\ta domesticated carnivorous mammal\tcanis familiaris;domestic dog\tcanine;domestic animal\n";
        assert(out.good());
    }

    mosaic::Lexicon lexicon;
    std::string error;
    assert(lexicon.load(path, &error));
    assert(lexicon.loaded());
    assert(lexicon.size() == 3);

    const auto apple = lexicon.lookup("Apple");
    assert(apple.size() == 2);
    assert(apple.front()->definition.find("fruit") != std::string::npos);
    assert(apple.front()->synonyms.find("orchard apple") != std::string::npos);

    const auto dog = lexicon.lookup("dog");
    assert(dog.size() == 1);
    assert(dog.front()->hypernyms.find("canine") != std::string::npos);
    assert(lexicon.lookup("unknownword").empty());

    std::remove(path.c_str());
    std::cout << "MOSAIC lexical tests passed\n";
    return 0;
}
