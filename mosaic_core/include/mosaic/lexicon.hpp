#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mosaic {

struct LexicalEntry {
    std::string lemma;
    std::string pos;
    std::string definition;
    std::string synonyms;
    std::string hypernyms;
};

class Lexicon {
public:
    bool load(const std::string& path, std::string* error = nullptr) {
        std::ifstream input(path, std::ios::binary);
        if (!input) return fail("cannot open lexical database: " + path, error);

        std::string line;
        if (!std::getline(input, line) || line != "MCR_WORDNET_2025_PLUS_V1")
            return fail("unsupported or corrupt lexical database header", error);

        std::vector<LexicalEntry> next;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::vector<std::string> fields;
            std::size_t begin = 0;
            while (fields.size() < 4) {
                const auto at = line.find('\t', begin);
                if (at == std::string::npos) break;
                fields.push_back(line.substr(begin, at - begin));
                begin = at + 1;
            }
            fields.push_back(line.substr(begin));
            if (fields.size() != 5 || fields[0].empty() || fields[2].empty())
                return fail("invalid lexical database record", error);
            next.push_back({std::move(fields[0]), std::move(fields[1]),
                            std::move(fields[2]), std::move(fields[3]),
                            std::move(fields[4])});
        }
        if (!input.eof()) return fail("error while reading lexical database", error);
        std::sort(next.begin(), next.end(), [](const LexicalEntry& a, const LexicalEntry& b) {
            if (a.lemma != b.lemma) return a.lemma < b.lemma;
            if (a.pos != b.pos) return a.pos < b.pos;
            return a.definition < b.definition;
        });
        entries_ = std::move(next);
        loaded_ = !entries_.empty();
        if (!loaded_) return fail("lexical database contains no entries", error);
        return true;
    }

    bool loaded() const noexcept { return loaded_; }
    std::size_t size() const noexcept { return entries_.size(); }

    std::vector<const LexicalEntry*> lookup(const std::string& input,
                                             std::size_t limit = 3) const {
        std::vector<const LexicalEntry*> result;
        if (!loaded_ || limit == 0) return result;
        const std::string lemma = normalize(input);
        const auto first = std::lower_bound(entries_.begin(), entries_.end(), lemma,
            [](const LexicalEntry& entry, const std::string& key) {
                return entry.lemma < key;
            });
        for (auto it = first; it != entries_.end() && it->lemma == lemma &&
             result.size() < limit; ++it) {
            result.push_back(&*it);
        }
        return result;
    }

    static std::string normalize(std::string value) {
        for (char& raw : value) {
            const auto c = static_cast<unsigned char>(raw);
            if (c >= 'A' && c <= 'Z') raw = static_cast<char>(c - 'A' + 'a');
            else if (raw == '_') raw = ' ';
        }
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
            value.erase(value.begin());
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
            value.pop_back();
        return value;
    }

private:
    static bool fail(const std::string& message, std::string* error) {
        if (error) *error = message;
        return false;
    }

    bool loaded_ = false;
    std::vector<LexicalEntry> entries_;
};

}  // namespace mosaic
