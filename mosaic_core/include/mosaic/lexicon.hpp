#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
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

// The full dictionary remains one contiguous byte buffer. The index stores
// only 32-bit line offsets, avoiding hundreds of thousands of heap strings.
class Lexicon {
public:
    bool load(const std::string& path, std::string* error = nullptr) {
        std::ifstream input(path, std::ios::binary);
        if (!input) return fail("cannot open lexical database: " + path, error);

        std::string next_data((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
        if (!input.eof() && input.fail())
            return fail("error while reading lexical database", error);

        const std::string header = "MCR_WORDNET_2025_PLUS_V1\n";
        if (next_data.compare(0, header.size(), header) != 0)
            return fail("unsupported or corrupt lexical database header", error);

        std::vector<std::uint32_t> next_offsets;
        std::size_t position = header.size();
        std::string previous;
        while (position < next_data.size()) {
            if (position > 0xFFFFFFFFULL)
                return fail("lexical database exceeds the 4 GiB offset limit", error);
            const auto end = next_data.find('\n', position);
            const std::size_t line_end = end == std::string::npos
                ? next_data.size() : end;
            if (line_end > position) {
                const auto tab = next_data.find('\t', position);
                if (tab == std::string::npos || tab >= line_end)
                    return fail("invalid lexical database record", error);
                const std::string key = next_data.substr(position, tab - position);
                if (key.empty() || (!previous.empty() && key < previous))
                    return fail("lexical database is not sorted by lemma", error);
                previous = key;
                next_offsets.push_back(static_cast<std::uint32_t>(position));
            }
            if (end == std::string::npos) break;
            position = end + 1;
        }
        if (next_offsets.empty())
            return fail("lexical database contains no entries", error);

        data_ = std::move(next_data);
        offsets_ = std::move(next_offsets);
        loaded_ = true;
        return true;
    }

    bool loaded() const noexcept { return loaded_; }
    std::size_t size() const noexcept { return offsets_.size(); }
    std::size_t data_bytes() const noexcept { return data_.size(); }

    std::vector<LexicalEntry> lookup(const std::string& input,
                                     std::size_t limit = 3) const {
        std::vector<LexicalEntry> result;
        if (!loaded_ || limit == 0) return result;
        const std::string lemma = normalize(input);
        const auto first = std::lower_bound(offsets_.begin(), offsets_.end(), lemma,
            [this](std::uint32_t offset, const std::string& key) {
                const auto end = data_.find('\t', offset);
                const std::string_view candidate(
                    data_.data() + offset, end == std::string::npos ? 0 : end - offset);
                return candidate < std::string_view(key);
            });
        for (auto it = first; it != offsets_.end() && result.size() < limit; ++it) {
            const auto entry = parse_line(*it);
            if (entry.lemma != lemma) break;
            result.push_back(entry);
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

    LexicalEntry parse_line(std::uint32_t offset) const {
        const auto newline = data_.find('\n', offset);
        const auto end = newline == std::string::npos ? data_.size() : newline;
        std::array<std::string, 5> fields;
        std::size_t begin = offset;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto tab = data_.find('\t', begin);
            if (tab == std::string::npos || tab >= end) return {};
            fields[i] = data_.substr(begin, tab - begin);
            begin = tab + 1;
        }
        fields[4] = data_.substr(begin, end - begin);
        return {std::move(fields[0]), std::move(fields[1]),
                std::move(fields[2]), std::move(fields[3]),
                std::move(fields[4])};
    }

    bool loaded_ = false;
    std::string data_;
    std::vector<std::uint32_t> offsets_;
};

}  // namespace mosaic
