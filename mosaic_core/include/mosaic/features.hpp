#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mosaic {

struct SparseFeature {
    std::uint32_t id = 0;
    float value = 1.0F;
};

inline std::uint32_t stable_feature_hash(const std::string& feature,
                                         std::uint32_t dimension) {
    // Canonical 64-bit FNV-1a, deterministic across Python and C++.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : feature) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    if (dimension == 0) return 0;
    return static_cast<std::uint32_t>(hash % dimension);
}

inline bool is_mosaic_word_byte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80;
}

inline std::string mosaic_lower_ascii(std::string value) {
    for (char& raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c >= 'A' && c <= 'Z') raw = static_cast<char>(c - 'A' + 'a');
    }
    return value;
}

inline std::vector<std::string> mosaic_tokenize(const std::string& input) {
    const std::string text = mosaic_lower_ascii(input);
    std::vector<std::string> tokens;
    std::string current;
    for (const unsigned char c : text) {
        if (is_mosaic_word_byte(c)) {
            current.push_back(static_cast<char>(c));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

inline std::vector<SparseFeature> extract_sparse_features(
    const std::string& input, std::uint32_t dimension = 65536) {
    const auto tokens = mosaic_tokenize(input);
    std::vector<SparseFeature> out;
    if (dimension == 0 || tokens.empty()) return out;

    out.reserve(tokens.size() * 14);
    const auto add = [&](const std::string& key, float value) {
        out.push_back({stable_feature_hash(key, dimension), value});
    };

    for (const auto& token : tokens) {
        add("w:" + token, 1.0F);
        std::string padded;
        padded.reserve(token.size() + 2);
        padded.push_back('^');
        padded += token;
        padded.push_back('$');
        for (std::size_t n = 3; n <= 5; ++n) {
            if (padded.size() < n) continue;
            for (std::size_t i = 0; i + n <= padded.size(); ++i)
                add("c" + std::to_string(n) + ":" + padded.substr(i, n), 0.20F);
        }
    }

    for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
        add("b:" + tokens[i] + "|" + tokens[i + 1], 0.80F);
    for (std::size_t i = 0; i + 2 < tokens.size(); ++i)
        add("t:" + tokens[i] + "|" + tokens[i + 1] + "|" + tokens[i + 2], 0.50F);
    return out;
}

}  // namespace mosaic
