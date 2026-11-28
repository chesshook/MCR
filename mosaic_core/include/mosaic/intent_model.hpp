#pragma once

#include "mosaic/features.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mosaic {

struct IntentPrediction {
    std::string label;
    float probability = 0.0F;
    std::size_t class_id = 0;
};

class IntentModel {
public:
    bool load(const std::string& path, std::string* error = nullptr) {
        std::ifstream input(path, std::ios::binary);
        if (!input) return fail("cannot open intent model: " + path, error);

        char magic[8]{};
        std::uint32_t dimension = 0;
        std::uint32_t classes = 0;
        float scale = 0.0F;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        input.read(reinterpret_cast<char*>(&classes), sizeof(classes));
        input.read(reinterpret_cast<char*>(&scale), sizeof(scale));

        const char expected[8] = {'M','C','R','I','N','T','1','\0'};
        if (!input || std::memcmp(magic, expected, sizeof(magic)) != 0)
            return fail("invalid intent-model magic or truncated header", error);
        if (dimension == 0 || dimension > 1048576 || classes == 0 || classes > 4096)
            return fail("unreasonable intent-model dimensions", error);
        if (!(scale > 0.0F) || !std::isfinite(scale))
            return fail("invalid intent-model quantization scale", error);

        std::vector<std::string> labels;
        labels.reserve(classes);
        for (std::uint32_t i = 0; i < classes; ++i) {
            std::uint16_t length = 0;
            input.read(reinterpret_cast<char*>(&length), sizeof(length));
            if (!input || length == 0 || length > 1024)
                return fail("invalid intent label entry", error);
            std::string label(length, '\0');
            input.read(label.data(), static_cast<std::streamsize>(length));
            if (!input) return fail("truncated intent labels", error);
            labels.push_back(std::move(label));
        }

        std::vector<float> bias(classes);
        input.read(reinterpret_cast<char*>(bias.data()),
                   static_cast<std::streamsize>(bias.size() * sizeof(float)));
        if (!input) return fail("truncated intent biases", error);

        const std::uint64_t count =
            static_cast<std::uint64_t>(dimension) * static_cast<std::uint64_t>(classes);
        if (count > 100000000ULL)
            return fail("intent-model weight matrix is too large", error);
        std::vector<std::int8_t> weights(static_cast<std::size_t>(count));
        input.read(reinterpret_cast<char*>(weights.data()),
                   static_cast<std::streamsize>(weights.size()));
        if (!input) return fail("truncated intent weights", error);

        dimension_ = dimension;
        classes_ = classes;
        scale_ = scale;
        labels_ = std::move(labels);
        bias_ = std::move(bias);
        weights_ = std::move(weights);
        loaded_ = true;
        return true;
    }

    bool loaded() const noexcept { return loaded_; }

    IntentPrediction predict(const std::string& text) const {
        IntentPrediction none;
        if (!loaded_) return none;

        const auto features = extract_sparse_features(text, dimension_);
        if (features.empty()) return none;

        std::vector<float> logits = bias_;
        for (const auto& feature : features) {
            if (feature.id >= dimension_) continue;
            const std::size_t base = static_cast<std::size_t>(feature.id) * classes_;
            const float factor = feature.value * scale_;
            for (std::size_t c = 0; c < classes_; ++c) {
                logits[c] += static_cast<float>(weights_[base + c]) * factor;
            }
        }
        const float denominator = std::sqrt(static_cast<float>(features.size()));
        if (denominator > 0.0F) {
            for (float& value : logits) value /= denominator;
        }

        float maximum = -std::numeric_limits<float>::infinity();
        std::size_t best = 0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > maximum) {
                maximum = logits[i];
                best = i;
            }
        }

        double total = 0.0;
        for (const float value : logits) total += std::exp(static_cast<double>(value - maximum));
        const float probability = total > 0.0
            ? static_cast<float>(1.0 / total)
            : 0.0F;
        return {labels_[best], probability, best};
    }

private:
    static bool fail(const std::string& message, std::string* error) {
        if (error) *error = message;
        return false;
    }

    bool loaded_ = false;
    std::uint32_t dimension_ = 0;
    std::uint32_t classes_ = 0;
    float scale_ = 1.0F;
    std::vector<std::string> labels_;
    std::vector<float> bias_;
    std::vector<std::int8_t> weights_;
};

}  // namespace mosaic
