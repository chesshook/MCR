#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

bool is_word_byte(unsigned char c) {
    return c >= 128 || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

std::vector<std::string> tokenize(const std::string& input) {
    std::vector<std::string> result;
    std::size_t i = 0;
    while (i < input.size()) {
        const auto c = static_cast<unsigned char>(input[i]);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
            ++i;
            continue;
        }
        if (is_word_byte(c)) {
            const std::size_t start = i++;
            while (i < input.size()) {
                const auto next = static_cast<unsigned char>(input[i]);
                if (is_word_byte(next)) {
                    ++i;
                } else if (next == '\'' && i + 1 < input.size() &&
                           is_word_byte(static_cast<unsigned char>(input[i + 1]))) {
                    ++i;
                } else {
                    break;
                }
            }
            std::string token = input.substr(start, i - start);
            for (char& ch : token) {
                const auto value = static_cast<unsigned char>(ch);
                if (value >= 'A' && value <= 'Z') ch = static_cast<char>(value + ('a' - 'A'));
            }
            result.push_back(std::move(token));
        } else {
            result.emplace_back(1, input[i]);
            ++i;
        }
    }
    return result;
}

std::uint32_t feature_hash(const std::string& feature, std::uint64_t buckets) {
    std::uint64_t h = kFnvOffset;
    for (unsigned char c : feature) {
        h ^= c;
        h *= kFnvPrime;
    }
    return static_cast<std::uint32_t>(h % buckets);
}

std::vector<std::uint32_t> extract_features(const std::string& text, std::uint64_t buckets) {
    const auto tokens = tokenize(text);
    std::unordered_set<std::uint32_t> found;
    found.reserve(tokens.size() * 24 + 1);
    found.insert(feature_hash("bias", buckets));
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        if (!token.empty() && is_word_byte(static_cast<unsigned char>(token.front()))) {
            found.insert(feature_hash("w:" + token, buckets));
            const std::string padded = "^" + token + "$";
            for (std::size_t width : {3U, 4U, 5U}) {
                if (padded.size() >= width) {
                    for (std::size_t j = 0; j + width <= padded.size(); ++j) {
                        found.insert(feature_hash("c:" + padded.substr(j, width), buckets));
                    }
                }
            }
        } else {
            found.insert(feature_hash("p:" + token, buckets));
        }
        if (i > 0) {
            found.insert(feature_hash("b:" + tokens[i - 1] + "_" + token, buckets));
        }
    }
    std::vector<std::uint32_t> result(found.begin(), found.end());
    std::sort(result.begin(), result.end());
    return result;
}

struct Prediction {
    std::string label;
    float score = 0.0F;
    float margin = 0.0F;
    std::string runner_up;
};

class SparseLinearModel {
public:
    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Cannot open model: " + path);
        std::string magic;
        std::getline(in, magic);
        if (magic != "MCR_CPU_LINEAR_V1") throw std::runtime_error("Unsupported model format");
        std::size_t label_count = 0;
        if (!(in >> buckets_ >> min_margin_ >> label_count) ||
            buckets_ < 1024 || label_count == 0 || label_count > 100000 || min_margin_ < 0.0F) {
            throw std::runtime_error("Invalid model header");
        }
        labels_.assign(label_count, {});
        for (std::size_t i = 0; i < label_count; ++i) {
            char marker = 0;
            std::size_t index = 0;
            std::string label;
            if (!(in >> marker >> index >> label) || marker != 'L' || index >= label_count) {
                throw std::runtime_error("Invalid label record");
            }
            labels_[index] = std::move(label);
        }
        std::size_t weight_count = 0;
        if (!(in >> weight_count)) throw std::runtime_error("Missing weight count");
        weights_.clear();
        weights_.reserve(weight_count / 2 + 1);
        for (std::size_t i = 0; i < weight_count; ++i) {
            char marker = 0;
            std::uint32_t feature = 0;
            std::uint32_t label = 0;
            float weight = 0.0F;
            if (!(in >> marker >> feature >> label >> weight) ||
                marker != 'W' || label >= label_count || feature >= buckets_) {
                throw std::runtime_error("Invalid weight record");
            }
            weights_[feature].emplace_back(label, weight);
        }
    }

    Prediction predict(const std::string& text) const {
        if (labels_.empty()) throw std::runtime_error("Model is not loaded");
        const auto features = extract_features(text, buckets_);
        std::vector<float> scores(labels_.size(), 0.0F);
        for (const auto feature : features) {
            const auto it = weights_.find(feature);
            if (it == weights_.end()) continue;
            for (const auto& entry : it->second) scores[entry.first] += entry.second;
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(std::max<std::size_t>(1, features.size())));
        for (float& value : scores) value *= scale;

        std::size_t best = 0, second = 0;
        for (std::size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > scores[best]) {
                second = best;
                best = i;
            } else if (i != best && (second == best || scores[i] > scores[second])) {
                second = i;
            }
        }
        Prediction result;
        result.label = labels_[best];
        result.score = scores[best];
        result.runner_up = labels_.size() > 1 ? labels_[second] : "";
        result.margin = labels_.size() > 1 ? scores[best] - scores[second] : scores[best];
        return result;
    }

    std::size_t label_count() const { return labels_.size(); }
    std::size_t feature_count() const { return weights_.size(); }
    float min_margin() const { return min_margin_; }

private:
    std::uint64_t buckets_ = 0;
    float min_margin_ = 0.0F;
    std::vector<std::string> labels_;
    std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint32_t, float>>> weights_;
};
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: mcr_intent <model.txt>\n"
                  << "Reads one message per line and prints the predicted intent.\n";
        return 2;
    }
    try {
        SparseLinearModel model;
        model.load(argv[1]);
        std::cerr << "MCR CPU intent module loaded: " << model.label_count()
                  << " labels, " << model.feature_count() << " active feature buckets.\n"
                  << "This module classifies intent only; it is not a complete chatbot.\n";
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            const auto p = model.predict(line);
            const bool abstain = p.margin < model.min_margin();
            std::cout << (abstain ? "__uncertain__" : p.label)
                      << "\tcandidate=" << p.label
                      << "\tscore=" << std::fixed << std::setprecision(4) << p.score
                      << "\tmargin=" << p.margin << "\tthreshold=" << model.min_margin()
                      << "\trunner_up=" << p.runner_up << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "MCR CPU intent error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
