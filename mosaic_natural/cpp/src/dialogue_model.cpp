#include "mosaic/dialogue_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mosaic {
namespace {

constexpr std::uint32_t kPad = 0;
constexpr std::uint32_t kBos = 1;
constexpr std::uint32_t kUser = 2;
constexpr std::uint32_t kAssistant = 3;
constexpr std::uint32_t kTurn = 4;
constexpr std::uint32_t kByteBase = 5;
constexpr std::uint32_t kPieceBase = 261;

bool fail(const std::string& message, std::string* error) {
    if (error) *error = message;
    return false;
}

template <typename T>
bool read_exact(std::istream& input, T* value, std::size_t bytes) {
    input.read(reinterpret_cast<char*>(value), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(input);
}

bool is_space_byte(unsigned char c) {
    return c == 9 || c == 10 || c == 11 || c == 12 || c == 13 || c == 32;
}

std::vector<std::string> split_pieces(const std::string& text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t start = i;
        while (i < text.size() && is_space_byte(static_cast<unsigned char>(text[i])))
            ++i;
        while (i < text.size() && !is_space_byte(static_cast<unsigned char>(text[i])))
            ++i;
        if (i == start) ++i;
        pieces.emplace_back(text.substr(start, i - start));
    }
    return pieces;
}

float sigmoid(float x) {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(x);
    return z / (1.0F + z);
}

void quantized_matvec(const std::vector<std::int8_t>& matrix,
                      float scale,
                      const std::vector<float>& input,
                      std::size_t rows,
                      std::size_t columns,
                      std::vector<float>& output) {
    output.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::int8_t* weights = matrix.data() + row * columns;
        float sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column)
            sum += static_cast<float>(weights[column]) * input[column];
        output[row] = sum * scale;
    }
}

bool read_quant_matrix(std::istream& input, std::size_t count,
                       std::vector<std::int8_t>& weights, float& scale) {
    if (!read_exact(input, &scale, sizeof(scale)) ||
        !(scale > 0.0F) || !std::isfinite(scale)) return false;
    weights.resize(count);
    return read_exact(input, weights.data(), count);
}

std::uint32_t sample_token(const std::vector<float>& logits,
                           const std::vector<std::uint32_t>& recent,
                           float temperature, float top_p,
                           std::uint32_t top_k,
                           std::mt19937& random) {
    struct Candidate {
        std::uint32_t id;
        float score;
        double weight;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());

    const float temp = std::max(0.15F, temperature);
    for (std::uint32_t id = 0; id < logits.size(); ++id) {
        if (id == kPad || id == kBos || id == kUser || id == kAssistant)
            continue;
        float score = logits[id] / temp;
        if (std::find(recent.begin(), recent.end(), id) != recent.end())
            score = score >= 0.0F ? score / 1.12F : score * 1.12F;
        candidates.push_back({id, score, 0.0});
    }
    if (candidates.empty()) return kTurn;

    const auto keep_top = std::min<std::size_t>(top_k, candidates.size());
    if (keep_top == 0) return kTurn;
    std::partial_sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep_top),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; }
    );
    candidates.resize(keep_top);

    const float maximum = candidates.front().score;
    double sum = 0.0;
    for (auto& candidate : candidates) {
        candidate.weight = std::exp(static_cast<double>(candidate.score - maximum));
        sum += candidate.weight;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) return kTurn;

    const double nucleus = std::clamp(static_cast<double>(top_p), 0.05, 1.0);
    double cumulative = 0.0;
    std::size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
        cumulative += candidates[keep].weight / sum;
        if (cumulative >= nucleus) {
            ++keep;
            break;
        }
    }
    keep = std::max<std::size_t>(1, std::min(keep, candidates.size()));
    candidates.resize(keep);
    std::vector<double> weights;
    weights.reserve(keep);
    for (const auto& candidate : candidates) weights.push_back(candidate.weight);
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(random)].id;
}

std::string sanitize_utf8(const std::string& input) {
    static const std::string replacement = "\xEF\xBF\xBD";
    std::string output;
    output.reserve(input.size());
    std::size_t i = 0;
    while (i < input.size()) {
        const auto first = static_cast<unsigned char>(input[i]);
        if (first <= 0x7F) {
            output.push_back(input[i++]);
            continue;
        }
        std::size_t length = 0;
        if (first >= 0xC2 && first <= 0xDF) length = 2;
        else if (first >= 0xE0 && first <= 0xEF) length = 3;
        else if (first >= 0xF0 && first <= 0xF4) length = 4;
        bool valid = length != 0 && i + length <= input.size();
        if (valid) {
            for (std::size_t j = 1; j < length; ++j) {
                const auto next = static_cast<unsigned char>(input[i + j]);
                if ((next & 0xC0) != 0x80) { valid = false; break; }
            }
        }
        if (valid && length == 3) {
            const auto second = static_cast<unsigned char>(input[i + 1]);
            if ((first == 0xE0 && second < 0xA0) ||
                (first == 0xED && second >= 0xA0)) valid = false;
        }
        if (valid && length == 4) {
            const auto second = static_cast<unsigned char>(input[i + 1]);
            if ((first == 0xF0 && second < 0x90) ||
                (first == 0xF4 && second > 0x8F)) valid = false;
        }
        if (valid) {
            output.append(input, i, length);
            i += length;
        } else {
            output += replacement;
            ++i;
        }
    }
    return output;
}

}  // namespace

bool DialogueModel::load(const std::string& path, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return fail("cannot open dialogue model: " + path, error);

    char magic[8]{};
    const char expected[8] = {'M','O','S','G','R','U','1','\0'};
    std::uint32_t vocab = 0, hidden = 0, layers = 0, context = 0, max_piece = 0;
    if (!read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, expected, sizeof(magic)) != 0)
        return fail("invalid MOSAIC dialogue model header", error);
    if (!read_exact(input, &vocab, sizeof(vocab)) ||
        !read_exact(input, &hidden, sizeof(hidden)) ||
        !read_exact(input, &layers, sizeof(layers)) ||
        !read_exact(input, &context, sizeof(context)) ||
        !read_exact(input, &max_piece, sizeof(max_piece)))
        return fail("truncated MOSAIC dialogue model header", error);

    if (vocab < kPieceBase || vocab > 65536 || hidden < 16 ||
        hidden > 2048 || layers < 1 || layers > 4 ||
        context < 16 || context > 8192 || max_piece < 2 || max_piece > 128)
        return fail("unreasonable MOSAIC model dimensions", error);

    std::vector<std::string> pieces(vocab);
    std::unordered_map<std::string, std::uint32_t> piece_map;
    piece_map.reserve(vocab - kPieceBase);
    for (std::uint32_t id = kPieceBase; id < vocab; ++id) {
        std::uint16_t length = 0;
        if (!read_exact(input, &length, sizeof(length)) || length == 0)
            return fail("invalid or truncated lexical vocabulary", error);
        std::string piece(length, '\0');
        if (!read_exact(input, piece.data(), length))
            return fail("truncated lexical vocabulary bytes", error);
        pieces[id] = piece;
        piece_map.emplace(std::move(piece), id);
    }

    std::vector<std::int8_t> embedding;
    float embedding_scale = 1.0F;
    if (!read_quant_matrix(input, static_cast<std::size_t>(vocab) * hidden,
                           embedding, embedding_scale))
        return fail("invalid or truncated embedding matrix", error);

    std::vector<GRULayer> gru(layers);
    const std::size_t gate_rows = static_cast<std::size_t>(3) * hidden;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        auto& item = gru[layer];
        if (!read_quant_matrix(input, gate_rows * hidden,
                               item.weight_ih, item.scale_ih) ||
            !read_quant_matrix(input, gate_rows * hidden,
                               item.weight_hh, item.scale_hh))
            return fail("invalid or truncated GRU matrices", error);
        item.bias_ih.resize(gate_rows);
        item.bias_hh.resize(gate_rows);
        if (!read_exact(input, item.bias_ih.data(), gate_rows * sizeof(float)) ||
            !read_exact(input, item.bias_hh.data(), gate_rows * sizeof(float)))
            return fail("truncated GRU biases", error);
    }

    std::vector<float> output_bias(vocab);
    if (!read_exact(input, output_bias.data(), vocab * sizeof(float)))
        return fail("truncated output bias", error);
    for (const float value : output_bias) {
        if (!std::isfinite(value)) return fail("non-finite output bias", error);
    }

    vocab_size_ = vocab;
    hidden_size_ = hidden;
    layers_ = layers;
    max_context_ = context;
    max_piece_bytes_ = max_piece;
    id_to_piece_ = std::move(pieces);
    piece_to_id_ = std::move(piece_map);
    embedding_ = std::move(embedding);
    embedding_scale_ = embedding_scale;
    gru_ = std::move(gru);
    output_bias_ = std::move(output_bias);
    loaded_ = true;
    return true;
}

std::vector<std::uint32_t> DialogueModel::encode(const std::string& text) const {
    std::vector<std::uint32_t> output;
    for (const auto& piece : split_pieces(text)) {
        std::size_t position = 0;
        while (position < piece.size()) {
            std::uint32_t found_id = 0;
            std::size_t found_length = 0;
            const std::size_t limit = std::min<std::size_t>(
                max_piece_bytes_, piece.size() - position);
            for (std::size_t length = limit; length > 0; --length) {
                const auto found = piece_to_id_.find(piece.substr(position, length));
                if (found != piece_to_id_.end()) {
                    found_id = found->second;
                    found_length = length;
                    break;
                }
            }
            if (found_length != 0) {
                output.push_back(found_id);
                position += found_length;
            } else {
                output.push_back(kByteBase +
                    static_cast<unsigned char>(piece[position]));
                ++position;
            }
        }
    }
    return output;
}

std::string DialogueModel::decode(const std::vector<std::uint32_t>& ids) const {
    std::string output;
    for (const auto id : ids) {
        if (id >= kByteBase && id < kPieceBase)
            output.push_back(static_cast<char>(id - kByteBase));
        else if (id >= kPieceBase && id < vocab_size_)
            output += id_to_piece_[id];
    }
    return sanitize_utf8(output);
}

bool DialogueModel::step(std::uint32_t token,
                         std::vector<std::vector<float>>& hidden,
                         std::vector<float>& logits,
                         std::vector<float>& layer_input,
                         std::vector<float>& input_gates,
                         std::vector<float>& hidden_gates) const {
    if (!loaded_ || token >= vocab_size_ || hidden.size() != layers_ ||
        layer_input.size() != hidden_size_) return false;
    const std::size_t embedding_offset = static_cast<std::size_t>(token) * hidden_size_;
    for (std::size_t j = 0; j < hidden_size_; ++j)
        layer_input[j] =
            static_cast<float>(embedding_[embedding_offset + j]) * embedding_scale_;

    const std::size_t gate_rows = static_cast<std::size_t>(3) * hidden_size_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        const auto& weights = gru_[layer];
        quantized_matvec(weights.weight_ih, weights.scale_ih,
                         layer_input, gate_rows, hidden_size_, input_gates);
        quantized_matvec(weights.weight_hh, weights.scale_hh,
                         hidden[layer], gate_rows, hidden_size_, hidden_gates);

        // Update the recurrent state in place; reuse the same buffers at every token.
        auto& state = hidden[layer];
        for (std::size_t j = 0; j < hidden_size_; ++j) {
            const float previous = state[j];
            const float reset = sigmoid(
                input_gates[j] + weights.bias_ih[j] +
                hidden_gates[j] + weights.bias_hh[j]);
            const float update = sigmoid(
                input_gates[hidden_size_ + j] +
                weights.bias_ih[hidden_size_ + j] +
                hidden_gates[hidden_size_ + j] +
                weights.bias_hh[hidden_size_ + j]);
            const float candidate = std::tanh(
                input_gates[2 * hidden_size_ + j] +
                weights.bias_ih[2 * hidden_size_ + j] +
                reset * (hidden_gates[2 * hidden_size_ + j] +
                         weights.bias_hh[2 * hidden_size_ + j]));
            state[j] = (1.0F - update) * candidate + update * previous;
        }
        std::copy(state.begin(), state.end(), layer_input.begin());
    }

    logits.resize(vocab_size_);
    const auto& final_hidden = hidden.back();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t token_id = 0;
         token_id < static_cast<std::int64_t>(vocab_size_); ++token_id) {
        const std::size_t base = static_cast<std::size_t>(token_id) * hidden_size_;
        float score = output_bias_[static_cast<std::size_t>(token_id)];
        for (std::size_t j = 0; j < hidden_size_; ++j)
            score += static_cast<float>(embedding_[base + j]) *
                     embedding_scale_ * final_hidden[j];
        logits[static_cast<std::size_t>(token_id)] = score;
    }
    return true;
}

std::string DialogueModel::generate(
    const std::vector<std::uint32_t>& prompt,
    std::size_t max_new_tokens,
    float temperature, float top_p, std::uint32_t top_k,
    std::uint32_t seed) const {
    if (!loaded_ || prompt.empty() || max_new_tokens == 0) return {};

    std::vector<std::uint32_t> bounded_prompt = prompt;
    if (bounded_prompt.size() > max_context_) {
        bounded_prompt.erase(
            bounded_prompt.begin(),
            bounded_prompt.end() - static_cast<std::ptrdiff_t>(max_context_));
    }

    std::vector<std::vector<float>> hidden(
        layers_, std::vector<float>(hidden_size_, 0.0F));
    std::vector<float> logits(vocab_size_);
    std::vector<float> layer_input(hidden_size_);
    std::vector<float> input_gates(static_cast<std::size_t>(3) * hidden_size_);
    std::vector<float> hidden_gates(static_cast<std::size_t>(3) * hidden_size_);
    for (const auto token : bounded_prompt) {
        if (!step(token, hidden, logits, layer_input, input_gates, hidden_gates))
            return {};
    }

    if (seed == 0) {
        seed = static_cast<std::uint32_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    std::mt19937 random(seed);
    std::vector<std::uint32_t> generated;
    std::vector<std::uint32_t> recent;
    generated.reserve(max_new_tokens);
    recent.reserve(32);

    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto next = sample_token(
            logits, recent, temperature, top_p, top_k, random);
        if (next == kTurn) break;
        generated.push_back(next);
        recent.push_back(next);
        if (recent.size() > 32) recent.erase(recent.begin());
        if (!step(next, hidden, logits, layer_input, input_gates, hidden_gates)) break;
    }

    std::string reply = decode(generated);
    while (!reply.empty() &&
           (reply.back() == ' ' || reply.back() == '\n' ||
            reply.back() == '\r' || reply.back() == '\t'))
        reply.pop_back();
    while (!reply.empty() &&
           (reply.front() == ' ' || reply.front() == '\n' ||
            reply.front() == '\r' || reply.front() == '\t'))
        reply.erase(reply.begin());
    return reply;
}

}  // namespace mosaic
