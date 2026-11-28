#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mosaic {

class DialogueModel {
public:
    bool load(const std::string& path, std::string* error = nullptr);
    bool loaded() const noexcept { return loaded_; }

    std::uint32_t vocab_size() const noexcept { return vocab_size_; }
    std::uint32_t hidden_size() const noexcept { return hidden_size_; }
    std::uint32_t max_context() const noexcept { return max_context_; }

    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& ids) const;

    std::string generate(
        const std::vector<std::uint32_t>& prompt,
        std::size_t max_new_tokens = 128,
        float temperature = 0.82F,
        float top_p = 0.90F,
        std::uint32_t top_k = 48,
        std::uint32_t seed = 0) const;

private:
    struct GRULayer {
        std::vector<std::int8_t> weight_ih;
        std::vector<std::int8_t> weight_hh;
        float scale_ih = 1.0F;
        float scale_hh = 1.0F;
        std::vector<float> bias_ih;
        std::vector<float> bias_hh;
    };

    bool loaded_ = false;
    std::uint32_t vocab_size_ = 0;
    std::uint32_t hidden_size_ = 0;
    std::uint32_t layers_ = 0;
    std::uint32_t max_context_ = 0;
    std::uint32_t max_piece_bytes_ = 0;

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, std::uint32_t> piece_to_id_;
    std::vector<std::int8_t> embedding_;
    float embedding_scale_ = 1.0F;
    std::vector<GRULayer> gru_;
    std::vector<float> output_bias_;

    bool step(std::uint32_t token,
              std::vector<std::vector<float>>& hidden,
              std::vector<float>& logits,
              std::vector<float>& layer_input,
              std::vector<float>& input_gates,
              std::vector<float>& hidden_gates) const;
};

}  // namespace mosaic
