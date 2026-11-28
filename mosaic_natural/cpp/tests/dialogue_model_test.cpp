#include "mosaic/dialogue_model.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::string path = "mosaic_dialogue_model_test.mos";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        assert(output.good());
        output.write("MOSGRU1\0", 8);

        const std::uint32_t vocab = 263;
        const std::uint32_t hidden = 16;
        const std::uint32_t layers = 1;
        const std::uint32_t context = 128;
        const std::uint32_t max_piece = 32;
        output.write(reinterpret_cast<const char*>(&vocab), 4);
        output.write(reinterpret_cast<const char*>(&hidden), 4);
        output.write(reinterpret_cast<const char*>(&layers), 4);
        output.write(reinterpret_cast<const char*>(&context), 4);
        output.write(reinterpret_cast<const char*>(&max_piece), 4);

        const std::string pieces[] = {"hello", " there"};
        for (const auto& piece : pieces) {
            const auto length = static_cast<std::uint16_t>(piece.size());
            output.write(reinterpret_cast<const char*>(&length), 2);
            output.write(piece.data(), static_cast<std::streamsize>(piece.size()));
        }

        const float scale = 1.0F;
        const std::vector<std::int8_t> embedding(vocab * hidden, 0);
        output.write(reinterpret_cast<const char*>(&scale), 4);
        output.write(reinterpret_cast<const char*>(embedding.data()),
                     static_cast<std::streamsize>(embedding.size()));

        const std::size_t gate_count = 3 * hidden * hidden;
        const std::vector<std::int8_t> matrix(gate_count, 0);
        for (int which = 0; which < 2; ++which) {
            output.write(reinterpret_cast<const char*>(&scale), 4);
            output.write(reinterpret_cast<const char*>(matrix.data()),
                         static_cast<std::streamsize>(matrix.size()));
        }
        const std::vector<float> bias(3 * hidden, 0.0F);
        output.write(reinterpret_cast<const char*>(bias.data()),
                     static_cast<std::streamsize>(bias.size() * sizeof(float)));
        output.write(reinterpret_cast<const char*>(bias.data()),
                     static_cast<std::streamsize>(bias.size() * sizeof(float)));

        std::vector<float> output_bias(vocab, 0.0F);
        output_bias[4] = 20.0F;
        output.write(reinterpret_cast<const char*>(output_bias.data()),
                     static_cast<std::streamsize>(output_bias.size() * sizeof(float)));
        assert(output.good());
    }

    mosaic::DialogueModel model;
    std::string error;
    assert(model.load(path, &error));
    assert(model.loaded());
    assert(model.vocab_size() == 263);

    const auto ids = model.encode("hello there");
    assert(ids.size() == 2);
    assert(ids[0] == 261 && ids[1] == 262);
    assert(model.decode(ids) == "hello there");

    const auto unknown = model.encode("Sanjay");
    assert(!unknown.empty());
    assert(model.decode(unknown) == "Sanjay");
    const std::vector<std::uint32_t> invalid_utf8 = {5U + 0xFFU};
    assert(model.decode(invalid_utf8) == "\xEF\xBF\xBD");

    const std::vector<std::uint32_t> prompt = {1, 2, 261, 4, 3};
    const auto reply = model.generate(prompt, 8, 0.8F, 0.9F, 48, 7);
    assert(reply.empty());

    std::remove(path.c_str());
    std::cout << "MOSAIC native dialogue-model tests passed\n";
}
