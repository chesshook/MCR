#include "mosaic/agent.hpp"
#include "mosaic/features.hpp"
#include "mosaic/intent_model.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::string lexicon_path = "mosaic_agent_test_wordnet.tsv";
    {
        std::ofstream out(lexicon_path, std::ios::binary | std::ios::trunc);
        out << "MCR_WORDNET_2025_PLUS_V1\n";
        out << "apple\tn\ta fruit produced by an apple tree\torchard apple;malus pumila\tfruit;edible fruit\n";
        out << "dog\tn\ta domesticated carnivorous mammal\tcanis familiaris;domestic dog\tcanine;domestic animal\n";
        assert(out.good());
    }

    mosaic::Agent agent;
    std::string lexical_error;
    assert(agent.load_lexicon(lexicon_path, &lexical_error));

    const auto greeting = agent.respond("Hi");
    assert(greeting.find("Hi!") == 0);
    assert(greeting.find("102 boxes") == std::string::npos);

    mosaic::Agent identity;
    assert(identity.respond("What is your name?").find("I'm MCR") != std::string::npos);

    const auto introduction = agent.respond("My name is Alex.");
    assert(introduction.find("Alex") != std::string::npos);
    assert(agent.respond("What is my name?").find("Alex") != std::string::npos);

    agent.respond("I'm building a game.");
    assert(agent.respond("What am I building?").find("a game") != std::string::npos);

    agent.respond("Actually, I'm building a chess engine.");
    const auto current = agent.respond("What is my project?");
    assert(current.find("chess engine") != std::string::npos);
    const auto project_history = agent.state().query("user", "current_project", true);
    assert(project_history.size() == 2);
    assert(project_history.front().active);
    assert(project_history.back().active == false);

    assert(agent.respond("What is 173 times 29?").find("5017") != std::string::npos);
    assert(agent.respond("How are you?").find("ready") != std::string::npos);
    assert(agent.respond("What is a dog?").find("domesticated carnivorous mammal") != std::string::npos);
    assert(agent.respond("Give me synonyms for dog").find("domestic dog") != std::string::npos);

    assert(agent.respond("I live in Ludhiana.").find("Ludhiana") != std::string::npos);
    assert(agent.respond("Where do I live?").find("Ludhiana") != std::string::npos);
    assert(agent.respond("My favorite color is blue.").find("blue") != std::string::npos);
    assert(agent.respond("What is my favorite color?").find("blue") != std::string::npos);
    agent.respond("I like coffee.");
    assert(agent.respond("What do I like?").find("coffee") != std::string::npos);
    agent.respond("Actually, I prefer tea.");
    assert(agent.respond("What do I prefer?").find("tea") != std::string::npos);

    const std::string memory_path = "mosaic_agent_test_memory.tsv";
    std::string error;
    assert(agent.save_memory(memory_path, &error));
    mosaic::Agent restored;
    assert(restored.load_memory(memory_path, &error));
    assert(restored.respond("What is my name?").find("Alex") != std::string::npos);
    assert(restored.respond("What am I building?").find("chess engine") != std::string::npos);
    assert(restored.respond("Where do I live?").find("Ludhiana") != std::string::npos);
    assert(restored.respond("What is my favorite color?").find("blue") != std::string::npos);
    std::remove(memory_path.c_str());
    std::remove(lexicon_path.c_str());

    // Exercise the exact binary format emitted by mosaic_train.py.
    const std::string model_path = "mosaic_intent_test.mcr";
    {
        std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
        assert(output.good());
        const char magic[8] = {'M','C','R','I','N','T','1','\0'};
        const std::uint32_t dimension = 64;
        const std::uint32_t classes = 2;
        const float scale = 0.10F;
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        output.write(reinterpret_cast<const char*>(&classes), sizeof(classes));
        output.write(reinterpret_cast<const char*>(&scale), sizeof(scale));

        const std::string labels[2] = {"oos", "greeting"};
        for (const auto& label : labels) {
            const auto length = static_cast<std::uint16_t>(label.size());
            output.write(reinterpret_cast<const char*>(&length), sizeof(length));
            output.write(label.data(), static_cast<std::streamsize>(label.size()));
        }
        const float bias[2] = {0.0F, 0.0F};
        output.write(reinterpret_cast<const char*>(bias), sizeof(bias));
        std::vector<std::int8_t> weights(dimension * classes, 0);
        const auto feature = mosaic::stable_feature_hash("w:hello", dimension);
        weights[static_cast<std::size_t>(feature) * classes + 1] = 120;
        output.write(reinterpret_cast<const char*>(weights.data()),
                     static_cast<std::streamsize>(weights.size()));
        assert(output.good());
    }

    mosaic::IntentModel intent_model;
    assert(intent_model.load(model_path, &error));
    const auto prediction = intent_model.predict("hello");
    assert(prediction.label == "greeting");
    assert(prediction.probability > 0.5F);
    std::remove(model_path.c_str());

    std::cout << "MOSAIC dialogue-agent tests passed\n";
    return 0;
}
