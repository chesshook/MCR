#include "mosaic/agent.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string model_path = "intent_model.mcr";
    std::string memory_path = "mosaic_memory.tsv";
    std::string lexicon_path = "wordnet.tsv";
    bool memory_enabled = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--memory" && i + 1 < argc) memory_path = argv[++i];
        else if (arg == "--lexicon" && i + 1 < argc) lexicon_path = argv[++i];
        else if (arg == "--no-memory") memory_enabled = false;
        else if (arg == "--help") {
            std::cout << "Usage: mosaic_chat [--model intent_model.mcr] "
                         "[--memory mosaic_memory.tsv] [--lexicon wordnet.tsv] [--no-memory]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 2;
        }
    }

    mosaic::Agent agent;
    std::string error;
    if (model_path != "-" && !agent.load_intent_model(model_path, &error)) {
        std::cerr << "Intent model unavailable: " << error
                  << "\nContinuing with the native semantic/dialogue core only.\n";
    }
    if (lexicon_path != "-" && !agent.load_lexicon(lexicon_path, &error))
        std::cerr << "Lexicon unavailable: " << error << "\nContinuing without dictionary lookup.\n";
    if (memory_enabled && !agent.load_memory(memory_path, &error)) {
        std::cerr << "Memory load failed: " << error << '\n';
        return 3;
    }

    std::cout << "MOSAIC / MCR native chat. Type /help for commands.\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "/exit" || line == "/quit") break;
        if (line == "/help") {
            std::cout << "Commands: /help, /reset, /exit. "
                         "Facts are stored in the local memory file.\n";
            continue;
        }
        if (line == "/reset") {
            agent.reset();
            if (memory_enabled && !agent.save_memory(memory_path, &error))
                std::cerr << "Memory save failed: " << error << '\n';
            std::cout << "MCR> Conversation memory cleared.\n";
            continue;
        }

        const std::string answer = agent.respond(line);
        std::cout << "MCR> " << answer << '\n';
        if (memory_enabled && !agent.save_memory(memory_path, &error))
            std::cerr << "Memory save failed: " << error << '\n';
    }

    if (memory_enabled && !agent.save_memory(memory_path, &error)) {
        std::cerr << "Memory save failed: " << error << '\n';
        return 4;
    }
    return 0;
}
