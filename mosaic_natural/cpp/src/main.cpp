#include "mosaic/dialogue_model.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Message {
    bool assistant = false;
    std::string text;
};

std::string hex_encode(const std::string& input) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(input.size() * 2);
    for (const unsigned char byte : input) {
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 15]);
    }
    return output;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hex_decode(const std::string& input, std::string& output) {
    if (input.size() % 2 != 0) return false;
    output.clear();
    output.reserve(input.size() / 2);
    for (std::size_t i = 0; i < input.size(); i += 2) {
        const int high = hex_value(input[i]);
        const int low = hex_value(input[i + 1]);
        if (high < 0 || low < 0) return false;
        output.push_back(static_cast<char>((high << 4) | low));
    }
    return true;
}

bool load_history(const std::string& path, std::vector<Message>& messages) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return true;
    std::string line;
    if (!std::getline(input, line) || line != "MOSAIC_DIALOGUE_HISTORY_V1")
        return false;

    while (std::getline(input, line)) {
        if (line.size() < 2 || line[1] != '\t') return false;
        std::string text;
        if (!hex_decode(line.substr(2), text)) return false;
        if (line[0] == 'U') messages.push_back({false, std::move(text)});
        else if (line[0] == 'A') messages.push_back({true, std::move(text)});
        else return false;
    }
    if (messages.size() > 2000)
        messages.erase(messages.begin(), messages.end() - 2000);
    return true;
}

bool save_history(const std::string& path, const std::vector<Message>& messages) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output << "MOSAIC_DIALOGUE_HISTORY_V1\n";
    for (const auto& message : messages) {
        output << (message.assistant ? 'A' : 'U') << '\t'
               << hex_encode(message.text) << '\n';
    }
    return static_cast<bool>(output);
}

std::vector<std::uint32_t> make_prompt(
    const mosaic::DialogueModel& model,
    const std::vector<Message>& history,
    const std::string& current_user) {
    constexpr std::uint32_t kBos = 1;
    constexpr std::uint32_t kUser = 2;
    constexpr std::uint32_t kAssistant = 3;
    constexpr std::uint32_t kTurn = 4;

    std::vector<std::uint32_t> prompt;
    prompt.push_back(kBos);

    const std::size_t first = history.size() > 12 ? history.size() - 12 : 0;
    for (std::size_t i = first; i < history.size(); ++i) {
        prompt.push_back(history[i].assistant ? kAssistant : kUser);
        const auto encoded = model.encode(history[i].text);
        prompt.insert(prompt.end(), encoded.begin(), encoded.end());
        prompt.push_back(kTurn);
    }

    prompt.push_back(kUser);
    const auto current = model.encode(current_user);
    prompt.insert(prompt.end(), current.begin(), current.end());
    prompt.push_back(kTurn);
    prompt.push_back(kAssistant);

    if (prompt.size() > model.max_context()) {
        prompt.erase(prompt.begin(),
                     prompt.end() - static_cast<std::ptrdiff_t>(model.max_context()));
        if (!prompt.empty() && prompt.front() != kBos) {
            // A truncated context remains valid without the original BOS marker.
        }
    }
    return prompt;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    std::string model_path = "mosaic_dialogue.mos";
    std::string history_path = "mosaic_history.tsv";
    std::size_t max_new_tokens = 128;
    float temperature = 0.82F;
    float top_p = 0.90F;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--history" && i + 1 < argc) history_path = argv[++i];
        else if (arg == "--max-new" && i + 1 < argc)
            max_new_tokens = static_cast<std::size_t>(std::stoull(argv[++i]));
        else if (arg == "--temperature" && i + 1 < argc)
            temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc)
            top_p = std::stof(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: mosaic_natural_chat --model mosaic_dialogue.mos "
                         "[--history mosaic_history.tsv] [--max-new 128] "
                         "[--temperature 0.82] [--top-p 0.90]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 2;
        }
    }

    mosaic::DialogueModel model;
    std::string error;
    if (!model.load(model_path, &error)) {
        std::cerr << "Could not load the baked dialogue model: " << error << '\n';
        return 3;
    }

    std::vector<Message> history;
    if (!load_history(history_path, history)) {
        std::cerr << "Conversation history is corrupt or has an unsupported version.\n";
        return 4;
    }

    std::cout << "MOSAIC natural-language chat is ready. Type /help for commands.\n" << std::flush;
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "/exit" || input == "/quit") break;
        if (input == "/help") {
            std::cout << "Commands: /help, /reset, /exit. "
                         "This model generates replies from its baked conversation weights.\n" << std::flush;
            continue;
        }
        if (input == "/reset") {
            history.clear();
            if (!save_history(history_path, history)) {
                std::cerr << "Could not save empty conversation history.\n";
                return 5;
            }
            std::cout << "MCR> Conversation history cleared.\n" << std::flush;
            continue;
        }
        if (input.empty()) {
            std::cout << "MCR> " << '\n' << std::flush;
            continue;
        }

        const auto prompt = make_prompt(model, history, input);
        const std::string reply = model.generate(
            prompt, max_new_tokens, temperature, top_p, 48
        );
        std::cout << "MCR> " << reply << '\n' << std::flush;

        history.push_back({false, input});
        history.push_back({true, reply});
        if (history.size() > 2000)
            history.erase(history.begin(), history.end() - 2000);
        if (!save_history(history_path, history)) {
            std::cerr << "Could not persist conversation history.\n";
            return 5;
        }
    }

    if (!save_history(history_path, history)) {
        std::cerr << "Could not persist conversation history.\n";
        return 5;
    }
    return 0;
}
