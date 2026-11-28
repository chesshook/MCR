#include "mosaic/agent.hpp"
#include "mosaic/features.hpp"
#include "mosaic/intent_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        p * static_cast<double>(values.size() - 1));
    return values[index];
}

void print_stats(const std::string& label, const std::vector<double>& values) {
    double total = 0.0;
    for (const double value : values) total += value;
    const double mean = values.empty() ? 0.0 : total / values.size();
    std::cout << label
              << " | p50=" << percentile(values, 0.50) << " us"
              << " | p95=" << percentile(values, 0.95) << " us"
              << " | mean=" << mean << " us\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "intent_model.mcr";
    std::size_t iterations = 10000;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--iterations" && i + 1 < argc)
            iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
        else if (arg == "--help") {
            std::cout << "Usage: mosaic_bench [--model intent_model.mcr] "
                         "[--iterations 10000]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 2;
        }
    }
    if (iterations == 0 || iterations > 10000000) {
        std::cerr << "iterations must be between 1 and 10000000\n";
        return 2;
    }

    mosaic::IntentModel intent;
    std::string error;
    if (model_path != "-" && !intent.load(model_path, &error))
        std::cerr << "Intent model unavailable: " << error << '\n';

    const std::vector<std::string> samples = {
        "hello there",
        "can you explain this to me",
        "what is my name",
        "I am building a small game",
        "please help me understand this problem",
        "what is 173 times 29",
        "actually I meant a chess engine",
        "tell me what we discussed earlier"
    };

    std::vector<double> inference_us;
    inference_us.reserve(iterations);
    std::size_t checksum = 0;
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto& text = samples[i % samples.size()];
        const auto start = std::chrono::steady_clock::now();
        if (intent.loaded()) {
            const auto prediction = intent.predict(text);
            checksum += prediction.class_id;
        } else {
            const auto features = mosaic::extract_sparse_features(text);
            checksum += features.size();
        }
        const auto end = std::chrono::steady_clock::now();
        inference_us.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }

    mosaic::Agent agent;
    if (intent.loaded()) (void)agent.load_intent_model(model_path, nullptr);
    std::vector<double> dialogue_us;
    dialogue_us.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        const std::string& text = (i % 2 == 0) ? samples[0] : samples[5];
        const auto start = std::chrono::steady_clock::now();
        const auto response = agent.respond(text);
        checksum += response.size();
        const auto end = std::chrono::steady_clock::now();
        dialogue_us.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::cout << "MOSAIC native single-process benchmark\n"
              << "Iterations: " << iterations << '\n'
              << "Intent model: " << (intent.loaded() ? "loaded" : "not loaded") << '\n';
    print_stats(intent.loaded() ? "Intent inference" : "Sparse feature extraction",
                inference_us);
    print_stats("Dialogue response path", dialogue_us);
    std::cout << "Checksum: " << checksum << '\n';
    return 0;
}
