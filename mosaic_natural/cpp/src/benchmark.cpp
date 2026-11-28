#include "mosaic/dialogue_model.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        p * static_cast<double>(values.size() - 1));
    return values[index];
}
}

int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    std::string model_path = "mosaic_dialogue.mos";
    std::size_t iterations = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--iterations" && i + 1 < argc)
            iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
        else if (arg == "--help") {
            std::cout << "Usage: mosaic_natural_bench [--model mosaic_dialogue.mos] "
                         "[--iterations 30]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 2;
        }
    }
    mosaic::DialogueModel model;
    std::string error;
    if (!model.load(model_path, &error)) {
        std::cerr << error << '\n';
        return 3;
    }

    const std::vector<std::uint32_t> prompt = [&]() {
        std::vector<std::uint32_t> result{1, 2};
        const auto user = model.encode("Hi! What would you like to talk about today?");
        result.insert(result.end(), user.begin(), user.end());
        result.push_back(4);
        result.push_back(3);
        return result;
    }();

    for (int warmup = 0; warmup < 3; ++warmup)
        (void)model.generate(prompt, 48, 0.82F, 0.90F, 48);

    std::vector<double> times_ms;
    times_ms.reserve(iterations);
    std::size_t produced_bytes = 0;
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto reply = model.generate(prompt, 64, 0.82F, 0.90F, 48);
        const auto end = std::chrono::steady_clock::now();
        produced_bytes += reply.size();
        times_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    const double mean = [&]() {
        double total = 0.0;
        for (double value : times_ms) total += value;
        return total / std::max<std::size_t>(1, times_ms.size());
    }();

    std::cout << "MOSAIC native generation benchmark (CPU, single process)\n"
              << "Model bytes: " << produced_bytes << " generated over "
              << iterations << " turns\n"
              << "Generation p50: " << percentile(times_ms, 0.50) << " ms\n"
              << "Generation p95: " << percentile(times_ms, 0.95) << " ms\n"
              << "Generation mean: " << mean << " ms\n"
              << "Record CPU model, compiler and build flags with these results.\n";
}
