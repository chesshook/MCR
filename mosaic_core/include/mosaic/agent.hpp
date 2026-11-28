#pragma once

#include "mosaic/intent_model.hpp"
#include "mosaic/state_kernel.hpp"

#include <string>

namespace mosaic {

class Agent {
public:
    Agent() = default;

    bool load_intent_model(const std::string& path, std::string* error = nullptr);
    bool load_memory(const std::string& path, std::string* error = nullptr);
    bool save_memory(const std::string& path, std::string* error = nullptr) const;

    std::string respond(const std::string& message);
    void reset();

    const StateKernel& state() const noexcept { return state_; }
    bool has_intent_model() const noexcept { return intent_model_.loaded(); }

private:
    StateKernel state_;
    IntentModel intent_model_;
    std::uint64_t turn_ = 0;

    void remember(const std::string& predicate, const std::string& value,
                  bool explicit_correction);
};

}  // namespace mosaic
