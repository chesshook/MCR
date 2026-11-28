#include "mosaic/state_kernel.hpp"

#include <iostream>
#include <string>

int main() {
    mosaic::StateKernel state;

    mosaic::Delta first;
    first.opcode = mosaic::OpCode::AssertFact;
    first.subject = "user";
    first.predicate = "current_project";
    first.value = "game";
    first.provenance = "user_message";
    first.turn = 1;
    const auto a = state.apply(first);
    if (!a.ok) {
        std::cerr << a.message << '\n';
        return 1;
    }
    std::cout << "Stored fact #" << a.created_id << ": user current_project = game\n";

    mosaic::Delta correction;
    correction.opcode = mosaic::OpCode::CorrectFact;
    correction.fact_id = a.created_id;
    correction.subject = "user";
    correction.predicate = "current_project";
    correction.value = "chess_engine";
    correction.provenance = "user_message";
    correction.note = "user explicitly corrected the earlier project";
    correction.turn = 2;
    const auto b = state.apply(correction);
    if (!b.ok) {
        std::cerr << b.message << '\n';
        return 1;
    }
    std::cout << "Applied correction; old fact #" << a.created_id
              << " superseded by #" << b.created_id << '\n';

    const auto answers = state.query("user", "current_project");
    std::cout << "Current answer candidates:\n";
    for (const auto& fact : answers) {
        std::cout << "  " << fact.value << " (turn " << fact.turn
                  << ", provenance " << fact.provenance << ")\n";
    }

    const auto history = state.query("user", "current_project", true);
    std::cout << "Historical records preserved: " << history.size() << '\n';
    return answers.size() == 1 && answers.front().value == "chess_engine" &&
           history.size() == 2 ? 0 : 2;
}
