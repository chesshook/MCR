#include "mosaic/state_kernel.hpp"

#include <cassert>
#include <iostream>

int main() {
    using namespace mosaic;

    // Assertions are stored with provenance and are queryable.
    StateKernel state;
    Delta first;
    first.opcode = OpCode::AssertFact;
    first.subject = "user";
    first.predicate = "current_project";
    first.value = "game";
    first.provenance = "user";
    first.turn = 1;
    const auto created = state.apply(first);
    assert(created.ok && created.created_id != 0);
    assert(state.query("user", "current_project").size() == 1);

    // An explicit correction changes the active view but does not erase history.
    Delta correction;
    correction.opcode = OpCode::CorrectFact;
    correction.fact_id = created.created_id;
    correction.subject = "user";
    correction.predicate = "current_project";
    correction.value = "chess_engine";
    correction.provenance = "user";
    correction.turn = 2;
    const auto corrected = state.apply(correction);
    assert(corrected.ok);
    const auto current = state.query("user", "current_project");
    assert(current.size() == 1 && current.front().value == "chess_engine");
    const auto history = state.query("user", "current_project", true);
    assert(history.size() == 2);
    const auto old = state.find_fact(created.created_id);
    assert(old && !old->active && old->superseded_by == corrected.created_id);

    // Conflicting assertions from distinct sources are not silently collapsed.
    Delta external = first;
    external.value = "other_project";
    external.provenance = "external_source";
    external.turn = 3;
    assert(state.apply(external).ok);
    assert(state.query("user", "current_project").size() == 2);

    // Retraction is explicit and idempotence errors are surfaced.
    Delta retract;
    retract.opcode = OpCode::RetractFact;
    retract.fact_id = corrected.created_id;
    retract.note = "user withdrew assertion";
    assert(state.apply(retract).ok);
    assert(state.apply(retract).ok == false);

    // Goals are persistent records with an active/inactive lifecycle.
    Delta goal;
    goal.opcode = OpCode::SetGoal;
    goal.subject = "assistant";
    goal.value = "answer user's question";
    goal.provenance = "dialogue_manager";
    goal.turn = 4;
    const auto goal_result = state.apply(goal);
    assert(goal_result.ok && state.active_goals("assistant").size() == 1);
    Delta clear;
    clear.opcode = OpCode::ClearGoal;
    clear.fact_id = goal_result.created_id;
    assert(state.apply(clear).ok);
    assert(state.active_goals("assistant").empty());

    std::cout << "MOSAIC state-kernel tests passed\n";
}
