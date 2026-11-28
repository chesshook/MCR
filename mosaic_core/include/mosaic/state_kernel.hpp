#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mosaic {

enum class EpistemicStatus : std::uint8_t {
    Asserted = 0,
    Inferred = 1,
    Hypothetical = 2,
    Quoted = 3,
    Denied = 4
};

enum class OpCode : std::uint8_t {
    AssertFact = 0,
    RetractFact = 1,
    CorrectFact = 2,
    SetGoal = 3,
    ClearGoal = 4
};

struct Fact {
    std::uint64_t id = 0;
    std::string subject;
    std::string predicate;
    std::string value;
    std::string provenance;
    std::string note;
    std::uint64_t turn = 0;
    std::uint32_t confidence_q16 = 65535;
    EpistemicStatus epistemic = EpistemicStatus::Asserted;
    bool active = true;
    std::uint64_t superseded_by = 0;
};

struct Goal {
    std::uint64_t id = 0;
    std::string owner;
    std::string description;
    std::string provenance;
    std::uint64_t turn = 0;
    bool active = true;
};

struct Delta {
    OpCode opcode = OpCode::AssertFact;
    std::string subject;
    std::string predicate;
    std::string value;
    std::string provenance;
    std::string note;
    std::uint64_t fact_id = 0;
    std::uint64_t turn = 0;
    std::uint32_t confidence_q16 = 65535;
    EpistemicStatus epistemic = EpistemicStatus::Asserted;
};

struct ApplyResult {
    bool ok = false;
    std::string message;
    std::uint64_t created_id = 0;
    std::uint64_t affected_id = 0;
};

class StateKernel {
public:
    ApplyResult apply(const Delta& delta) {
        switch (delta.opcode) {
            case OpCode::AssertFact:
                return assert_fact(delta);
            case OpCode::RetractFact:
                return retract_fact(delta.fact_id, delta.note);
            case OpCode::CorrectFact:
                return correct_fact(delta);
            case OpCode::SetGoal:
                return set_goal(delta);
            case OpCode::ClearGoal:
                return clear_goal(delta.fact_id, delta.note);
        }
        return {false, "unknown opcode", 0, 0};
    }

    std::vector<Fact> query(const std::string& subject,
                            const std::string& predicate,
                            bool include_history = false) const {
        const auto key = make_key(subject, predicate);
        const auto found = index_.find(key);
        if (found == index_.end()) return {};
        std::vector<Fact> result;
        result.reserve(found->second.size());
        for (const auto position : found->second) {
            const auto& fact = facts_.at(position);
            if (include_history || fact.active) result.push_back(fact);
        }
        std::stable_sort(result.begin(), result.end(),
            [](const Fact& a, const Fact& b) {
                if (a.turn != b.turn) return a.turn > b.turn;
                return a.id > b.id;
            });
        return result;
    }

    std::optional<Fact> find_fact(std::uint64_t id) const {
        const auto found = by_id_.find(id);
        if (found == by_id_.end()) return std::nullopt;
        return facts_.at(found->second);
    }

    std::vector<Goal> active_goals(const std::string& owner = {}) const {
        std::vector<Goal> result;
        for (const auto& goal : goals_) {
            if (goal.active && (owner.empty() || goal.owner == owner)) result.push_back(goal);
        }
        std::stable_sort(result.begin(), result.end(),
            [](const Goal& a, const Goal& b) {
                if (a.turn != b.turn) return a.turn > b.turn;
                return a.id > b.id;
            });
        return result;
    }

    const std::vector<Fact>& fact_log() const noexcept { return facts_; }
    const std::vector<Goal>& goal_log() const noexcept { return goals_; }

    std::size_t active_fact_count() const noexcept {
        return static_cast<std::size_t>(std::count_if(
            facts_.begin(), facts_.end(),
            [](const Fact& fact) { return fact.active; }));
    }

private:
    static std::string make_key(const std::string& subject,
                                const std::string& predicate) {
        std::string key;
        key.reserve(subject.size() + predicate.size() + 1);
        key.append(subject);
        key.push_back('\x1f');
        key.append(predicate);
        return key;
    }

    ApplyResult assert_fact(const Delta& delta) {
        if (delta.subject.empty() || delta.predicate.empty() || delta.value.empty()) {
            return {false, "ASSERT requires subject, predicate and value", 0, 0};
        }
        if (delta.confidence_q16 > 65535U) {
            return {false, "confidence must be in Q0.16 range [0,65535]", 0, 0};
        }
        Fact fact;
        fact.id = next_id_++;
        fact.subject = delta.subject;
        fact.predicate = delta.predicate;
        fact.value = delta.value;
        fact.provenance = delta.provenance;
        fact.note = delta.note;
        fact.turn = delta.turn;
        fact.confidence_q16 = delta.confidence_q16;
        fact.epistemic = delta.epistemic;
        const auto position = facts_.size();
        facts_.push_back(std::move(fact));
        by_id_.emplace(facts_.back().id, position);
        index_[make_key(delta.subject, delta.predicate)].push_back(position);
        return {true, "fact asserted", facts_.back().id, facts_.back().id};
    }

    ApplyResult retract_fact(std::uint64_t id, const std::string& reason) {
        const auto found = by_id_.find(id);
        if (found == by_id_.end()) return {false, "fact id not found", 0, id};
        auto& fact = facts_.at(found->second);
        if (!fact.active) return {false, "fact is already inactive", 0, id};
        fact.active = false;
        fact.note = reason.empty() ? "retracted" : reason;
        return {true, "fact retracted; history preserved", 0, id};
    }

    ApplyResult correct_fact(const Delta& delta) {
        const auto found = by_id_.find(delta.fact_id);
        if (found == by_id_.end()) return {false, "fact to correct was not found", 0, delta.fact_id};
        const auto& old = facts_.at(found->second);
        if (!old.active) return {false, "fact to correct is already inactive", 0, delta.fact_id};
        if (delta.subject.empty() || delta.predicate.empty() || delta.value.empty()) {
            return {false, "CORRECT requires replacement subject, predicate and value", 0, delta.fact_id};
        }
        if (delta.confidence_q16 > 65535U) {
            return {false, "confidence must be in Q0.16 range [0,65535]", 0, delta.fact_id};
        }

        // Append the replacement first. Only after successful insertion do we
        // deactivate the old assertion, keeping correction atomic at this level.
        const auto inserted = assert_fact(delta);
        if (!inserted.ok) return inserted;
        auto& previous = facts_.at(found->second);
        previous.active = false;
        previous.superseded_by = inserted.created_id;
        previous.note = delta.note.empty() ? "superseded by explicit correction" : delta.note;
        return {true, "correction committed; prior fact retained in history",
                inserted.created_id, delta.fact_id};
    }

    ApplyResult set_goal(const Delta& delta) {
        if (delta.subject.empty() || delta.value.empty()) {
            return {false, "SET_GOAL requires owner (subject) and description (value)", 0, 0};
        }
        Goal goal;
        goal.id = next_id_++;
        goal.owner = delta.subject;
        goal.description = delta.value;
        goal.provenance = delta.provenance;
        goal.turn = delta.turn;
        goals_.push_back(std::move(goal));
        return {true, "goal activated", goals_.back().id, goals_.back().id};
    }

    ApplyResult clear_goal(std::uint64_t id, const std::string& reason) {
        const auto found = std::find_if(goals_.begin(), goals_.end(),
            [id](const Goal& goal) { return goal.id == id; });
        if (found == goals_.end()) return {false, "goal id not found", 0, id};
        if (!found->active) return {false, "goal is already inactive", 0, id};
        found->active = false;
        (void)reason; // Goal audit detail is retained in higher-level episode log.
        return {true, "goal deactivated", 0, id};
    }

    std::uint64_t next_id_ = 1;
    std::vector<Fact> facts_;
    std::vector<Goal> goals_;
    std::unordered_map<std::uint64_t, std::size_t> by_id_;
    std::unordered_map<std::string, std::vector<std::size_t>> index_;
};

}  // namespace mosaic
