#include "mosaic/agent.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

int main() {
    mosaic::Agent agent;

    const auto greeting = agent.respond("Hi");
    assert(greeting.find("Hi!") == 0);
    assert(greeting.find("102 boxes") == std::string::npos);

    const auto introduction = agent.respond("My name is Alex.");
    assert(introduction.find("Alex") != std::string::npos);
    assert(agent.respond("What is my name?").find("Alex") != std::string::npos);

    agent.respond("I am building a game.");
    assert(agent.respond("What am I building?").find("a game") != std::string::npos);

    agent.respond("Actually, I am building a chess engine.");
    const auto current = agent.respond("What is my project?");
    assert(current.find("chess engine") != std::string::npos);
    const auto project_history = agent.state().query("user", "current_project", true);
    assert(project_history.size() == 2);
    assert(project_history.front().active);
    assert(project_history.back().active == false);

    assert(agent.respond("What is 173 times 29?").find("5017") != std::string::npos);
    assert(agent.respond("How are you?").find("ready") != std::string::npos);

    const std::string memory_path = "mosaic_agent_test_memory.tsv";
    std::string error;
    assert(agent.save_memory(memory_path, &error));
    mosaic::Agent restored;
    assert(restored.load_memory(memory_path, &error));
    assert(restored.respond("What is my name?").find("Alex") != std::string::npos);
    assert(restored.respond("What am I building?").find("chess engine") != std::string::npos);
    std::remove(memory_path.c_str());

    std::cout << "MOSAIC dialogue-agent tests passed\n";
    return 0;
}
