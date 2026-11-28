# MOSAIC state kernel — native proof of concept

This is the first executable fragment of the MOSAIC meaning-delta architecture.

## What it proves

The C++17 kernel stores provenance-aware assertions and applies explicit state deltas:
- ASSERT adds a fact.
- CORRECT deactivates an old fact, links it to the replacement, and preserves both in history.
- RETRACT deactivates a fact without deleting its record.
- SET_GOAL / CLEAR_GOAL maintain a persistent active-goal list.
- Queries use a subject/predicate index and return active facts or full history.

The kernel deliberately preserves conflicts from different sources. It does not silently decide that the newest statement is always true.

## Build and test

\`\`\`bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
./build/mosaic_demo
\`\`\`

With MinGW directly:

\`\`\`bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/demo.cpp -o mosaic_demo
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude tests/state_kernel_test.cpp -o mosaic_kernel_test
./mosaic_kernel_test
\`\`\`

## Scope boundary

This is a state-management kernel, not a natural-language parser or chatbot. Inputs are currently constructed as typed deltas by a caller. The next research component is the Meaning-Delta Compiler that learns to produce these deltas from language. No natural-language intelligence or CPU speed claim follows from this kernel alone.
