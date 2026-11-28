# MCR CPU-Native V2 — first runtime component

This directory contains **Phase 1's intent-classification prototype**, not a complete chatbot.

## Properties

- Native C++17 inference; CPU only.
- No PyTorch, CUDA, tokenizer download, pretrained model, or Python runtime is required by the compiled executable.
- Sparse hashed word, character 3–5-gram, punctuation and adjacent-token features.
- Offline trainer uses a multiclass averaged sparse perceptron and Python standard library only.
- A validation-calibrated margin gate can return \`__uncertain__\` instead of confidently guessing.
- Runtime reads one input line and prints an intent label, raw score, margin and runner-up. Scores are **not probabilities**.

This is in the fastText family of ideas (compact n-gram feature learning + linear classification), but is not fastText itself.

## Build on Windows (MSVC)

From \`cpu_native\`:

\`\`\`powershell
cmake -S . -B build -A x64
cmake --build build --config Release
\`\`\`

Or with MinGW:

\`\`\`bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic src/main.cpp -o mcr_intent
\`\`\`

## Prepare the benchmark data

Run in a training environment with internet:

\`\`\`bash
python prepare_clinc.py
\`\`\`

This downloads CLINC150/OOS-Eval and creates:
- \`data/clinc150/train.tsv\`
- \`data/clinc150/valid.tsv\`
- \`data/clinc150/test.tsv\`
- \`data/clinc150/manifest.json\`

The CLINC150 task has 150 in-scope intents plus out-of-scope examples. It is a single-intent classifier benchmark, **not an open-domain chat corpus**.

## Train and evaluate

\`\`\`bash
python train_intent.py \
  --train data/clinc150/train.tsv \
  --valid data/clinc150/valid.tsv \
  --output artifacts/intent_model.txt \
  --epochs 12
\`\`\`

The validation split selects an abstention margin to meet the requested selective accuracy when possible. Inspect both validation accuracy and coverage: a model that only answers a tiny fraction of queries is not useful.

Evaluate once on the untouched test split; do not tune on test:

\`\`\`bash
python evaluate_intent.py --model artifacts/intent_model.txt --test data/clinc150/test.tsv
\`\`\`

Run the feature-unit tests:

\`\`\`bash
python -m unittest discover -s tests -v
\`\`\`

## Run the C++ module

\`\`\`bash
./mcr_intent artifacts/intent_model.txt
\`\`\`

Enter one sentence per line. The output is an intent hypothesis. It deliberately does not fabricate a natural-language answer.

## Known limitations

- English-only first pass.
- Character feature hashing is byte-based UTF-8; it is not Unicode linguistic analysis.
- This first model predicts one primary intent; mixed-intent utterances need a later multi-label layer.
- CLINC150 is narrow and cannot establish general conversation ability.
- No sentence parser, semantic frame builder, memory manager, reasoning engine or response generator is implemented here yet.
