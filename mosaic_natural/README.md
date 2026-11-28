# MOSAIC / MCR — Natural-language dialogue generator

This branch replaces retrieval-only chat and intent-only routing with a from-scratch, open-ended conversational language generator. It is trained to continue multi-turn dialogue and compose new assistant replies conditioned on the recent conversation.

## Model architecture

- A two-layer GRU recurrent language model.
- 320-dimensional recurrent state (configurable).
- A learned lexical/subword vocabulary trained only on the selected training conversations.
- UTF-8 byte fallback for unseen names, words and symbols.
- Tied input/output embedding matrix.
- Assistant-turn next-token objective conditioned on previous user and assistant turns.
- Random initialization: no pretrained LLM, pretrained embeddings, or downloaded tokenizer.
- Symmetric int8 weight export.
- Native C++17 autoregressive inference with recurrent state caching, temperature, top-k/top-p sampling and repetition penalty.
- Conversation transcript persists locally between runs.

The recurrent generator is intentionally smaller and more CPU-friendly than a conventional large transformer. It is a real text generator: replies are composed token-by-token by learned weights rather than selected from a stored response bank.

## Training corpora

The Kaggle trainer attempts to load:
- UltraChat 200k (HuggingFaceH4/ultrachat_200k): MIT; multi-turn user/assistant conversations.
- OpenAssistant OASST1 (OpenAssistant/oasst1): Apache-2.0; the loader reconstructs each assistant message's parent-linked conversation path.
- DailyDialog (roskoN/dailydialog): CC BY-NC-SA 4.0; use and redistribution are subject to non-commercial/share-alike terms.

The report records exactly which datasets loaded and how many conversations were used. The training script refuses to silently continue with zero examples.

## One-time Kaggle bake

1. Select GPU T4 x2 and enable Internet.
2. Run MOSAIC_NATURAL_CHAT_ONE_CELL.py as a single notebook cell.
3. The cell loads and normalizes the corpora, learns the vocabulary from training dialogues, trains the GRU from random initialization using two-process NCCL DDP, evaluates held-out validation/test perplexity, exports an int8 model, compiles the C++ runtime, tests it, packages everything, and opens an in-notebook chat UI.
4. Save a Kaggle notebook version to preserve the output.

Default training settings:
- Vocabulary size: 8,192 including reserved byte/special IDs.
- Hidden width: 320.
- GRU layers: 2.
- Maximum context: 352 tokens.
- Maximum response target: 128 tokens.
- Training examples: up to 250,000 assistant turns.
- Epochs: 3.
- Per-GPU batch: 24.

These are defaults, not quality guarantees. Use the report's measured validation/test loss and generated examples to decide whether additional training is useful.

## Native CPU build and run

C++17 and CMake 3.16+:

    cmake -S mosaic_natural/cpp -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel 2
    ctest --test-dir build --output-on-failure

Run Linux:

    ./build/mosaic_natural_chat --model mosaic_dialogue.mos --history mosaic_history.tsv

Run Windows:

    .\build\Release\mosaic_natural_chat.exe --model mosaic_dialogue.mos --history mosaic_history.tsv

The C++ runtime uses only the exported int8 model. It requires no Python, CUDA, PyTorch, or remote service. Build on the target computer for maximum CPU compatibility.

## Benchmark

    ./build/mosaic_natural_bench --model mosaic_dialogue.mos --iterations 30

Record the exact CPU model, compiler and build options together with p50/p95 generation times. Kaggle GPU speed is not a CPU inference benchmark.

## Scope and evaluation

This model is focused on conversational language: greetings, social turn-taking, follow-up, casual replies and multi-turn continuation. It is not trained as a general knowledge engine and should not be expected to reliably answer factual questions outside its dialogue training distribution.

The training report includes dataset counts, train/validation/test token loss and perplexity, parameter count, quantized artifact size and sample generations. Perplexity measures prediction of held-out text; it does not prove human-level conversational ability. Manual conversation tests are still required.

DailyDialog's CC BY-NC-SA terms impose non-commercial/share-alike conditions. The report preserves this provenance so the resulting model is not silently represented as unrestricted for commercial redistribution.
