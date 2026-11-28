# MOSAIC / MCR — Native CPU Chatbot

This branch turns the MOSAIC state kernel into a small, runnable conversational program. The deployed inference path is native C++17. The optional learned intent classifier is trained offline, quantized to int8, and loaded by C++; Python and CUDA are not needed at inference.

## What is implemented

- Native C++17 chat executable and native latency benchmark.
- Deterministic UTF-8 byte hashed word, word-bigram, word-trigram and character n-gram features.
- Optional int8 multiclass intent model with a compact binary format.
- Persistent user facts with provenance and correction history.
- Greeting, farewell, thanks, identity, capability and basic arithmetic responses.
- Fact capture for patterns such as "my name is ..." and "I'm building ...".
- Questions about remembered name and current project.
- Explicit correction handling, e.g. "Actually, I'm building a chess engine."
- Local memory save/load. The chat can continue across process restarts on the same machine.
- C++ and Python tests, including a binary model-format test.

The intent classifier uses CLINC150 plus (150 in-scope intents and an out-of-scope class). It learns from random initialization. No pretrained LLM, tokenizer, word vectors or generative model is used. The dataset is licensed CC BY 3.0 according to its Hugging Face dataset card; preserve attribution and verify the upstream terms before redistribution.

## What is not implemented

This is a real first chatbot build, but it is not a general-purpose human-level AI. The initial release does not yet have broad factual world knowledge, unrestricted compositional parsing, general causal reasoning, unrestricted novel prose generation, or proven consciousness. CLINC intent accuracy is not a measure of general intelligence.

Unsupported questions should receive an uncertainty response rather than a fabricated answer. Only intents with an implemented response path are allowed to produce a task-specific answer.

## Train once on Kaggle T4 x2

1. Create a Kaggle notebook, select GPU T4 x2, and enable Internet.
2. Run the complete MOSAIC_KAGGLE_ONE_CELL.py in one cell.
3. The cell checks the two T4 GPUs, loads clinc/clinc_oos plus, trains an EmbeddingBag linear classifier from random initialization with two-process NCCL DDP, evaluates on the supplied validation and test splits, quantizes the model, builds the C++ program, and exports MCR_MOSAIC_CPU.zip.
4. Save a Kaggle notebook version to preserve the output files.

The two T4s are used for offline baking only. The exported C++ executable does not require CUDA. The classifier is intentionally small: 65,536 hashed features × 151 output classes before int8 quantization. It is an intent classifier, not a language generator.

## Run locally

Requires a C++17 compiler and CMake 3.16+.

    cmake -S mosaic_core -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --parallel 2

Linux:

    ./build/mosaic_chat --model intent_model.mcr --memory mosaic_memory.tsv
    ./build/mosaic_bench --model intent_model.mcr --iterations 10000

Windows:

    cmake -S mosaic_core -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --parallel 2
    .\build\Release\mosaic_chat.exe --model intent_model.mcr --memory mosaic_memory.tsv
    .\build\Release\mosaic_bench.exe --model intent_model.mcr --iterations 10000

The benchmark reports p50 and p95 inference and dialogue-path latency on the machine where it is run. It is single-process and single-threaded, matching the CPU inference target. Record the exact CPU model and compiler when comparing results. No dual-core latency claim is made before measuring on the target computer.

The model argument may be omitted or set to "-" to run without learned intent classification. Facts and corrections are written to the local memory file. Do not share that file if it contains personal information.

## Notebook chat interface

After the Kaggle bake cell finishes, it automatically displays a notebook chat UI. The UI sends messages to the compiled C++ executable; it does not perform model inference in Python.

## Evaluation

Report the following independently:
- CLINC150 held-out intent accuracy and macro-F1.
- C++ binary size.
- Model file size and resident memory.
- p50/p95 latency on a named CPU.
- Multi-turn exact fact recall and correction accuracy.
- Greeting / question / unsupported-request test results.

Do not call the agent human-like or general intelligence based only on intent accuracy or a successful build.
