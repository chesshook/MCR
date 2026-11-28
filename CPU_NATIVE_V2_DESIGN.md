# MCR CPU-Native V2 — Engineering Design

## Objective

Build a meaning-first conversational system whose **inference runtime is native C++17**, CPU-only, dependency-light, and suitable for a low-end dual-core computer. All expensive corpus preparation and model fitting happen offline (Kaggle or another training machine); deployment receives compact learned tables, lexicons, and rules.

This is a research target, not a claim that a tiny model can already match a large language model. "Human-like" conversation is an evaluation goal, not a property implied by training volume or a benchmark score.

## Hard constraints

- No pretrained LLM, transformer, remote API, cloud inference, or downloaded generative model.
- No GPU requirement at inference.
- C++17 runtime; no Python, PyTorch, CUDA, or large runtime dependency after export.
- Maximum two inference worker threads; single-thread fallback.
- Keep each subsystem inspectable and testable. Unknown words and unresolved parses remain explicit; do not silently invent certainty.
- Preserve the completed `math-native-v1` checkpoint as an experimental baseline. It is not the production CPU runtime.
- All dataset sources, licenses, versions, and transformations must be recorded in a machine-readable manifest.

## Important correction about fastText

fastText is a useful design reference: bagged word/character n-gram features, compact learned representations, and a linear classifier can be extremely fast. Its published results include training on over one billion words in under ten minutes on a multicore CPU. That result concerns **text classification**, not general conversation, compositional reasoning, or free-form response generation. MCR will use this family of sparse/shallow methods for bounded sub-tasks rather than treating an intent classifier as the whole brain.

Reference: Joulin et al., *Bag of Tricks for Efficient Text Classification*, EACL 2017, https://aclanthology.org/E17-2068/

## Proposed architecture

### A. Input and lexical analysis

1. UTF-8 validation and normalization.
2. Fast tokenizer with punctuation, contractions, numbers, and sentence boundaries preserved.
3. Character n-grams and conservative morphology for misspellings and inflections.
4. Compact lexical database with candidate senses and typed relations.
5. Unknown words remain first-class symbols with surface form and context.

Candidate lexical source: Open English WordNet 2025, CC BY 4.0. Store the version and attribution beside the compact export. https://en-word.net/downloads

### B. Lightweight learned language modules

Use sparse/hashed features, averaged perceptrons, linear classifiers, and finite-state models where appropriate. Train offline; export quantized or compact arrays.

Separate models, rather than one opaque score, for:
- communicative act: greeting, farewell, thanks, apology, question, request, correction, disagreement, explanation request, etc.;
- task intent and out-of-domain detection;
- token POS and dependency transitions;
- entity/slot extraction (names, quantities, dates, locations, references);
- dialogue-turn function and response strategy.

A score is not automatically a probability. Calibrate thresholds on held-out data and support abstention.

### C. Compositional semantic representation

Convert parsed clauses into an explicit graph/frame representation:

- predicates/events;
- participants and semantic roles (agent, patient/theme, recipient, instrument, location);
- polarity and negation scope;
- tense, aspect, modality and temporal anchors;
- quantities and units;
- coreference links and entity identity;
- causal, conditional, comparative, and contrast relations;
- unresolved/ambiguous alternatives with evidence.

Do not assume WordNet alone supplies sentence meaning. Lexical relations are evidence; grammar and discourse determine composition.

### D. Dialogue state and memory

Keep a bounded, inspectable dialogue state:
- active topic and current user request;
- entities introduced and their attributes;
- unresolved questions and commitments;
- corrections (new values supersede old values);
- short-term turn history and salience;
- explicit user-taught facts separated from inferred facts.

A pronoun or follow-up must resolve against this state with confidence and an explicit unknown outcome when ambiguous.

### E. Reasoning

Begin with deterministic, auditable operations:
- arithmetic and unit conversion;
- comparisons and constraints;
- temporal ordering;
- negation and contradiction checks;
- simple causal/conditional graph traversal;
- entity/coreference consistency;
- question decomposition and missing-information detection.

Only claim a reasoning skill when it passes a dedicated held-out test set. Do not equate NLI classification accuracy with general reasoning.

### F. Response planning and native realization

Select a communicative act first, then construct a meaning plan:
- answer directly;
- ask one targeted clarification;
- acknowledge/correct;
- explain with evidence;
- decline to guess when evidence is inadequate.

Realize the plan through a compact grammar and morphology engine, with controlled phrase variation. A bounded n-gram model may help rank word sequences, but must not be allowed to replace the semantic plan or create unsupported facts. Retrieval may be used as supporting examples, never as unconditional nearest-neighbour reply selection.

## Training data plan

Use task-labelled data for each task; raw word count alone does not teach dialogue policy or compositional semantics.

| Task | Candidate source | Use and license note |
|---|---|---|
| Intent + out-of-scope | CLINC150 | CC BY 4.0; 150 intents and out-of-scope examples. |
| Multi-domain dialogue acts/state/slots | Schema-Guided Dialogue | CC BY-SA 4.0; task-oriented dialogue annotations, not open-domain small talk. |
| Everyday dialogue acts/emotions | DailyDialog | CC BY-NC-SA 4.0; non-commercial/share-alike restrictions. Load from a maintained data mirror, not the retired dataset script. |
| Syntax | Universal Dependencies English EWT, optionally GUM | Verify treebank annotation and underlying-text terms separately; EWT annotations are CC BY-SA 4.0. |
| Lexical graph | Open English WordNet 2025 | CC BY 4.0; preserve attribution. |
| Semantic relations | SNLI/MultiNLI only as one narrow auxiliary task | Record each source's terms; do not use as a proxy for full understanding. |

Do not combine licenses into one blanket claim. The manifest must retain source-level provenance and usage restrictions. Dataset packaging and source terms must be rechecked at download time.

## Evaluation protocol (required before calling it a chatbot)

1. Split dialogue corpora by complete conversation, never by individual context/reply pair.
2. Keep a fixed, untouched test set.
3. Test greeting/opening, identity, follow-up, correction, pronoun/coreference, paraphrase, negation, ambiguity, topic shift, explanation, refusal/unknown, and multi-turn consistency.
4. Evaluate intent macro-F1 and out-of-scope AUROC/FPR; POS accuracy; dependency UAS/LAS; slot F1; dialogue-state accuracy; semantic frame role F1; response relevance and contradiction rate.
5. Human-review a stratified set of full conversations. Report failures, not just aggregate scores.
6. Compare against trivial baselines (keyword rules, TF-IDF retrieval, fastText-style classifier) to prove each added subsystem helps.

## Runtime budget (targets to measure, not promises)

- C++17, CPU-only; no AVX requirement for correctness.
- 1–2 threads; bounded input and memory.
- Target model + lexical assets: under 128 MiB initially, with a documented path to a larger optional lexicon.
- Target p95 response latency: under 100 ms for a typical short turn on a specified dual-core reference CPU. Measure cold start separately.
- Every release records CPU model, compiler, build flags, model/lexicon sizes, p50/p95 latency, and peak resident memory.

"Any dual-core CPU" is not a measurable hardware specification. A 2-core modern laptop CPU and an old low-clock desktop CPU differ substantially; publish benchmark hardware before promising speed.

## Implementation phases

1. **CPU runtime foundation:** tokenizer, feature hashing, sparse classifier loader, calibration/abstention, deterministic unit tests and benchmark harness.
2. **Conversation data repair:** reliable DailyDialog/SGD/CLINC loading, conversation-level splits, source/license manifest, label audits.
3. **Meaning layer:** lexical graph + syntax + event/role frames; held-out parser and frame evaluation.
4. **Dialogue state:** coreference, corrections, topic tracking, unresolved references.
5. **Response planner + surface realizer:** meaning-first answer planning and native grammar realization.
6. **Integrated training/evaluation:** task-specific compact models, quantization, C++ parity tests, dual-core benchmarks.

A phase is complete only when its tests and measured metrics are published. Do not label the early intent classifier as a general chatbot.

## Relationship to existing MCR

- `math-native-v1` remains the record of the first from-scratch neural encoder experiment.
- The 28.3M-parameter byte CNN/GRU and 341 MiB retrieval index are not required by this CPU-native runtime.
- Reuse inspectable lexical/compositional ideas from NAV where appropriate, but re-evaluate old parser accuracy before shipping. The previous NAV parser's reported held-out UAS/LAS were not adequate for dependable open-domain semantics.
- Do not merge this branch into `main` or overwrite the existing baseline until the new branch passes tests and the user explicitly requests integration.
