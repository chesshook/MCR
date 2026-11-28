# MOSAIC: Meaning-Oriented Sparse Activation and Incremental Cognition

## Research proposal for a CPU-native MCR cognitive architecture

**Status:** proposed architecture; hypotheses require implementation and falsifiable experiments.

MOSAIC is a proposed alternative to autoregressive language-model inference. It treats conversation as incremental state estimation and controlled state transformation. A message is not primarily a sequence from which the next token must be predicted. It is evidence that may require a small change to an explicit, persistent model of the current situation.

The target is a useful conversational agent whose deployment runtime can execute on an old dual-core CPU with bounded memory and work per turn. Training, corpus compilation and program discovery may be expensive offline. The runtime is intended to use sparse lookup, integer scoring, compact graph traversal and small bounded searches—not dense token-by-token transformer inference.

This design does **not** establish that the system would be phenomenally conscious. It proposes functional mechanisms for continuity, selective attention, metacognition, global availability of a current thought, self-modeling and goal-directed dialogue. Those mechanisms are testable engineering properties; subjective experience remains a separate scientific question.

---

## 1. The central idea: compute changes, not entire sequences

A conversation is a changing state.

Let the active cognitive state at turn \(t\) be:

\[
S_t = (W_t, D_t, G_t, M_t, C_t)
\]

where:

- \(W_t\): working model of the world and the current situation;
- \(D_t\): dialogue state (speaker, active topic, unresolved requests, references);
- \(G_t\): current goals and constraints;
- \(M_t\): episodic and semantic memory indexes;
- \(C_t\): confidence, evidence, provenance and unresolved ambiguity.

A new message \(u_t\) is converted to one or more candidate **meaning deltas**:

\[
\Delta_t^{(i)} = \operatorname{Interpret}(u_t,S_t)
\]

Each delta is a small typed set of operations, not a replacement of the whole state:

- ASSERT(entity, property, value, provenance)
- RETRACT(fact_id, reason)
- CORRECT(old_fact_id, new_fact)
- ASK(target, requested_relation)
- REFER(expression, candidate_entity)
- GOAL(action, object, constraints)
- NEGATE(scope)
- CAUSE(cause_event, result_event)
- TIME(event, interval, relation)
- COMPARE(left, right, dimension)
- COMMUNICATIVE_ACT(type, target)

Candidate deltas are checked against syntax, dialogue history, semantic constraints, provenance and the current goal. The chosen delta is applied to the state. Uncertain alternatives remain explicit.

\[
S_{t+1} = \operatorname{Commit}(S_t,\Delta_t^*)
\]

This is **state-difference cognition**: most turns should touch only a small portion of the memory graph.

### Example: learning and correcting a fact

Conversation:

1. "I am building a game."
2. "Actually, I changed my mind. I am building a chess engine."
3. "What am I building now?"

The first message creates:

\`ASSERT(user, current_project, game, source=user, turn=1)\`

The correction creates:

\`RETRACT(fact_1, reason=explicit_correction)\`
\`ASSERT(user, current_project, chess_engine, source=user, turn=2)\`
\`CORRECT(fact_1, fact_2)\`

The question creates:

\`ASK(subject=user, relation=current_project)\`

The answer is obtained by reading the latest applicable fact with user-provided provenance. No approximate reply search is needed.

### Example: a greeting

"Hi" is identified as a greeting speech act. It proposes a dialogue action:

\`COMMUNICATIVE_ACT(GREETING, recipient=assistant)\`

The response planner chooses a greeting response policy. It does not search a 116,000-entry store and risk returning a mathematics worksheet.

---

## 2. The three new mechanisms

### Mechanism A — Meaning-Delta Compiler (MDC)

The MDC maps raw language into compact, typed state-change programs.

#### A1. Evidence-preserving lexical stream

Input is normalized into token records:

- surface span and byte offsets;
- normalized spelling and morphological candidates;
- punctuation and sentence boundary;
- lexical candidate senses;
- POS candidates;
- local syntactic features;
- source location in the original utterance.

Unknown words remain symbols. The compiler must never replace an unknown name, product, acronym or invented word with a guessed known word.

A lexical graph proposes senses and relations; grammar and context decide which proposals are plausible. Dictionary similarity alone is not treated as sentence understanding.

#### A2. Bounded incremental parse

Instead of running an expensive global parser without a budget, the MDC maintains a small set of structural hypotheses. It processes tokens left-to-right and revises only local structures when new evidence demands it.

Examples of distinctions that must survive compilation:

- "Alice hit Bob" versus "Bob hit Alice";
- "not all birds fly" versus "all birds do not fly";
- "I saw the man with a telescope" (attachment ambiguity);
- "John told Mark that he won" (reference ambiguity);
- "I used to work there" versus "I am working there."

The compiler emits alternatives and their evidence rather than forcing an unsupported parse.

#### A3. Typed event frames

A clause becomes one or more frames:

\`EVENT(id, predicate, tense, aspect, polarity, modality)\`

\`ROLE(event, role_type, filler, evidence)\`

Roles can include agent, patient/theme, recipient, experiencer, instrument, source, destination, location, time and manner.

Relations between events include before, after, overlaps, causes, enables, prevents, conditionally-implies, contrasts-with and refers-to.

The graph stores provenance for each assertion: speaker, turn, source type, confidence, and whether it was asserted, quoted, hypothetical, denied or inferred.

This prevents a quoted lie, a hypothetical plan, and a fact asserted by the user from collapsing into one undifferentiated sentence embedding.

---

### Mechanism B — Executable Knowledge Compiler (EKC)

This is the most important learning proposal.

A conventional small classifier learns a mapping from features to labels. A response retriever learns which stored answer resembles a query. MOSAIC instead attempts to learn **reusable transition programs** from examples of language, dialogue and state changes.

A microprogram has:

1. a typed precondition pattern;
2. input bindings;
3. a state transformation;
4. expected consequences;
5. a response-plan family;
6. evidence and confidence metadata;
7. a cost estimate.

Example microprogram:

\`WHEN speech_act = GREETING\`
\`AND recipient = assistant\`
\`THEN propose response_plan = RECIPROCATE_GREETING\`

A more general state update:

\`WHEN user asserts PROPERTY(subject, key, value)\`
\`THEN add FACT(subject, key, value, source=user)\`

An explicit correction is not merely a new similar utterance:

\`WHEN user marks assertion A as corrected by B\`
\`THEN deactivate A for current-state queries\`
\`AND preserve A in historical memory\`
\`AND activate B as latest user assertion\`

#### B1. Learning the microprograms

Offline training data must contain more than raw text. Each training episode should ideally have:

- prior dialogue state;
- new utterance;
- gold meaning delta;
- resulting state;
- acceptable response plans;
- negative or contrastive examples.

For example, paired examples that differ only in negation, participant order, time, modality or correction marker teach which state transition must change.

A candidate program is induced by unifying multiple examples into a typed pattern. Its parameters become slots. A candidate is retained only when it covers multiple examples and survives contrastive tests.

#### B2. Minimum-description-length objective

The compiler should prefer a small, reusable program set rather than memorizing every example.

\[
J(P) = L(P) + \lambda L(E \mid P) + \mu R(P)
\]

where:

- \(P\) is the selected microprogram library;
- \(L(P)\) is the encoded size of the program library;
- \(L(E \mid P)\) is the description length of examples not explained by the library;
- \(R(P)\) penalizes contradictions, unsupported effects and poor held-out generalization.

A program that memorizes one sentence is expensive. A program that correctly explains hundreds of structurally related state changes is valuable.

The objective does not magically guarantee intelligence. It is a concrete way to trade compactness against explanatory coverage and error.

#### B3. Negative evidence and counterfactual learning

For each learned program, train contrastive tests:

- swap the agent and patient;
- insert or remove negation;
- change past to future;
- change "might" to "will";
- replace a correction with a hypothetical;
- change the referent;
- remove context required to resolve a pronoun.

A valid program must change only the expected parts of the meaning delta. It must not silently change unrelated facts.

#### B4. Compilation for the runtime

The offline learner writes a compact binary program image:
- opcode table;
- typed slot descriptions;
- decision/trie indexes for preconditions;
- fixed-point weights;
- semantic relation IDs;
- response-plan IDs;
- provenance and version manifest.

No Python runtime or neural framework is required to execute the image.

---

### Mechanism C — Sparse Cognitive Scheduler (SCS)

The scheduler decides what deserves computation.

At each turn, the system generates a small number of candidate events and queries. A scheduler estimates priority:

\[
priority(x) =
\frac{relevance(x)\cdot uncertainty(x)\cdot expected\_information\_gain(x)}
{estimated\_cost(x)+\epsilon}
\]

It activates only the highest-value operators within a fixed per-turn budget.

Examples:

- Greeting: activate greeting/social policy only.
- "What is 12 times 17?": activate quantity parsing and exact arithmetic.
- "What was the project I mentioned earlier?": activate entity/coreference lookup and dialogue memory.
- "Why did the engine stop?": activate causal explanation retrieval; if no evidence exists, ask for logs or context.
- "Compare these two options": activate property extraction and comparison.

The scheduler uses a bounded execution budget, not unbounded chain-of-thought generation.

#### C1. Bounded hypothesis competition

Maintain at most \(K\) interpretations (initial default \(K=3\)) for an ambiguous input. Score each using:

- grammar validity;
- lexical support;
- agreement with dialogue state;
- causal and temporal consistency;
- source/provenance consistency;
- task-goal fit;
- contradiction penalties.

If the best candidates remain too close, ask one clarification rather than inventing a fact.

#### C2. Sparse activation and stopping

The engine stops when one of these occurs:

- the current goal is satisfied;
- required information is missing;
- candidate interpretations remain ambiguous;
- a safety or consistency constraint blocks action;
- the bounded cost budget is reached.

This makes computation proportional to activated structures, not to the size of all stored knowledge.

---

## 3. A compact semantic memory that can be searched cheaply

Use three distinct stores.

### 3.1 Current-state graph

A small graph of active entities, properties, events, goals and unresolved references. Store adjacency in compressed sparse-row (CSR) arrays or compact sorted edge arrays.

Each edge has:
- source ID;
- relation ID;
- target/value ID;
- source/provenance ID;
- time/version;
- confidence/epistemic status.

Current state should be small and mutable. Old values move to history rather than being repeatedly scanned.

### 3.2 Episodic memory

Store compact episodes:
- situation signature;
- state before;
- delta;
- action/response plan;
- outcome or correction;
- time and salience.

Index episodes by typed symbols and relation patterns, not solely by a dense vector nearest-neighbor search. Retrieval returns candidate episodes; exact graph constraints verify relevance.

### 3.3 Semantic memory

Store lexical and conceptual relations separately from user-specific facts. This prevents a dictionary definition from overriding what a user has just told the system about their own project.

### 3.4 Optional binary semantic signatures

An optional 256–2048-bit signature can act as a cheap fuzzy prefilter for structurally similar episodes. Role-specific permutations/binding and XOR or bitwise operations can encode rough relationships; Hamming distance and POPCNT can find candidate matches.

Crucially, a binary signature is **not** the authoritative meaning. It only suggests candidates. A typed graph and exact constraints verify them. This borrows from Vector Symbolic Architectures / Hyperdimensional Computing, an existing research family, rather than claiming those algebraic operations are new.

---

## 4. The "conscious-like" functional layer

The user objective mentions consciousness. Separate two questions:

1. Can the system maintain a coherent, continuous, self-referential cognitive process?
2. Does it have subjective experience—something it feels like to be that system?

The architecture can attempt the first. The second is not established by passing behavioural tests or by using a global-workspace-like design.

MOSAIC's functional continuity layer has five mechanisms.

### 4.1 Persistent self-model

A structured record of:
- system identity and current operating mode;
- capabilities and known limitations;
- current goals;
- active commitments;
- previous conclusions;
- uncertainty and unresolved tasks;
- what evidence supports its claims.

It is operational data, not a claim of a soul or inner observer.

### 4.2 Limited global workspace

Only one primary thought/goal plus a few competing candidates occupy the global workspace. Specialized modules can publish:
- a new event;
- a contradiction;
- an important memory;
- a confidence collapse;
- a goal-completion signal.

The workspace broadcasts the selected content to all relevant modules. This is a functional design inspired by global-workspace theories; it is not proof of consciousness.

### 4.3 Metacognitive monitor

Before answering, estimate:
- what is directly known;
- what is inferred;
- what is merely plausible;
- whether evidence conflicts;
- whether the system has enough information;
- whether a tool or clarification is needed.

The monitor can veto an answer and require abstention.

### 4.4 Recurrent self-correction

After response planning, run a short consistency pass:
- Does the answer contradict the current state?
- Did it invert participant roles?
- Did it mistake a hypothetical for a fact?
- Did it ignore a correction?
- Does it answer the actual question?
- Does the answer claim more than the evidence supports?

Only affected frames are revisited.

### 4.5 Memory consolidation

After a conversation, merge repeated stable facts, preserve corrections and provenance, decay irrelevant temporary details, and retain unusual episodes that may be useful later. Consolidation is incremental and can happen between turns or in a low-priority idle phase.

These mechanisms may produce better continuity, access, reportability and self-monitoring. They do not answer the hard problem of phenomenal experience.

---

## 5. Response generation without a pretrained language model

A meaning-first planner should choose a semantic response plan before any surface words are selected.

Examples of response plans:

- \`SOCIAL_GREETING(reciprocal=true, invite_topic=true)\`
- \`ANSWER(value=chess_engine, evidence=user_turn_2)\`
- \`CLARIFY(reference=he, candidates=[John, Mark])\`
- \`EXPLAIN(cause=?, evidence_missing=true, ask_for_logs=true)\`
- \`CORRECT(previous_claim, evidence=source_X)\`

A surface realizer turns the plan into words using:

1. a grammar of clause and phrase constructions;
2. agreement and morphology rules;
3. lexical choice constrained by semantic type;
4. a compact learned phrase-preference model;
5. punctuation and discourse ordering.

The realizer may generate sentences never present in training by combining grammatical constructions and bound semantic slots. This is not guaranteed to produce the breadth or stylistic fluency of a large neural generator. It must be evaluated separately for grammaticality, relevance, factual consistency and diversity.

A small n-gram model can rank local word choices, but must not be the system's reasoning engine or allowed to introduce unsupported entities and facts.

---

## 6. The CPU execution model

The design avoids:
- quadratic self-attention;
- large floating-point matrix multiplication;
- full-vocabulary softmax on every token;
- repeatedly embedding the entire conversation;
- scanning the entire memory store for each turn;
- loading a GPU runtime.

Core inference operations:
- token scan;
- fixed-size feature hashing;
- integer/fixed-point scoring;
- trie and inverted-index lookup;
- typed graph traversal;
- bitset operations and POPCNT;
- bounded rule firing;
- deterministic response construction.

A possible turn budget (initial design hypothesis, not measured performance):

| Resource | Initial limit |
|---|---:|
| Active interpretation candidates | 3 |
| Microprograms considered | 64 |
| Microprogram firings | 128 |
| Semantic graph hops | 4 |
| Coreference candidates | 8 |
| Global workspace items | 8 |
| Response planning passes | 2 |
| Worker threads | 1–2 |

The scheduler can stop early when the current goal is resolved. These are configurable caps; evaluation must show whether they preserve quality.

### Complexity target

For a typical short turn, work should scale approximately with:

\[
O(T + F + A + E_a)
\]

where:
- \(T\): input tokens/characters;
- \(F\): activated feature count;
- \(A\): number of activated microprogram operations;
- \(E_a\): edges traversed in the relevant active subgraph.

This is not a proof of constant-time intelligence. Worst-case memory lookup and ambiguous inputs can cost more, so every index and search must be bounded.

Target deployment budget: under 128 MiB initially and p95 under 100 ms on a named dual-core reference CPU. These are targets to test, not performance claims. Old CPU generations differ enormously; report the exact hardware, compiler and build flags.

---

## 7. Offline training: learn programs, then compile them

A Kaggle T4 x2 or larger training machine can be used for data preparation, parser learning, rule discovery and validation. The deployed executable does not inherit the training runtime.

### Training streams

1. **Lexical stream:** lemma, morphology, senses, synonymy, antonymy, hypernymy, derivation, multiword expressions.
2. **Syntax stream:** tokenization, POS, morphology, dependency structure and clause boundaries.
3. **Semantic frame stream:** predicates, roles, negation, modality, time, causality and coreference.
4. **Dialogue stream:** communicative acts, dialogue state, requests, references, corrections, commitments and topic changes.
5. **Reasoning stream:** arithmetic, comparison, temporal ordering, set relations, conditionals and consistency.
6. **Realization stream:** meaning plan paired with grammatical surface forms and controlled variations.

### Recommended source families

Each source has a separate license/provenance record. Do not assume that a dataset's hosting platform license overrides the original terms.

- Open English WordNet for lexical-semantic relations (CC BY 4.0).
- Universal Dependencies treebanks for syntax; verify each treebank's individual text and annotation terms.
- FrameNet and PropBank for semantic roles; verify redistribution terms before packaging.
- CLINC150/OOS-Eval for intent and out-of-domain detection, not open-domain conversation.
- Schema-Guided Dialogue for task-oriented dialogue acts, slots and state tracking; it is not a replacement for broad small-talk data.
- DailyDialog for everyday dialogue and dialogue acts, with its non-commercial/share-alike restrictions.
- Carefully sourced human dialogue episodes with conversation-level train/validation/test splits.

Raw billion-word corpora are useful for frequency, morphology and construction statistics. They do not alone teach truth, causality, dialogue state, goals or grounded meaning.

### Data split rule

Split by complete source conversation/document before extracting turn examples. Never randomly place adjacent turns from one conversation in both train and validation. Keep a final test suite untouched.

---

## 8. How MOSAIC learns a genuinely reusable rule

For each candidate microprogram:

1. Find multiple examples with similar typed structures.
2. Anti-unify their event graphs, replacing changing values with typed slots.
3. Infer the minimum set of conditions that predicts the shared state update.
4. Generate contrastive counterfactual examples by role swap, negation, tense, modality and context change.
5. Fit a small fixed-point score for ambiguous rule applicability.
6. Evaluate on held-out conversations, including adversarial near-misses.
7. Keep the program only if it improves held-out exact semantic-delta accuracy without increasing contradiction rate.
8. Compile its predicates into a discrimination trie or feature index.

A learned rule is not trusted because it has a high training score. It must earn support from held-out examples and retain its provenance.

---

## 9. Mandatory evaluation

MOSAIC must be compared against these baselines:
- keyword/finite-state rules;
- TF-IDF or BM25 retrieval;
- fastText-like word/character n-gram intent classifier;
- the existing MCR math-native-v1 encoder/retriever.

Every evaluation should publish examples, metrics, software version, model size, runtime and failure categories.

### Required capability tests

- Greetings and social turn-taking.
- Names, introductions and user-specific facts.
- Corrections and superseded facts.
- Pronouns and coreference.
- Role reversal ("A hit B" vs "B hit A").
- Negation scope.
- Tense and modality.
- Hypotheticals versus assertions.
- Arithmetic and unit reasoning.
- Causal questions with sufficient versus insufficient evidence.
- Topic changes and multi-turn follow-up.
- Ambiguous inputs and appropriate clarification.
- Out-of-domain prompts and abstention.
- Long conversation state consistency.
- Response grammaticality and novelty.

### Required metrics

- intent macro-F1 and out-of-scope false-accept rate;
- POS accuracy, UAS and LAS;
- semantic role F1 and exact frame accuracy;
- coreference accuracy;
- exact state-delta accuracy;
- correction-resolution accuracy;
- contradiction rate;
- answer relevance, factual support and grammaticality;
- selective accuracy versus coverage;
- memory footprint and p50/p95 latency on named hardware.

A system does not pass because it produces a friendly response to "Hi". It passes only when it can maintain the meaning and facts of a conversation across controlled tests.

---

## 10. Implementation plan

### Phase 0 — executable state kernel

Implement in C++:
- interned symbol IDs;
- typed node/edge arena;
- provenance-aware fact assertions and retractions;
- event/delta opcodes;
- bounded rule engine;
- deterministic state serialization;
- property-based tests for correction, negation and role preservation.

No training required for this first kernel.

### Phase 1 — learnable sparse compiler

Add:
- token/character feature extraction;
- learned intent and dialogue-act classifiers;
- calibrated abstention;
- compact POS and dependency modules;
- binary model export and C++ parity tests.

### Phase 2 — executable semantics

Add:
- lexical graph importer;
- event frame compiler;
- semantic role and negation models;
- typed rule induction;
- contrastive/counterfactual validation.

### Phase 3 — continuity and agency

Add:
- persistent dialogue state;
- coreference;
- goal stack;
- limited global workspace;
- self-model and metacognitive monitor;
- incremental episodic consolidation.

### Phase 4 — native response realization

Add:
- meaning-plan grammar;
- morphology and agreement;
- controlled phrase selection;
- response verifier;
- unseen-sentence composition evaluation.

### Phase 5 — integrated compact intelligence

Compile all trained components into a versioned C++ package. Benchmark on the actual target machines. Do not ship unless the integrated system beats the baselines on meaningful multi-turn tasks without exceeding the CPU/memory budgets.

---

## 11. What is genuinely new here, and what is not

Existing work already includes:
- fastText and sparse n-gram linear models;
- symbolic and neuro-symbolic systems;
- vector-symbolic/hyperdimensional computing;
- finite-state language processing;
- program induction and minimum-description-length learning;
- global-workspace-inspired cognitive architectures.

MOSAIC does not claim to have invented those fields.

The proposed new contribution is their specific organization around a **meaning-delta execution contract**:

1. Language is compiled into typed changes to a persistent state.
2. Offline learning discovers compact transition microprograms rather than storing only answer associations.
3. A sparse scheduler activates only relevant microprograms and memories under a hard compute budget.
4. A verifier checks proposed state changes and responses against provenance, counterfactuals and the active goal.
5. The complete runtime is compiled to compact native code and data.

This combination is a research hypothesis. It becomes a technology only if experiments demonstrate that the learned microprograms generalize, the response realizer remains useful, and the bounded runtime achieves the quality/speed target.

---

## 12. Scientific and engineering boundaries

- A two-core CPU can execute compact language, graph, rule and state operations quickly.
- A compact system can be highly capable in a bounded domain.
- No architecture can be promised to reproduce all of a large model's open-domain breadth with negligible memory and computation without evidence.
- Compression cannot remove information that the system never learned or that cannot be inferred from its stored structure.
- Functional continuity, global access and self-monitoring are measurable; subjective experience is not established by those properties alone.
- Neither this proposal nor a future benchmark should claim that MCR is conscious.

The target is a compact system that understands, remembers, reasons and responds through explicit, testable operations—not a small model that merely imitates the appearance of those capabilities.
