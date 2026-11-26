"""Download and normalize conversation + natural-language-inference training data.

This module downloads datasets only. It never downloads model weights or tokenizers.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import json
import random

SOURCES = {
    "ultrachat_200k": "HuggingFaceH4/ultrachat_200k",
    "oasst1": "OpenAssistant/oasst1",
    "daily_dialog": "daily_dialog",
    "snli": "stanfordnlp/snli",
    "multi_nli": "nyu-mll/multi_nli",
}


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    """Normalize common chat schemas into alternating user/assistant messages."""
    if not isinstance(raw, list):
        return []
    role_map = {
        "user": "user", "human": "user", "prompter": "user",
        "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
        "system": "system",
    }
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role_raw = item.get("role", item.get("from", ""))
        role = role_map.get(str(role_raw).lower())
        value = item.get("content")
        if value is None:
            value = item.get("text", item.get("value", ""))
        content = clean_text(value)
        if not role or not content:
            continue
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n" + content
        else:
            result.append({"role": role, "content": content})
    while result and result[0]["role"] == "assistant":
        result.pop(0)
    return result


def messages_to_pairs(messages: list[dict[str, str]], max_context_turns: int = 6) -> list[dict[str, str]]:
    """Turn a multi-turn dialogue into context -> assistant-reply examples."""
    pairs: list[dict[str, str]] = []
    usable: list[dict[str, str]] = []
    for message in messages:
        role, content = message.get("role"), clean_text(message.get("content"))
        if role in {"user", "assistant"} and content:
            usable.append({"role": role, "content": content})
    for index, message in enumerate(usable):
        if message["role"] != "assistant" or index == 0:
            continue
        if not any(m["role"] == "user" for m in usable[:index]):
            continue
        history = usable[max(0, index - max_context_turns):index]
        context_lines = []
        for history_index, item in enumerate(history):
            if history_index == len(history) - 1 and item["role"] == "user":
                prefix = "CURRENT USER: "
            else:
                prefix = "USER: " if item["role"] == "user" else "ASSISTANT: "
            context_lines.append(prefix + item["content"])
        context = "\n".join(context_lines)
        reply = message["content"]
        if len(context) >= 3 and len(reply) >= 2:
            pairs.append({"context": context, "reply": reply})
    return pairs


def _sample_rows(dataset, limit: int, seed: int):
    if limit <= 0 or len(dataset) == 0:
        return []
    shuffled = dataset.shuffle(seed=seed)
    return [shuffled[i] for i in range(min(limit, len(shuffled)))]


def _oasst_paths(limit: int, seed: int) -> list[dict[str, str]]:
    from datasets import load_dataset
    print("Loading OpenAssistant English conversations...", flush=True)
    ds = load_dataset(SOURCES["oasst1"], split="train")
    rows: dict[str, dict[str, Any]] = {}
    for row in ds:
        if row.get("lang") != "en" or row.get("deleted") is True or row.get("synthetic") is True:
            continue
        if row.get("tree_state") not in (None, "ready_for_export"):
            continue
        mid = row.get("message_id")
        if mid:
            rows[str(mid)] = row
    children: dict[str, list[str]] = defaultdict(list)
    for mid, row in rows.items():
        parent = row.get("parent_id")
        if parent is not None and str(parent) in rows:
            parent_key = str(parent)
            if rows[parent_key].get("message_tree_id") == row.get("message_tree_id"):
                children[parent_key].append(mid)
    leaves = [mid for mid in rows if not children.get(mid)]
    rng = random.Random(seed)
    rng.shuffle(leaves)
    paths: list[dict[str, str]] = []
    role_map = {"prompter": "user", "assistant": "assistant"}
    for leaf in leaves:
        chain: list[str] = []
        cursor: str | None = leaf
        seen: set[str] = set()
        tree_id = rows[leaf].get("message_tree_id")
        while cursor and cursor in rows and cursor not in seen:
            seen.add(cursor)
            row = rows[cursor]
            if row.get("message_tree_id") != tree_id:
                break
            chain.append(cursor)
            parent = row.get("parent_id")
            cursor = str(parent) if parent is not None and str(parent) in rows else None
        chain.reverse()
        messages = []
        for mid in chain:
            row = rows[mid]
            role = role_map.get(str(row.get("role", "")).lower())
            content = clean_text(row.get("text"))
            if role and content:
                messages.append({"role": role, "content": content})
        pairs = messages_to_pairs(messages)
        if pairs:
            paths.extend(pairs)
        if len(paths) >= limit:
            break
    print(f"  reconstructed {len(paths):,} English context/reply pairs", flush=True)
    return paths[:limit]


def build_conversation_pairs(chat_limit: int, seed: int = 42) -> tuple[list[dict[str, str]], dict[str, int]]:
    from datasets import load_dataset
    collected: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    # Everyday dialogues first, then broad multi-turn instruction/conversation data.
    try:
        print("Loading DailyDialog...", flush=True)
        daily = load_dataset(SOURCES["daily_dialog"], split="train")
        rows = _sample_rows(daily, min(15_000, chat_limit // 5), seed)
        added = 0
        for row in rows:
            utterances = row.get("dialog", [])
            messages = []
            for i, utterance in enumerate(utterances):
                content = clean_text(utterance)
                if content:
                    messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
            pairs = messages_to_pairs(messages)
            collected.extend(pairs)
            added += len(pairs)
        counts["daily_dialog"] = added
        print(f"  retained {added:,} context/reply pairs", flush=True)
    except Exception as exc:
        print(f"  DailyDialog skipped: {type(exc).__name__}: {exc}", flush=True)
        counts["daily_dialog"] = 0

    try:
        print("Loading UltraChat 200k conversations...", flush=True)
        ultra = load_dataset(SOURCES["ultrachat_200k"], split="train_sft")
        remaining = max(0, chat_limit - len(collected))
        rows = _sample_rows(ultra, min(remaining, 65_000), seed + 1)
        added = 0
        for row in rows:
            pairs = messages_to_pairs(normalize_messages(row.get("messages")))
            collected.extend(pairs)
            added += len(pairs)
        counts["ultrachat_200k"] = added
        print(f"  retained {added:,} context/reply pairs", flush=True)
    except Exception as exc:
        print(f"  UltraChat skipped: {type(exc).__name__}: {exc}", flush=True)
        counts["ultrachat_200k"] = 0

    remaining = max(0, chat_limit - len(collected))
    if remaining:
        try:
            pairs = _oasst_paths(remaining, seed + 2)
            collected.extend(pairs)
            counts["oasst1"] = len(pairs)
        except Exception as exc:
            print(f"  OpenAssistant skipped: {type(exc).__name__}: {exc}", flush=True)
            counts["oasst1"] = 0
    else:
        counts["oasst1"] = 0

    # Deduplicate by input and target; remove near-empty and pathological samples.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in collected:
        context = clean_text(row.get("context"))
        reply = clean_text(row.get("reply"))
        if not (3 <= len(context) <= 12_000 and 2 <= len(reply) <= 5_000):
            continue
        unique.setdefault((context.casefold(), reply.casefold()), {"context": context, "reply": reply})
    result = list(unique.values())
    random.Random(seed).shuffle(result)
    return result[:chat_limit], counts


def build_nli_pairs(limit: int, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load SNLI/MultiNLI sentence-pair judgments; labels: entail/neutral/contradict."""
    from datasets import load_dataset
    collected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    sources = [("snli", SOURCES["snli"]), ("multi_nli", SOURCES["multi_nli"])]
    per_source = max(1, limit // len(sources))
    for source_index, (name, repo) in enumerate(sources):
        wanted = min(per_source, limit - len(collected))
        if wanted <= 0:
            break
        try:
            print(f"Loading {name} semantic-relation data ({repo})...", flush=True)
            ds = load_dataset(repo, split="train")
            sampled = _sample_rows(ds, min(len(ds), wanted * 2), seed + 10 + source_index)
            accepted = 0
            for row in sampled:
                premise = clean_text(row.get("premise"))
                hypothesis = clean_text(row.get("hypothesis"))
                try:
                    label = int(row.get("label", -1))
                except (TypeError, ValueError):
                    continue
                if label not in (0, 1, 2) or not premise or not hypothesis:
                    continue
                # Official SNLI/MultiNLI mapping: 0 entailment, 1 neutral, 2 contradiction.
                collected.append({"premise": premise, "hypothesis": hypothesis, "label": label})
                accepted += 1
                if accepted >= wanted:
                    break
            counts[name] = accepted
            print(f"  retained {accepted:,} labeled sentence pairs", flush=True)
        except Exception as exc:
            print(f"  {name} skipped: {type(exc).__name__}: {exc}", flush=True)
            counts[name] = 0
    random.Random(seed + 20).shuffle(collected)
    return collected[:limit], counts


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_data(output_dir: str, chat_limit: int = 120_000, nli_limit: int = 200_000, seed: int = 42) -> None:
    if chat_limit < 1000 or nli_limit < 1000:
        raise ValueError("Use at least 1,000 examples for each training objective.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    print("\n=== MCR mathematical-native data preparation ===", flush=True)
    print("No model checkpoint, tokenizer vocabulary, or pretrained embeddings are downloaded.", flush=True)

    chat, chat_counts = build_conversation_pairs(chat_limit, seed)
    nli, nli_counts = build_nli_pairs(nli_limit, seed)
    if len(chat) < 1000:
        raise RuntimeError(f"Only {len(chat)} conversation pairs were loaded; at least 1,000 are required.")
    if len(nli) < 1000:
        raise RuntimeError(f"Only {len(nli)} NLI pairs were loaded; at least 1,000 are required.")

    rng = random.Random(seed)
    rng.shuffle(chat)
    rng.shuffle(nli)
    chat_val_n = max(500, int(len(chat) * 0.03))
    nli_val_n = max(500, int(len(nli) * 0.03))
    chat_train, chat_val = chat[chat_val_n:], chat[:chat_val_n]
    nli_train, nli_val = nli[nli_val_n:], nli[:nli_val_n]

    counts = {
        "chat_train": _write_jsonl(root / "chat_train.jsonl", chat_train),
        "chat_validation": _write_jsonl(root / "chat_validation.jsonl", chat_val),
        "nli_train": _write_jsonl(root / "nli_train.jsonl", nli_train),
        "nli_validation": _write_jsonl(root / "nli_validation.jsonl", nli_val),
    }
    manifest = {
        "architecture": "MCR byte-semantic encoder + metric learning + NLI relation head",
        "pretrained_model_weights": False,
        "pretrained_tokenizer": False,
        "autoregressive_language_model": False,
        "input_representation": "UTF-8 byte values; no BPE, WordPiece, SentencePiece, or downloaded vocabulary",
        "chat_source_counts": chat_counts,
        "nli_source_counts": nli_counts,
        "split_counts": counts,
        "label_map": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "seed": seed,
        "licenses": {
            "ultrachat_200k": "MIT (verify current upstream dataset card and source terms)",
            "oasst1": "Apache-2.0 (verify current upstream dataset card)",
            "daily_dialog": "CC BY-NC-SA 4.0 (non-commercial; verify upstream terms)",
            "snli": "CC BY-SA 4.0 (verify current upstream dataset card)",
            "multi_nli": "CC BY-SA 4.0 (verify current upstream dataset card)",
        },
        "warning": "Dataset labels supervise limited tasks; they do not establish human-level understanding.",
    }
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Data written to: {root}", flush=True)
