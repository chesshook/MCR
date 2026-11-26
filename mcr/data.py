"""Build a reproducible, quality-checked English conversation mixture for MCR."""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import json
import random
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DATASET_SOURCES = {
    "smol_smoltalk": {
        "repo": "HuggingFaceTB/smol-smoltalk",
        "split": "train",
        "declared_license": "Apache-2.0 (dataset card; inspect upstream component terms)",
        "sample_cap": 50_000,
        "role": "synthetic conversational/instruction examples",
    },
    "ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "declared_license": "MIT",
        "sample_cap": 25_000,
        "role": "synthetic assistant conversations",
    },
    "oasst1": {
        "repo": "OpenAssistant/oasst1",
        "split": "train",
        "declared_license": "Apache-2.0",
        "sample_cap": 10_000,
        "role": "human-generated and human-annotated conversation trees",
    },
}


def _clean_text(value: Any) -> str:
    """Remove invalid nulls while preserving line breaks, code, and indentation."""
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    """Normalize common chat schemas and merge adjacent messages with same role."""
    if not isinstance(raw, list):
        return []
    role_map = {
        "user": "user", "human": "user", "prompter": "user",
        "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
        "system": "system",
    }
    cleaned: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = role_map.get(str(item.get("role", item.get("from", ""))).lower())
        content_value = item.get("content")
        if content_value is None:
            content_value = item.get("value", item.get("text", ""))
        content = _clean_text(content_value)
        if not role or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    while cleaned and cleaned[0]["role"] == "assistant":
        cleaned.pop(0)
    if not cleaned or cleaned[0]["role"] not in {"user", "system"}:
        return []
    if not any(m["role"] == "user" for m in cleaned):
        return []
    # A training example should end with an assistant target, not an unanswered prompt.
    if cleaned[-1]["role"] != "assistant":
        return []
    return cleaned


def _sample_dataset_rows(ds, limit: int, seed: int, source_name: str) -> list[dict[str, Any]]:
    if limit <= 0 or len(ds) == 0:
        return []
    take = min(limit, len(ds))
    selected = ds.shuffle(seed=seed).select(range(take))
    output: list[dict[str, Any]] = []
    for row in selected:
        messages = normalize_messages(row.get("messages"))
        if messages:
            output.append({"messages": messages, "source": source_name})
    return output


def build_oasst_conversations(limit: int, seed: int = 42) -> list[dict[str, Any]]:
    """Reconstruct English OASST1 root-to-leaf paths from parent-linked messages."""
    if limit <= 0:
        return []
    print("Loading OpenAssistant/oasst1 (English human conversation paths)...", flush=True)
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id: dict[str, dict[str, Any]] = {}
    for row in ds:
        if row.get("lang") != "en" or row.get("deleted") is True or row.get("synthetic") is True:
            continue
        if row.get("tree_state") not in (None, "ready_for_export"):
            continue
        message_id = row.get("message_id")
        if message_id:
            by_id[str(message_id)] = row

    children: dict[str, list[str]] = defaultdict(list)
    for message_id, row in by_id.items():
        parent_id = row.get("parent_id")
        if parent_id is None:
            continue
        parent_key = str(parent_id)
        if parent_key in by_id:
            parent = by_id[parent_key]
            if parent.get("message_tree_id") == row.get("message_tree_id"):
                children[parent_key].append(message_id)

    leaves = [mid for mid in by_id if not children.get(mid)]
    paths: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    role_map = {"prompter": "user", "assistant": "assistant"}
    for leaf in leaves:
        chain: list[str] = []
        cursor: str | None = leaf
        visited: set[str] = set()
        tree_id = by_id[leaf].get("message_tree_id")
        while cursor and cursor in by_id and cursor not in visited:
            visited.add(cursor)
            row = by_id[cursor]
            if row.get("message_tree_id") != tree_id:
                break
            chain.append(cursor)
            parent_id = row.get("parent_id")
            cursor = str(parent_id) if parent_id is not None and str(parent_id) in by_id else None
        chain.reverse()
        key = tuple(chain)
        if key in seen or len(chain) < 2:
            continue
        seen.add(key)
        messages = []
        for mid in chain:
            row = by_id[mid]
            role = role_map.get(str(row.get("role", "")).lower())
            content = _clean_text(row.get("text"))
            if role and content:
                messages.append({"role": role, "content": content})
        normalized = normalize_messages(messages)
        if normalized:
            paths.append({"messages": normalized, "source": "oasst1"})

    rng = random.Random(seed)
    rng.shuffle(paths)
    result = paths[: min(limit, len(paths))]
    print(f"  reconstructed {len(result):,} OASST1 English conversation paths", flush=True)
    return result


def _tokenize_conversation(tokenizer, messages: list[dict[str, str]], max_length: int):
    """Tokenize with strict assistant-only loss; reject missing/broken masks."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        raise RuntimeError(
            "The tokenizer chat template did not return assistant_masks. "
            "Refusing to train on user prompts as assistant targets."
        )
    if hasattr(assistant_mask, "tolist"):
        assistant_mask = assistant_mask.tolist()
    if assistant_mask and isinstance(assistant_mask[0], list):
        assistant_mask = assistant_mask[0]
    if len(assistant_mask) != len(input_ids):
        raise RuntimeError(
            f"Assistant mask length {len(assistant_mask)} differs from token length {len(input_ids)}."
        )

    labels = [token if bool(mask) else -100 for token, mask in zip(input_ids, assistant_mask)]
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    assistant_label_count = sum(label != -100 for label in labels)
    if not input_ids or assistant_label_count == 0:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
        "assistant_label_tokens": assistant_label_count,
    }


def prepare_conversation_dataset(
    output_dir: str,
    model_id: str = MODEL_ID,
    smol_limit: int = 50_000,
    ultra_limit: int = 25_000,
    oasst_limit: int = 10_000,
    max_length: int = 2048,
    seed: int = 42,
) -> None:
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer

    if max_length < 256:
        raise ValueError("max_length must be at least 256 tokens")
    print("\n=== MCR conversation data preparation ===", flush=True)
    examples: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    print("Loading HuggingFaceTB/smol-smoltalk...", flush=True)
    smol = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    smol_rows = _sample_dataset_rows(smol, smol_limit, seed, "smol-smoltalk")
    examples.extend(smol_rows)
    counts["smol-smoltalk"] = len(smol_rows)
    print(f"  retained {len(smol_rows):,} completed conversations", flush=True)

    print("Loading HuggingFaceH4/ultrachat_200k train_sft...", flush=True)
    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultra_rows = _sample_dataset_rows(ultra, ultra_limit, seed + 1, "ultrachat_200k")
    examples.extend(ultra_rows)
    counts["ultrachat_200k"] = len(ultra_rows)
    print(f"  retained {len(ultra_rows):,} completed conversations", flush=True)

    oasst_rows = build_oasst_conversations(oasst_limit, seed + 2)
    examples.extend(oasst_rows)
    counts["oasst1"] = len(oasst_rows)

    unique: dict[str, dict[str, Any]] = {}
    for row in examples:
        key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, row)
    examples = list(unique.values())
    rng = random.Random(seed)
    rng.shuffle(examples)
    if len(examples) < 1_000:
        raise RuntimeError(f"Only {len(examples)} usable conversations were prepared; refusing a tiny run.")

    print(f"Total unique completed conversations: {len(examples):,}", flush=True)
    print("Loading SmolLM3 tokenizer and applying its native chat template...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The selected model did not provide a chat template.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = Dataset.from_list(examples)

    def tokenize_batch(batch):
        rows = []
        for messages in batch["messages"]:
            item = _tokenize_conversation(tokenizer, messages, max_length)
            if item is not None:
                rows.append(item)
        keys = ("input_ids", "attention_mask", "labels", "length", "assistant_label_tokens")
        return {key: [row[key] for row in rows] for key in keys}

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        batch_size=64,
        num_proc=2,
        remove_columns=raw.column_names,
        desc="Tokenizing assistant turns",
    )
    if len(tokenized) < 1_000:
        raise RuntimeError(f"Tokenization retained only {len(tokenized)} conversations.")

    split = tokenized.train_test_split(test_size=0.02, seed=seed, shuffle=True)
    data = DatasetDict({"train": split["train"], "validation": split["test"]})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(output_path))

    token_sample = tokenized.select(range(min(1000, len(tokenized))))
    assistant_labels = sum(int(x) for x in token_sample["assistant_label_tokens"])
    total_tokens = sum(int(x) for x in token_sample["length"])
    manifest = {
        "model": model_id,
        "max_sequence_length": max_length,
        "seed": seed,
        "requested_sample_caps": {"smol_smoltalk": smol_limit, "ultrachat": ultra_limit, "oasst1": oasst_limit},
        "source_counts_before_deduplication": counts,
        "unique_conversation_count": len(examples),
        "tokenized_conversation_count": len(tokenized),
        "train_count": len(data["train"]),
        "validation_count": len(data["validation"]),
        "assistant_loss_mask_verified": True,
        "sample_assistant_target_token_fraction": round(assistant_labels / max(1, total_tokens), 4),
        "assistant_target_tokens_in_sample": assistant_labels,
        "input_tokens_in_sample": total_tokens,
        "sources": DATASET_SOURCES,
        "dataset_revision_policy": "Hugging Face dataset main revisions at run time; see dataset fingerprints in the local Datasets cache.",
        "note": "Only assistant spans contribute to loss; user/system spans are masked. Conversations are required to end with an assistant reply.",
    }
    (output_path / "mcr_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Prepared dataset saved: {output_path}", flush=True)
