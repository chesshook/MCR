"""Load licensed conversational corpora and create turn-conditioned dialogue samples."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

Conversation = List[Tuple[str, str]]


def _normalise_messages(raw_messages: Sequence[dict]) -> Optional[Conversation]:
    result: Conversation = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("speaker", ""))).lower()
        if role in ("user", "human", "prompter", "customer"):
            role = "user"
        elif role in ("assistant", "gpt", "bot", "agent"):
            role = "assistant"
        else:
            continue

        text = item.get("content", item.get("text", item.get("value", "")))
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text or len(text) > 10000:
            continue

        if result and result[-1][0] == role:
            result[-1] = (role, result[-1][1] + "\n" + text)
        else:
            result.append((role, text))

    if len(result) < 2 or not any(role == "user" for role, _ in result) or not any(
        role == "assistant" for role, _ in result
    ):
        return None
    return result


def _is_english(row) -> bool:
    language = str(row.get("lang", "en")).lower()
    return language.startswith("en") or language == "eng" or language == ""


def _oasst_split_to_conversations(split) -> List[Conversation]:
    rows = list(split)
    by_id: Dict[str, dict] = {}
    for row in rows:
        if not _is_english(row):
            continue
        message_id = row.get("message_id")
        if message_id:
            by_id[str(message_id)] = row

    result: List[Conversation] = []
    seen = set()
    for row in rows:
        if not _is_english(row) or str(row.get("role", "")).lower() != "assistant":
            continue

        chain = []
        current = row
        visited = set()
        while current is not None:
            message_id = str(current.get("message_id", ""))
            if not message_id or message_id in visited:
                break
            visited.add(message_id)
            chain.append(current)
            parent_id = current.get("parent_id")
            current = by_id.get(str(parent_id)) if parent_id else None
        chain.reverse()

        raw = []
        path_ids = []
        for node in chain:
            role = str(node.get("role", "")).lower()
            if role == "prompter":
                role = "user"
            if role in ("user", "assistant"):
                raw.append({"role": role, "text": node.get("text", "")})
                path_ids.append(str(node.get("message_id", "")))
        if len(raw) < 2 or not path_ids:
            continue
        key = tuple(path_ids)
        if key in seen:
            continue
        seen.add(key)
        conversation = _normalise_messages(raw)
        if conversation and conversation[-1][0] == "assistant":
            result.append(conversation)
    return result


def load_conversational_corpora():
    from datasets import load_dataset

    train: List[Conversation] = []
    validation: List[Conversation] = []
    test: List[Conversation] = []
    manifest = {}

    try:
        print("Loading UltraChat 200k train_sft/test_sft...", flush=True)
        train_rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        test_rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        ultra_train = []
        for row in train_rows:
            conversation = _normalise_messages(row.get("messages", []))
            if conversation:
                ultra_train.append(conversation)
        ultra_test = []
        for row in test_rows:
            conversation = _normalise_messages(row.get("messages", []))
            if conversation:
                ultra_test.append(conversation)

        rng = random.Random(20261128)
        rng.shuffle(ultra_train)
        cut = max(1, int(len(ultra_train) * 0.97))
        train.extend(ultra_train[:cut])
        validation.extend(ultra_train[cut:])
        test.extend(ultra_test)
        manifest["ultrachat_200k"] = {
            "train_conversations": len(ultra_train[:cut]),
            "validation_conversations": len(ultra_train[cut:]),
            "test_conversations": len(ultra_test),
            "license": "MIT",
        }
    except Exception as exc:
        print(f"WARNING: UltraChat could not be loaded: {exc}", flush=True)
        manifest["ultrachat_200k"] = {"error": str(exc)}

    try:
        print("Loading OpenAssistant OASST1 message trees...", flush=True)
        oasst = load_dataset("OpenAssistant/oasst1")
        split_names = set(oasst.keys())
        oasst_train = (
            _oasst_split_to_conversations(oasst["train"])
            if "train" in split_names else []
        )
        oasst_val = (
            _oasst_split_to_conversations(oasst["validation"])
            if "validation" in split_names else []
        )
        if oasst_train:
            train.extend(oasst_train)
        if oasst_val:
            validation.extend(oasst_val)
        manifest["oasst1"] = {
            "train_conversations": len(oasst_train),
            "validation_conversations": len(oasst_val),
            "license": "Apache-2.0",
        }
    except Exception as exc:
        print(f"WARNING: OASST1 could not be loaded: {exc}", flush=True)
        manifest["oasst1"] = {"error": str(exc)}

    try:
        print("Loading DailyDialog parquet mirror...", flush=True)
        daily = load_dataset("roskoN/dailydialog")
        daily_counts = {}
        for split_name, destination in (
            ("train", train), ("validation", validation), ("test", test)
        ):
            if split_name not in daily:
                continue
            count = 0
            for row in daily[split_name]:
                utterances = row.get("utterances", row.get("dialog", []))
                if isinstance(utterances, str):
                    continue
                raw = []
                for index, utterance in enumerate(utterances):
                    role = "user" if index % 2 == 0 else "assistant"
                    raw.append({"role": role, "text": str(utterance)})
                conversation = _normalise_messages(raw)
                if conversation:
                    destination.append(conversation)
                    count += 1
            daily_counts[split_name] = count
        manifest["daily_dialog"] = {
            **daily_counts,
            "license": "CC BY-NC-SA 4.0; non-commercial/share-alike terms apply",
        }
    except Exception as exc:
        print(f"WARNING: DailyDialog could not be loaded: {exc}", flush=True)
        manifest["daily_dialog"] = {"error": str(exc)}

    def deduplicate(items: List[Conversation]) -> List[Conversation]:
        seen = set()
        output = []
        for conversation in items:
            key = tuple(conversation)
            if key not in seen:
                seen.add(key)
                output.append(conversation)
        return output

    train = deduplicate(train)
    validation = deduplicate(validation)
    test = deduplicate(test)
    if not train:
        raise RuntimeError(
            "No training dialogues were loaded. Inspect dataset errors; "
            "refusing to train on an empty or synthetic fallback corpus."
        )
    return train, validation, test, manifest


def make_response_examples(conversations, tokenizer, max_context=352,
                           max_target=128, max_messages=8, max_examples=None):
    from tokenizer import BOS_ID, USER_ID, ASSISTANT_ID, TURN_ID

    examples = []
    for conversation in conversations:
        for target_index, (role, response) in enumerate(conversation):
            if role != "assistant" or target_index == 0:
                continue
            prior = conversation[:target_index]
            if not any(previous_role == "user" for previous_role, _ in prior):
                continue
            prior = prior[-max_messages:]

            encoded_history = []
            for previous_role, message in prior:
                marker = USER_ID if previous_role == "user" else ASSISTANT_ID
                encoded_history.append((marker, tokenizer.encode(message), TURN_ID))

            while encoded_history:
                total = 1 + sum(2 + len(tokens) for _, tokens, _ in encoded_history)
                if total <= max_context:
                    break
                encoded_history.pop(0)

            prefix = [BOS_ID]
            for marker, tokens, end_marker in encoded_history:
                prefix.append(marker)
                prefix.extend(tokens)
                prefix.append(end_marker)
            if len(prefix) > max_context:
                prefix = [BOS_ID] + prefix[-(max_context - 1):]

            target = tokenizer.encode(response)[:max_target]
            if not target:
                continue
            sequence = prefix + [ASSISTANT_ID] + target + [TURN_ID]
            target_start = len(prefix) + 1
            examples.append((sequence, target_start))
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples
