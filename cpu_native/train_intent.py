#!/usr/bin/env python3
"""Train a compact averaged sparse-perceptron intent model (no ML framework)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
import random
from pathlib import Path

BUCKETS_DEFAULT = 1_048_573
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def _is_word_byte(value: int) -> bool:
    return value >= 128 or 48 <= value <= 57 or 65 <= value <= 90 or 97 <= value <= 122


def tokenize(raw: bytes) -> list[bytes]:
    result: list[bytes] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c in (9, 10, 11, 12, 13, 32):
            i += 1
            continue
        if _is_word_byte(c):
            start = i
            i += 1
            while i < len(raw):
                if _is_word_byte(raw[i]):
                    i += 1
                elif raw[i] == 39 and i + 1 < len(raw) and _is_word_byte(raw[i + 1]):
                    i += 1
                else:
                    break
            token = bytearray(raw[start:i])
            for p, ch in enumerate(token):
                if 65 <= ch <= 90:
                    token[p] = ch + 32
            result.append(bytes(token))
        else:
            result.append(bytes((c,)))
            i += 1
    return result


def fnv1a(data: bytes, buckets: int) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h % buckets


def features(raw: bytes, buckets: int) -> list[int]:
    tokens = tokenize(raw)
    found = {fnv1a(b"bias", buckets)}
    for i, token in enumerate(tokens):
        if token and _is_word_byte(token[0]):
            found.add(fnv1a(b"w:" + token, buckets))
            padded = b"^" + token + b"$"
            for width in (3, 4, 5):
                for j in range(max(0, len(padded) - width + 1)):
                    found.add(fnv1a(b"c:" + padded[j:j+width], buckets))
        else:
            found.add(fnv1a(b"p:" + token, buckets))
        if i:
            found.add(fnv1a(b"b:" + tokens[i-1] + b"_" + token, buckets))
    return sorted(found)


def read_tsv(path: Path) -> list[tuple[str, bytes]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line or "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            label = label.strip()
            if label and text.strip():
                rows.append((label, text.encode("utf-8", errors="replace")))
    if not rows:
        raise ValueError(f"No labelled examples found in {path}; expected label<TAB>text")
    return rows


def predict_one(x: list[int], weights: dict[int, dict[int, float]], label_count: int) -> tuple[int, list[float]]:
    scores = [0.0] * label_count
    for feature_id in x:
        row = weights.get(feature_id)
        if row:
            for label, weight in row.items():
                scores[label] += weight
    scale = 1.0 / math.sqrt(max(1, len(x)))
    scores = [v * scale for v in scores]
    best = max(range(label_count), key=scores.__getitem__)
    return best, scores


def evaluate(rows, label_to_id, weights, buckets, min_margin=0.0):
    correct = accepted = accepted_correct = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for label, raw in rows:
        if label not in label_to_id:
            continue
        gold = label_to_id[label]
        pred, scores = predict_one(features(raw, buckets), weights, len(label_to_id))
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        correct += pred == gold
        if margin >= min_margin:
            accepted += 1
            accepted_correct += pred == gold
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    f1s = []
    for label_id in range(len(label_to_id)):
        precision = tp[label_id] / max(1, tp[label_id] + fp[label_id])
        recall = tp[label_id] / max(1, tp[label_id] + fn[label_id])
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
    return {"accuracy": correct / max(1, len(rows)), "macro_f1": sum(f1s) / max(1, len(f1s)),
            "selective_accuracy": accepted_correct / max(1, accepted),
            "coverage": accepted / max(1, len(rows)), "examples": len(rows)}


def calibrate_margin(rows, label_to_id, weights, buckets, target_accuracy):
    scored = []
    for label, raw in rows:
        if label not in label_to_id:
            continue
        gold = label_to_id[label]
        pred, scores = predict_one(features(raw, buckets), weights, len(label_to_id))
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        scored.append((margin, pred == gold))
    if not scored:
        return 0.0
    candidates = sorted({0.0, *(margin for margin, _ in scored)})
    best_threshold = 0.0
    best_coverage = -1.0
    for threshold in candidates:
        accepted = [correct for margin, correct in scored if margin >= threshold]
        if not accepted:
            continue
        accuracy = sum(accepted) / len(accepted)
        coverage = len(accepted) / len(scored)
        if accuracy >= target_accuracy and coverage > best_coverage:
            best_threshold, best_coverage = threshold, coverage
    return best_threshold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--valid", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--buckets", type=int, default=BUCKETS_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-selective-accuracy", type=float, default=0.90)
    args = ap.parse_args()
    if args.epochs < 1 or args.buckets < 1024:
        ap.error("epochs must be >=1 and buckets must be >=1024")
    if not 0.50 <= args.target_selective_accuracy <= 1.0:
        ap.error("target-selective-accuracy must be between 0.50 and 1.0")

    train_rows = read_tsv(args.train)
    labels = sorted({label for label, _ in train_rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    encoded = [(label_to_id[label], features(raw, args.buckets)) for label, raw in train_rows]

    # Sparse averaged multiclass perceptron. Only active feature/class weights exist.
    weights: dict[int, dict[int, float]] = defaultdict(dict)
    totals: dict[tuple[int, int], float] = defaultdict(float)
    stamps: dict[tuple[int, int], int] = defaultdict(int)
    step = 0
    rng = random.Random(args.seed)
    print(f"Examples={len(encoded):,} labels={len(labels)} buckets={args.buckets:,}", flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        mistakes = 0
        for row_index in order:
            gold, x = encoded[row_index]
            pred, _ = predict_one(x, weights, len(labels))
            if pred != gold:
                mistakes += 1
                delta = 1.0 / math.sqrt(max(1, len(x)))
                for feature_id in x:
                    for label, change in ((gold, delta), (pred, -delta)):
                        key = (feature_id, label)
                        current = weights[feature_id].get(label, 0.0)
                        totals[key] += (step - stamps[key]) * current
                        stamps[key] = step
                        weights[feature_id][label] = current + change
            step += 1
        print(f"epoch={epoch+1}/{args.epochs} mistakes={mistakes:,} ({mistakes/max(1,len(encoded)):.3%})", flush=True)

    averaged: dict[int, dict[int, float]] = defaultdict(dict)
    for feature_id, row in weights.items():
        for label, current in row.items():
            key = (feature_id, label)
            total = totals[key] + (step - stamps[key]) * current
            value = total / max(1, step)
            if abs(value) >= 1e-7:
                averaged[feature_id][label] = value

    valid_rows = read_tsv(args.valid) if args.valid else []
    min_margin = calibrate_margin(valid_rows, label_to_id, averaged, args.buckets,
                                  args.target_selective_accuracy) if valid_rows else 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = sum(len(v) for v in averaged.values())
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        out.write("MCR_CPU_LINEAR_V1\n")
        out.write(f"{args.buckets}\n")
        out.write(f"{min_margin:.9g}\n")
        out.write(f"{len(labels)}\n")
        for i, label in enumerate(labels):
            out.write(f"L\t{i}\t{label}\n")
        out.write(f"{count}\n")
        for feature_id in sorted(averaged):
            for label in sorted(averaged[feature_id]):
                out.write(f"W\t{feature_id}\t{label}\t{averaged[feature_id][label]:.9g}\n")
    print(f"Exported {count:,} sparse weights to {args.output} ({args.output.stat().st_size/1024**2:.2f} MiB)", flush=True)

    if valid_rows:
        print(f"Calibrated abstention margin={min_margin:.6f} (target selective accuracy={args.target_selective_accuracy:.1%})", flush=True)
        print("VALIDATION", evaluate(valid_rows, label_to_id, averaged, args.buckets, min_margin), flush=True)


if __name__ == "__main__":
    main()
