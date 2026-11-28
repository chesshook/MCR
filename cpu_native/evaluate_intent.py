#!/usr/bin/env python3
"""Evaluate a trained MCR sparse intent model on an untouched TSV split."""
from __future__ import annotations
import argparse
from pathlib import Path

from train_intent import evaluate, read_tsv, features


def load_model(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "MCR_CPU_LINEAR_V1":
            raise ValueError("Unsupported model format")
        buckets = int(handle.readline())
        min_margin = float(handle.readline())
        label_count = int(handle.readline())
        labels = [""] * label_count
        for _ in range(label_count):
            marker, index, label = handle.readline().rstrip("\n").split("\t", 2)
            if marker != "L":
                raise ValueError("Malformed label record")
            labels[int(index)] = label
        weight_count = int(handle.readline())
        weights = {}
        for _ in range(weight_count):
            marker, feature_id, label_id, weight = handle.readline().split()
            if marker != "W":
                raise ValueError("Malformed weight record")
            row = weights.setdefault(int(feature_id), {})
            row[int(label_id)] = float(weight)
    if not labels or any(not label for label in labels):
        raise ValueError("Model contains missing labels")
    return buckets, min_margin, labels, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    args = ap.parse_args()
    buckets, min_margin, labels, weights = load_model(args.model)
    metrics = evaluate(read_tsv(args.test), {label: i for i, label in enumerate(labels)},
                       weights, buckets, min_margin)
    print("TEST METRICS (do not tune thresholds on this split)")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
