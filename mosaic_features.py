"""Feature extractor shared by MOSAIC's offline trainer and C++ runtime.

The representation is deterministic UTF-8 byte hashing. It downloads no
vocabulary, tokenizer, embeddings, or pretrained model.
"""
from __future__ import annotations

from typing import List, Tuple

FEATURE_DIM = 65536


def _tokens(text: str) -> List[bytes]:
    raw = bytearray(text.encode("utf-8"))
    for i, value in enumerate(raw):
        if 65 <= value <= 90:
            raw[i] = value + 32
    tokens: List[bytes] = []
    current = bytearray()
    for value in raw:
        is_word = (97 <= value <= 122) or (48 <= value <= 57) or value >= 128
        if is_word:
            current.append(value)
        elif current:
            tokens.append(bytes(current))
            current.clear()
    if current:
        tokens.append(bytes(current))
    return tokens


def stable_hash(feature: bytes, dimension: int = FEATURE_DIM) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    value = 14695981039346656037
    for byte in feature:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value % dimension


def extract_features(text: str, dimension: int = FEATURE_DIM) -> List[Tuple[int, float]]:
    tokens = _tokens(text)
    result: List[Tuple[int, float]] = []

    def add(feature: bytes, weight: float) -> None:
        result.append((stable_hash(feature, dimension), weight))

    for token in tokens:
        add(b"w:" + token, 1.0)
        padded = b"^" + token + b"$"
        for n in (3, 4, 5):
            for start in range(0, len(padded) - n + 1):
                add(b"c" + str(n).encode("ascii") + b":" + padded[start:start+n], 0.20)

    for i in range(len(tokens) - 1):
        add(b"b:" + tokens[i] + b"|" + tokens[i + 1], 0.80)
    for i in range(len(tokens) - 2):
        add(b"t:" + tokens[i] + b"|" + tokens[i + 1] + b"|" + tokens[i + 2], 0.50)
    return result
