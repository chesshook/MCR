"""Tokenizer-free UTF-8 byte encoding helpers."""
from __future__ import annotations


def encode_bytes(text: str, max_bytes: int = 384) -> list[int]:
    """Map UTF-8 bytes to 1..256; zero is reserved solely for padding.

    Long strings keep both their beginning and end so the current user turn is
    less likely to disappear when older conversation context is truncated.
    """
    if max_bytes < 8:
        raise ValueError("max_bytes must be at least 8")
    raw = str(text).encode("utf-8", errors="replace")
    if len(raw) > max_bytes:
        left = max_bytes * 2 // 3
        raw = raw[:left] + raw[-(max_bytes - left):]
    return [byte + 1 for byte in raw]


def pad_sequences(sequences: list[list[int]], pad_to: int | None = None) -> tuple[list[list[int]], list[list[int]], list[int]]:
    if not sequences:
        raise ValueError("Cannot pad an empty batch")
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths) if pad_to is None else max(pad_to, max(lengths))
    ids = [sequence + [0] * (width - len(sequence)) for sequence in sequences]
    masks = [[1] * length + [0] * (width - length) for length in lengths]
    return ids, masks, lengths
