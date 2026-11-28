"""MOSAIC learned lexical segmenter with UTF-8 byte fallback.

Vocabulary pieces are learned only from the training corpus. Unknown names and
words are decomposed into bytes and are never discarded as UNK.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


PAD_ID = 0
BOS_ID = 1
USER_ID = 2
ASSISTANT_ID = 3
TURN_ID = 4
BYTE_BASE = 5
BYTE_COUNT = 256
PIECE_BASE = BYTE_BASE + BYTE_COUNT


def split_pieces(text: str) -> List[bytes]:
    raw = text.encode("utf-8", errors="replace")
    pieces: List[bytes] = []
    i = 0
    while i < len(raw):
        start = i
        while i < len(raw) and raw[i] in (9, 10, 11, 12, 13, 32):
            i += 1
        while i < len(raw) and raw[i] not in (9, 10, 11, 12, 13, 32):
            i += 1
        if i == start:
            i += 1
        pieces.append(raw[start:i])
    return pieces


@dataclass
class LearnedTokenizer:
    pieces: List[bytes]
    max_piece_bytes: int = 32

    def __post_init__(self) -> None:
        self.to_id: Dict[bytes, int] = {
            piece: PIECE_BASE + i for i, piece in enumerate(self.pieces)
        }
        if len(self.to_id) != len(self.pieces):
            raise ValueError("duplicate vocabulary pieces")
        if any(not piece for piece in self.pieces):
            raise ValueError("empty vocabulary piece")

    @property
    def vocab_size(self) -> int:
        return PIECE_BASE + len(self.pieces)

    def encode(self, text: str) -> List[int]:
        result: List[int] = []
        for piece in split_pieces(text):
            pos = 0
            while pos < len(piece):
                max_len = min(self.max_piece_bytes, len(piece) - pos)
                found_id = None
                found_len = 0
                for size in range(max_len, 0, -1):
                    token_id = self.to_id.get(piece[pos:pos + size])
                    if token_id is not None:
                        found_id = token_id
                        found_len = size
                        break
                if found_id is not None:
                    result.append(found_id)
                    pos += found_len
                else:
                    result.append(BYTE_BASE + piece[pos])
                    pos += 1
        return result

    def decode(self, ids: Sequence[int]) -> str:
        output = bytearray()
        for token_id in ids:
            if BYTE_BASE <= token_id < PIECE_BASE:
                output.append(token_id - BYTE_BASE)
            elif PIECE_BASE <= token_id < self.vocab_size:
                output.extend(self.pieces[token_id - PIECE_BASE])
        return output.decode("utf-8", errors="replace")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        min_piece_frequency: int = 2,
        max_piece_bytes: int = 32,
        subword_source_limit: int = 60000,
    ) -> "LearnedTokenizer":
        full_counts: Counter[bytes] = Counter()
        for text in texts:
            full_counts.update(split_pieces(text))

        capacity = max(0, vocab_size - PIECE_BASE)
        if capacity == 0:
            raise ValueError("vocab_size must exceed reserved byte/special IDs")

        full_budget = min(capacity, int(capacity * 0.68))
        full_pieces = [
            piece for piece, count in full_counts.most_common()
            if count >= min_piece_frequency and len(piece) <= max_piece_bytes
        ][:full_budget]
        selected = set(full_pieces)

        sub_counts: Counter[bytes] = Counter()
        for piece, frequency in full_counts.most_common(subword_source_limit):
            if len(piece) < 3:
                continue
            for start in range(len(piece)):
                max_end = min(len(piece), start + max_piece_bytes)
                for end in range(start + 2, max_end + 1):
                    candidate = piece[start:end]
                    if candidate not in selected:
                        sub_counts[candidate] += frequency

        sub_budget = capacity - len(full_pieces)
        sub_pieces = []
        for piece, count in sub_counts.most_common():
            if count < min_piece_frequency:
                break
            if piece not in selected and len(piece) <= max_piece_bytes:
                sub_pieces.append(piece)
                selected.add(piece)
                if len(sub_pieces) >= sub_budget:
                    break

        pieces = full_pieces + sub_pieces
        if not pieces:
            raise RuntimeError("training corpus produced an empty tokenizer vocabulary")
        return cls(pieces=pieces, max_piece_bytes=max_piece_bytes)
