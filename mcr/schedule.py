"""Pure-Python deterministic fair interleaving for multi-task training."""
from __future__ import annotations
import random


def make_task_schedule(chat_batches: int, nli_batches: int, seed: int) -> list[str]:
    """Consume each batch once while distributing tasks across the epoch."""
    if chat_batches < 0 or nli_batches < 0:
        raise ValueError("Batch counts cannot be negative")
    tasks: list[str] = []
    c = n = 0
    rng = random.Random(seed)
    while c < chat_batches or n < nli_batches:
        if c >= chat_batches:
            choose_chat = False
        elif n >= nli_batches:
            choose_chat = True
        else:
            choose_chat = (c / max(1, chat_batches)) <= (n / max(1, nli_batches))
            if abs((c / max(1, chat_batches)) - (n / max(1, nli_batches))) < 1e-12:
                choose_chat = bool(rng.getrandbits(1))
        if choose_chat:
            tasks.append("chat")
            c += 1
        else:
            tasks.append("nli")
            n += 1
    return tasks
