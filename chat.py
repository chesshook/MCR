"""Interactive MCR chat using semantic nearest-neighbour response retrieval.

There is no language-model decoder here: each reply is selected from a paired
conversation example by the trained semantic metric model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mcr.encoding import encode_bytes, pad_sequences
from mcr.model import MCRModel

LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
FALLBACK = (
    "I'm not confident I've matched that to a reliable response. "
    "Could you rephrase it or add a little context?"
)


class MCRChat:
    def __init__(self, model_dir: str, device: str = "auto", max_bytes: int | None = None,
                 min_similarity: float = 0.20):
        self.root = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        config = json.loads((self.root / "mcr_config.json").read_text(encoding="utf-8"))
        self.max_bytes = int(max_bytes or config["max_bytes"])
        self.min_similarity = float(config.get("default_min_similarity", min_similarity))
        if min_similarity != 0.20:
            self.min_similarity = float(min_similarity)
        self.model = MCRModel().to(self.device)
        try:
            state = torch.load(self.root / "mcr_model.pt", map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.root / "mcr_model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        # Convert once at load time; avoid copying the complete index on every turn.
        self.index = np.asarray(np.load(self.root / "response_context_index.npy"), dtype=np.float32)
        with (self.root / "response_bank.jsonl").open("r", encoding="utf-8") as handle:
            self.responses = [json.loads(line)["reply"] for line in handle if line.strip()]
        if len(self.responses) != self.index.shape[0]:
            raise RuntimeError("MCR retrieval index and response bank have different row counts.")
        self.last_matches: list[tuple[float, str]] = []

    def _encode(self, text: str) -> np.ndarray:
        ids, masks, _ = pad_sequences([encode_bytes(text, self.max_bytes)])
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
        masks_t = torch.tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                vector = self.model.encoder(ids_t, masks_t)
        return vector.float().cpu().numpy()[0]

    @staticmethod
    def make_context(history: list[dict[str, str]], current: str) -> str:
        recent = history[-5:]
        lines = [
            ("USER: " if row["role"] == "user" else "ASSISTANT: ") + row["content"]
            for row in recent
        ]
        lines.append("CURRENT USER: " + current)
        return "\n".join(lines)

    def answer(self, history: list[dict[str, str]], prompt: str, top_k: int = 5) -> str:
        query = self._encode(self.make_context(history, prompt)).astype(np.float32)
        scores = self.index @ query
        k = min(max(1, top_k), len(scores))
        candidate_ids = np.argpartition(-scores, k - 1)[:k]
        candidate_ids = candidate_ids[np.argsort(-scores[candidate_ids])]
        matches: list[tuple[float, str]] = []
        seen: set[str] = set()
        for index in candidate_ids:
            reply = self.responses[int(index)].strip()
            key = reply.casefold()
            if reply and key not in seen:
                matches.append((float(scores[index]), reply))
                seen.add(key)
        self.last_matches = matches
        if not matches or matches[0][0] < self.min_similarity:
            return FALLBACK
        return matches[0][1]

    def judge(self, premise: str, hypothesis: str) -> tuple[str, list[float]]:
        left, right = encode_bytes(premise, self.max_bytes), encode_bytes(hypothesis, self.max_bytes)
        li, lm, _ = pad_sequences([left])
        ri, rm, _ = pad_sequences([right])
        batch = {
            "premise_ids": torch.tensor(li, dtype=torch.long, device=self.device),
            "premise_mask": torch.tensor(lm, dtype=torch.bool, device=self.device),
            "hypothesis_ids": torch.tensor(ri, dtype=torch.long, device=self.device),
            "hypothesis_mask": torch.tensor(rm, dtype=torch.bool, device=self.device),
        }
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = self.model(task="nli", **batch)
            probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        label = LABELS[int(np.argmax(probabilities))]
        return label, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/kaggle/working/MCR_OUTPUT")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--min-similarity", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    bot = MCRChat(args.model_dir, args.device, min_similarity=args.min_similarity)
    history: list[dict[str, str]] = []
    print("MCR mathematical-native chat ready.")
    print("Replies are retrieved from learned semantic memory; there is no text decoder.")
    print("Commands: /exit, /reset, /judge PREMISE || HYPOTHESIS, /matches")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        lowered = prompt.lower()
        if lowered in {"/exit", "exit", "quit"}:
            break
        if lowered == "/reset":
            history.clear()
            print("MCR> Conversation history cleared.")
            continue
        if lowered.startswith("/judge "):
            payload = prompt[7:]
            if "||" not in payload:
                print("Usage: /judge premise || hypothesis")
                continue
            premise, hypothesis = payload.split("||", 1)
            label, probabilities = bot.judge(premise.strip(), hypothesis.strip())
            print("MCR>", label, "| probabilities:", [round(x, 4) for x in probabilities])
            continue
        if lowered == "/matches":
            if not bot.last_matches:
                print("MCR> No retrieval candidates yet.")
            for score, reply in bot.last_matches:
                print(f"  {score:.4f}  {reply[:240]}")
            continue
        response = bot.answer(history, prompt, top_k=args.top_k)
        print("\nMCR>", response, flush=True)
        if args.debug and bot.last_matches:
            print(f"[retrieval similarity={bot.last_matches[0][0]:.4f}]", flush=True)
        history.extend([{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}])
        history = history[-12:]


if __name__ == "__main__":
    main()
