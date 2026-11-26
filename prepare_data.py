"""CLI wrapper for MCR multi-source chat data preparation."""
from __future__ import annotations

import argparse

from mcr.data import prepare_conversation_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--smol-limit", type=int, default=50_000)
    parser.add_argument("--ultra-limit", type=int, default=25_000)
    parser.add_argument("--oasst-limit", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_conversation_dataset(
        output_dir=args.output_dir,
        model_id=args.model_id,
        smol_limit=args.smol_limit,
        ultra_limit=args.ultra_limit,
        oasst_limit=args.oasst_limit,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
