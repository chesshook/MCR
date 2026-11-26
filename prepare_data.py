"""Prepare conversational and semantic-relation datasets; no model downloads."""
from __future__ import annotations
import argparse
from mcr.data import prepare_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="/kaggle/working/MCR_DATA")
    p.add_argument("--chat-limit", type=int, default=120_000)
    p.add_argument("--nli-limit", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_data(args.output_dir, args.chat_limit, args.nli_limit, args.seed)


if __name__ == "__main__":
    main()
