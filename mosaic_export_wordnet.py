"""Export Open English WordNet 2025+ into a compact TSV for the C++ runtime.

The WordNet data is downloaded only during the one-time bake. Runtime lookup is
offline and does not need Python or network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

LEXICON_ID = "oewn:2025+"
LICENSE = "CC BY 4.0"


def clean(value: str) -> str:
    value = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-senses", type=int, default=4)
    args = parser.parse_args()
    if args.max_senses < 1 or args.max_senses > 12:
        raise ValueError("--max-senses must be between 1 and 12")

    try:
        import wn
    except ImportError as exc:
        raise RuntimeError("Install the 'wn' package before exporting WordNet.") from exc

    print(f"Loading {LEXICON_ID}; first run downloads the official lexical data.", flush=True)
    wn.download(LEXICON_ID)
    wordnet = wn.Wordnet(LEXICON_ID)

    rows: set[tuple[str, str, str, str, str]] = set()
    words = wordnet.words()
    print(f"Enumerating {len(words):,} lexical entries...", flush=True)
    for index, word in enumerate(words, start=1):
        lemma = clean(word.lemma()).replace("_", " ").lower()
        if not lemma:
            continue
        senses = word.synsets()[:args.max_senses]
        for synset in senses:
            definition = clean(synset.definition())
            if not definition:
                continue
            synonyms = sorted({
                clean(item).replace("_", " ").lower()
                for item in synset.lemmas()
                if clean(item) and clean(item).replace("_", " ").lower() != lemma
            })
            hypernyms = sorted({
                clean(lemma_name).replace("_", " ").lower()
                for parent in synset.hypernyms()[:4]
                for lemma_name in parent.lemmas()[:4]
                if clean(lemma_name)
            })
            rows.add((
                lemma,
                clean(word.pos()),
                definition,
                ";".join(synonyms[:12]),
                ";".join(hypernyms[:12]),
            ))
        if index % 20000 == 0:
            print(f"  processed {index:,}/{len(words):,} lexical entries", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "wordnet.tsv"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("MCR_WORDNET_2025_PLUS_V1\n")
        for row in sorted(rows):
            stream.write("\t".join(clean(field) for field in row) + "\n")

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "lexicon": LEXICON_ID,
        "license": LICENSE,
        "attribution": (
            "Open English WordNet 2025+, Global WordNet Association; "
            "derived from Princeton WordNet. https://en-word.net/"
        ),
        "entries": len(rows),
        "max_senses_per_lexical_entry": args.max_senses,
        "file": output.name,
        "file_bytes": output.stat().st_size,
        "sha256": digest,
        "runtime_dependencies": [],
    }
    (output_dir / "wordnet_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
