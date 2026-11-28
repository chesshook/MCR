#!/usr/bin/env python3
"""Download and normalize CLINC150 into label<TAB>text files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
OUT = Path("data/clinc150")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "MCR-CPU-Native/2.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))

    def rows(split: str):
        result = []
        for pair in data.get(split, []):
            if isinstance(pair, list) and len(pair) >= 2:
                text, label = str(pair[0]).strip(), str(pair[1]).strip()
                if text and label:
                    result.append((label, text))
        oos_key = {"train": "oos_train", "val": "oos_val", "test": "oos_test"}[split]
        for pair in data.get(oos_key, []):
            if isinstance(pair, list) and pair:
                text = str(pair[0]).strip()
                if text:
                    result.append(("__out_of_scope__", text))
        return result

    counts = {}
    for split, filename in (("train", "train.tsv"), ("val", "valid.tsv"), ("test", "test.tsv")):
        items = rows(split)
        path = OUT / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for label, text in items:
                handle.write(label.replace("\t", " ") + "\t" + text.replace("\t", " ").replace("\n", " ") + "\n")
        counts[split] = len(items)

    manifest = {
        "dataset": "CLINC150 / OOS-Eval full split",
        "source": URL,
        "sha256": digest,
        "license": "CC BY 4.0 (UCI Machine Learning Repository dataset record)",
        "attribution": "Larson et al., EMNLP-IJCNLP 2019, An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
        "counts_including_oos": counts,
        "note": "Single-intent English classification benchmark. It does not train open-domain dialogue or response generation. Verify source terms before redistribution.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
