from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import mosaic_export_wordnet


class FakeHypernym:
    def lemmas(self):
        return ["canine", "domestic animal"]


class FakeSynset:
    def definition(self):
        return "a domesticated canine kept as a pet"

    def lemmas(self):
        return ["dog", "domestic dog", "canis familiaris"]

    def hypernyms(self):
        return [FakeHypernym()]


class FakeWord:
    def lemma(self):
        return "dog"

    def pos(self):
        return "n"

    def synsets(self):
        return [FakeSynset()]


class FakeWordnet:
    def words(self):
        return [FakeWord()]


class WordnetExportTest(unittest.TestCase):
    def test_export_matches_native_tsv_contract(self):
        fake_wn = types.SimpleNamespace(
            download=lambda lexicon_id: self.assertEqual(lexicon_id, "oewn:2025+"),
            Wordnet=lambda lexicon_id: FakeWordnet(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"wn": fake_wn}):
                with patch.object(
                    sys, "argv",
                    ["mosaic_export_wordnet.py", "--output-dir", directory],
                ):
                    mosaic_export_wordnet.main()

            output = Path(directory) / "wordnet.tsv"
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "MCR_WORDNET_2025_PLUS_V1")
            self.assertEqual(len(lines), 2)
            fields = lines[1].split("\t")
            self.assertEqual(fields[0], "dog")
            self.assertEqual(fields[1], "n")
            self.assertIn("domesticated canine", fields[2])
            self.assertIn("domestic dog", fields[3])
            self.assertIn("canine", fields[4])

            manifest = json.loads(
                (Path(directory) / "wordnet_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["lexicon"], "oewn:2025+")
            self.assertEqual(manifest["license"], "CC BY 4.0")
            self.assertEqual(manifest["entries"], 1)
            self.assertEqual(len(manifest["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
