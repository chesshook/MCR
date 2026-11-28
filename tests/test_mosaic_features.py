import unittest

from mosaic_features import FEATURE_DIM, extract_features, stable_hash


class MosaicFeatureTests(unittest.TestCase):
    def test_hash_is_deterministic_and_in_range(self):
        first = stable_hash(b"w:hello")
        self.assertEqual(first, stable_hash(b"w:hello"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, FEATURE_DIM)

    def test_case_normalization_matches(self):
        self.assertEqual(extract_features("HELLO there"), extract_features("hello there"))

    def test_features_include_word_character_and_order_features(self):
        features = extract_features("hello world")
        self.assertGreater(len(features), 2)
        self.assertTrue(all(0 <= index < FEATURE_DIM for index, _ in features))

    def test_empty_text_has_no_features(self):
        self.assertEqual(extract_features(" !!! "), [])


if __name__ == "__main__":
    unittest.main()
