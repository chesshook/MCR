import unittest
from train_intent import features, tokenize, fnv1a


class FeatureTests(unittest.TestCase):
    def test_ascii_case_is_normalized(self):
        self.assertEqual(features(b"HELLO THERE!", 8191), features(b"hello there!", 8191))

    def test_repeatable_and_bounded(self):
        first = features("don't stop".encode("utf-8"), 8191)
        second = features("don't stop".encode("utf-8"), 8191)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(set(first)))
        self.assertTrue(all(0 <= value < 8191 for value in first))

    def test_punctuation_is_a_feature(self):
        self.assertNotEqual(features(b"hello", 8191), features(b"hello?", 8191))

    def test_bucket_range(self):
        self.assertLess(fnv1a(b"sample", 1024), 1024)


if __name__ == "__main__":
    unittest.main()
