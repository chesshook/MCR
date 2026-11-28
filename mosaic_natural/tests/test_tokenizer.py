import unittest

from tokenizer import (
    ASSISTANT_ID, BOS_ID, LearnedTokenizer, PIECE_BASE, TURN_ID, USER_ID,
)


class LearnedTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = LearnedTokenizer.train(
            [
                " hello there!",
                " hello friend.",
                " how are you?",
                " I am fine.",
                " hello there!",
            ],
            vocab_size=512,
            min_piece_frequency=1,
            max_piece_bytes=24,
        )

    def test_round_trip_known_and_unknown_text(self):
        samples = [
            "hello there!",
            "This is an unseen sentence with numbers 2026.",
            "Sanjay's chess-engine 🧠",
            "tabs\tand\nnewlines",
            "",
        ]
        for text in samples:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_ids_reserve_roles_and_bytes(self):
        self.assertEqual((BOS_ID, USER_ID, ASSISTANT_ID, TURN_ID), (1, 2, 3, 4))
        self.assertGreaterEqual(self.tokenizer.vocab_size, PIECE_BASE)
        ids = self.tokenizer.encode("\x00")
        self.assertEqual(ids, [5])

    def test_vocabulary_is_built_from_training_text(self):
        self.assertIn(b" hello", self.tokenizer.to_id)
        self.assertTrue(all(isinstance(piece, bytes) for piece in self.tokenizer.pieces))


if __name__ == "__main__":
    unittest.main()
