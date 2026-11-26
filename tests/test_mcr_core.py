import unittest

from mcr.data import normalize_messages, messages_to_pairs
from mcr.encoding import encode_bytes, pad_sequences
from mcr.schedule import make_task_schedule


class ConversationNormalizationTests(unittest.TestCase):
    def test_normalizes_user_assistant_aliases_and_merges_adjacent_turns(self):
        got = normalize_messages([
            {"role": "prompter", "text": "Hello"},
            {"role": "assistant", "text": "Hi"},
            {"role": "assistant", "text": "How can I help?"},
        ])
        self.assertEqual(got, [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi\nHow can I help?"},
        ])

    def test_pairs_keep_prior_context_and_target_the_assistant_reply(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is a river bank?"},
            {"role": "assistant", "content": "It is the land beside a river."},
        ]
        pairs = messages_to_pairs(messages)
        self.assertEqual(len(pairs), 2)
        self.assertIn("CURRENT USER: What is a river bank?", pairs[1]["context"])
        self.assertEqual(pairs[1]["reply"], "It is the land beside a river.")

    def test_rejects_non_message_payload(self):
        self.assertEqual(normalize_messages(None), [])


class ByteEncodingTests(unittest.TestCase):
    def test_utf8_bytes_are_offset_and_zero_is_reserved_for_padding(self):
        values = encode_bytes("Aé")
        self.assertEqual(values, [66, 196, 170])
        self.assertNotIn(0, values)
        self.assertTrue(all(1 <= x <= 256 for x in values))

    def test_long_text_preserves_beginning_and_end(self):
        text = "abcdefghij" * 10
        values = encode_bytes(text, max_bytes=12)
        self.assertEqual(len(values), 12)
        self.assertEqual(values[:8], [ord(c) + 1 for c in "abcdefgh"])
        self.assertEqual(values[-4:], [ord(c) + 1 for c in "ghij"])

    def test_padding_masks_match_lengths(self):
        ids, masks, lengths = pad_sequences([[2, 3], [4]])
        self.assertEqual(ids, [[2, 3], [4, 0]])
        self.assertEqual(masks, [[1, 1], [1, 0]])
        self.assertEqual(lengths, [2, 1])

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            pad_sequences([])

    def test_model_has_expected_non_autoregressive_outputs(self):
        try:
            import torch
            from mcr.model import MCRModel, contrastive_loss
        except ImportError:
            self.skipTest("PyTorch is not installed in this offline test environment")
        model = MCRModel(embedding_dim=16, conv_channels=24, recurrent_hidden=16,
                         output_dim=32, dropout=0.0)
        left = torch.randint(1, 257, (2, 24))
        right = torch.randint(1, 257, (2, 20))
        left_mask = torch.ones_like(left, dtype=torch.bool)
        right_mask = torch.ones_like(right, dtype=torch.bool)
        query, reply = model(task="chat", context_ids=left, context_mask=left_mask,
                             reply_ids=right, reply_mask=right_mask)
        self.assertEqual(tuple(query.shape), (2, 32))
        self.assertEqual(tuple(reply.shape), (2, 32))
        loss = contrastive_loss(query, reply)
        self.assertTrue(torch.isfinite(loss).item())
        logits = model(task="nli", premise_ids=left, premise_mask=left_mask,
                       hypothesis_ids=right, hypothesis_mask=right_mask)
        self.assertEqual(tuple(logits.shape), (2, 3))


class TrainingScheduleTests(unittest.TestCase):
    def test_schedule_consumes_each_batch_once(self):
        schedule = make_task_schedule(7, 13, 99)
        self.assertEqual(schedule.count("chat"), 7)
        self.assertEqual(schedule.count("nli"), 13)
        self.assertEqual(len(schedule), 20)

    def test_schedule_is_reproducible(self):
        self.assertEqual(make_task_schedule(5, 9, 17), make_task_schedule(5, 9, 17))

class LauncherPathTests(unittest.TestCase):
    def test_kaggle_launcher_points_to_the_packaged_trainer(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "mcr" / "train.py").is_file())
        launcher = (root / "launch_kaggle.py").read_text(encoding="utf-8")
        self.assertIn('"--module", "mcr.train"', launcher)


if __name__ == "__main__":
    unittest.main()
