import unittest

from mcr.data import normalize_messages, _tokenize_conversation


class FakeTokenizer:
    def __init__(self, with_masks=True):
        self.with_masks = with_masks

    def apply_chat_template(self, messages, **kwargs):
        result = {"input_ids": [10, 11, 12, 13, 14, 15]}
        if self.with_masks:
            result["assistant_masks"] = [0, 0, 0, 1, 1, 1]
        return result


class NormalizeMessagesTests(unittest.TestCase):
    def test_maps_oasst_roles_and_merges_consecutive_turns(self):
        messages = normalize_messages([
            {"role": "prompter", "text": "Hello"},
            {"role": "assistant", "text": "Hi"},
            {"role": "assistant", "text": "How can I help?"},
        ])
        self.assertEqual(messages, [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi\nHow can I help?"},
        ])

    def test_rejects_assistant_only_conversation(self):
        self.assertEqual(
            normalize_messages([{"role": "assistant", "content": "Hello"}]), []
        )

    def test_rejects_unanswered_final_user_turn(self):
        self.assertEqual(normalize_messages([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "One more thing"},
        ]), [])

    def test_handles_standard_chat_schema_and_preserves_newlines(self):
        messages = normalize_messages([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Explain a river bank.\nUse an example."},
            {"role": "assistant", "content": "The land beside a river."},
        ])
        self.assertEqual([x["role"] for x in messages], ["system", "user", "assistant"])
        self.assertIn("\n", messages[1]["content"])


class AssistantMaskTests(unittest.TestCase):
    def test_masks_user_tokens_from_loss_and_counts_assistant_targets(self):
        item = _tokenize_conversation(
            FakeTokenizer(),
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            max_length=20,
        )
        self.assertEqual(item["labels"], [-100, -100, -100, 13, 14, 15])
        self.assertEqual(item["assistant_label_tokens"], 3)

    def test_refuses_to_train_without_assistant_masks(self):
        with self.assertRaises(RuntimeError):
            _tokenize_conversation(
                FakeTokenizer(with_masks=False),
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                max_length=20,
            )

    def test_truncation_keeps_alignment_of_assistant_mask(self):
        item = _tokenize_conversation(
            FakeTokenizer(),
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            max_length=4,
        )
        self.assertEqual(len(item["input_ids"]), 4)
        self.assertEqual(len(item["labels"]), 4)
        self.assertEqual(item["assistant_label_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
