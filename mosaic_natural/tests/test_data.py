import unittest

from data import _normalise_messages, _oasst_split_to_conversations, make_response_examples
from tokenizer import LearnedTokenizer, ASSISTANT_ID, BOS_ID, TURN_ID, USER_ID


class ConversationDataTests(unittest.TestCase):
    def test_normalizes_roles_and_merges_adjacent_messages(self):
        result = _normalise_messages([
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "there"},
            {"role": "assistant", "content": "hi"},
        ])
        self.assertEqual(result, [("user", "hello\nthere"), ("assistant", "hi")])

    def test_reconstructs_parent_linked_oasst_paths(self):
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "lang": "en", "text": "Hello"},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "lang": "en", "text": "Hi"},
            {"message_id": "u2", "parent_id": "a1", "role": "prompter", "lang": "en", "text": "How are you?"},
            {"message_id": "a2", "parent_id": "u2", "role": "assistant", "lang": "en", "text": "I am well."},
            {"message_id": "a3", "parent_id": "u1", "role": "assistant", "lang": "es", "text": "Hola"},
        ]
        conversations = _oasst_split_to_conversations(rows)
        self.assertEqual(len(conversations), 2)
        self.assertEqual(conversations[-1][-1], ("assistant", "I am well."))

    def test_target_examples_include_only_assistant_turns(self):
        tokenizer = LearnedTokenizer.train(
            ["hello", "hi", "how are you", "I am fine"],
            vocab_size=400, min_piece_frequency=1,
        )
        conversations = [[
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", "how are you"),
            ("assistant", "I am fine"),
        ]]
        examples = make_response_examples(
            conversations, tokenizer, max_context=64, max_target=32,
        )
        self.assertEqual(len(examples), 2)
        for sequence, target_start in examples:
            self.assertEqual(sequence[0], BOS_ID)
            self.assertEqual(sequence[target_start - 1], ASSISTANT_ID)
            self.assertEqual(sequence[-1], TURN_ID)
            self.assertGreaterEqual(target_start, 1)
            self.assertIn(USER_ID, sequence)


if __name__ == "__main__":
    unittest.main()
