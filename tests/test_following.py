import importlib.util
import unittest
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None
       for name in ("numpy", "torch", "tokenizers", "lm_eval", "language_model_evaluation")):
    raise unittest.SkipTest("Install training and evaluation dependencies")

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain import following
from pretrain.conversation import chat_prompt
from pretrain.model import LanguageModel, ModelConfig


def conversation_tokenizer():
    """A tokenizer of the bytes, BOS, EOS and the reserved tokens that mark a conversation's turns."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=266, show_progress=False,
        special_tokens=["<|bos|>", "<|eos|>", *(f"<|reserved_{index}|>" for index in range(8))],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


ASKED = [{"key": 1, "prompt": "Write two sentences about rain.", "instruction_id_list": [], "kwargs": []},
         {"key": 2, "prompt": "雨について書いてください。", "instruction_id_list": [], "kwargs": []}]


class FollowingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = conversation_tokenizer()
        self.model = LanguageModel(self.tokenizer.get_vocab_size(), ModelConfig(width=8, heads=2, layers=1, context=4096))
        self.asked, self.limits = [], []

        def continue_greedily(prompts, limit, *args, **kwargs):
            self.asked += [self.tokenizer.decode(prompt[1:], skip_special_tokens=False) for prompt in prompts]
            self.limits.append(limit)
            return [self.tokenizer.encode("It rains.").ids for _ in prompts]

        patcher = patch.object(self.model, "continue_greedily", continue_greedily)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_各問題を利用者の発言として会話形式で問い上限まで書かせる(self):
        written = following.replies(self.model, self.tokenizer, torch.device("cpu"), "float32", 16384, ASKED, 2048)
        self.assertEqual(written, {row["prompt"]: "It rains." for row in ASKED})
        self.assertEqual(sorted(self.asked),
                         sorted(chat_prompt([{"role": "user", "content": row["prompt"]}]) for row in ASKED))
        self.assertEqual(set(self.limits), {2048})

    def test_言語ごとに作者のコードで採点し定義の区分にまとめる(self):
        from language_model_evaluation import instructions

        def score(language, replies, work):
            share = {"en": 0.30, "ja": 0.15}[language]
            return {"strict": share, "loose": share + 0.05, "results": {"strict": [], "loose": []}}

        with patch.object(instructions, "code", lambda work: work), \
                patch.object(instructions, "prompts", lambda language, home: ASKED), \
                patch.object(instructions, "score", score):
            result = following.measure(self.model, self.tokenizer, torch.device("cpu"), "float32", "work")
        self.assertEqual(result["definition"], "instructions-ja-en-1")
        self.assertEqual(result["categories"], {"ifeval": 0.30, "ja_ifeval": 0.15})
        self.assertEqual(result["shares"], {"en": {"strict": 0.30, "loose": 0.35}, "ja": {"strict": 0.15, "loose": 0.2}})

    def test_定義と違う版の作者のコードでは測らない(self):
        from language_model_evaluation import definition

        other = definition.load("instructions-ja-en-1")
        other["instructions"] = {**other["instructions"], "commit": "0" * 40}
        with self.assertRaisesRegex(ValueError, "authors' code"):
            following.measure(self.model, self.tokenizer, torch.device("cpu"), "float32", "work", chosen=other)


if __name__ == "__main__":
    unittest.main()
