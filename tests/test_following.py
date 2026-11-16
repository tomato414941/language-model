import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None
       for name in ("numpy", "torch", "tokenizers", "language_model_evaluation")):
    raise unittest.SkipTest("Install training and evaluation dependencies")

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain import following
from pretrain.__main__ import main
from pretrain.comparison import compare_results
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

    def test_言語ごとに作者のコードで採点し定義の区分にまとめ偶然の水準も添える(self):
        from language_model_evaluation import instructions

        def score(language, replies, work):
            # in English the first prompt is followed, in Japanese none
            rows = [{"follow_all_instructions": language == "en" and at == 0} for at in range(len(replies))]
            share = sum(row["follow_all_instructions"] for row in rows) / len(rows)
            return {"strict": share, "loose": share, "results": {"strict": rows, "loose": rows}}

        def chance(language, replies, work):
            return {level: [0.25] * len(replies) for level in instructions.LEVELS}

        with patch.object(instructions, "code", lambda work: work), \
                patch.object(instructions, "prompts", lambda language, home: ASKED), \
                patch.object(instructions, "score", score), patch.object(following, "chance", chance):
            result = following.measure(self.model, self.tokenizer, torch.device("cpu"), "float32", "work")
        self.assertEqual(result["definition"], "instructions-ja-en-1")
        self.assertEqual(result["categories"], {"ifeval": 0.5, "ja_ifeval": 0.0})
        self.assertEqual(result["shares"], {"en": {"strict": 0.5, "loose": 0.5}, "ja": {"strict": 0.0, "loose": 0.0}})
        self.assertEqual(result["items"], {"prompts": {"en": [1, 0], "ja": [0, 0]}})
        self.assertEqual(result["chance"], {"categories": {"ifeval": 0.25, "ja_ifeval": 0.25},
                                            "items": {"prompts": {"en": [0.25, 0.25], "ja": [0.25, 0.25]}}})

    def test_返答を他の問題に付け替え各問題に別の返答を1つずつ渡す(self):
        asked = [{"prompt": text} for text in ("雨は？", "雪は？", "風は？", "霧は？")]
        replies = {row["prompt"]: row["prompt"][0] for row in asked}  # each reply names its prompt's weather
        for seed in range(5):
            given = following.others(asked, replies, seed)
            self.assertEqual(sorted(given), sorted(replies))
            self.assertEqual(sorted(given.values()), sorted(replies.values()))
            self.assertTrue(all(given[prompt] != replies[prompt] for prompt in given))

    def test_偶然の水準は他の問題への返答が各問題の指示に従う割合(self):
        from language_model_evaluation import instructions

        asked = [{"prompt": text} for text in ("雨は？", "雪は？", "風は？", "霧は？")]
        replies = {row["prompt"]: row["prompt"][0] for row in asked}

        def evaluated(asked, replies, directory, home, python):
            # a prompt is followed by the reply about rain, and by no other
            rows = [{"follow_all_instructions": replies[row["prompt"]] == "雨"} for row in asked]
            return {level: rows for level in instructions.LEVELS}

        with patch.object(instructions, "code", lambda work: work), \
                patch.object(instructions, "environment", lambda work: None), \
                patch.object(instructions, "prompts", lambda language, home: asked), \
                patch.object(instructions, "evaluated", evaluated):
            found = following.chance("ja", replies, "work", times=8)
        for level in instructions.LEVELS:
            self.assertEqual(found[level][0], 0)  # the reply about rain is the rain prompt's own
            self.assertEqual(sum(found[level]), 1)  # and is given to one other prompt each time

    def test_偶然の水準と比べ指示に従った分が読める差かを示す(self):
        result = {"definition": "instructions-ja-en-1", "categories": {"ifeval": 0.5, "ja_ifeval": 0.12},
                  "items": {"prompts": {"en": [1, 0] * 100, "ja": [1] * 24 + [0] * 176}},
                  "chance": {"categories": {"ifeval": 0.1, "ja_ifeval": 0.1},
                             "items": {"prompts": {"en": [0.1] * 200, "ja": [0.1] * 200}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instructions.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main(["improvement", "--other", str(path), "--chance"])
        found = json.loads(printed.getvalue())
        self.assertEqual(found, compare_results({"definition": "instructions-ja-en-1", **result["chance"]}, result))
        self.assertAlmostEqual(found["improvement"], ((0.5 / 0.1 - 1) + (0.12 / 0.1 - 1)) / 2)
        self.assertTrue(found["categories"]["ifeval"]["readable"])  # half the prompts against a tenth
        self.assertFalse(found["categories"]["ja_ifeval"]["readable"])  # 24 of 200 against a tenth

    def test_定義と違う版の作者のコードでは測らない(self):
        from language_model_evaluation import definition

        other = definition.load("instructions-ja-en-1")
        other["instructions"] = {**other["instructions"], "commit": "0" * 40}
        with self.assertRaisesRegex(ValueError, "authors' code"):
            following.measure(self.model, self.tokenizer, torch.device("cpu"), "float32", "work", chosen=other)


if __name__ == "__main__":
    unittest.main()
