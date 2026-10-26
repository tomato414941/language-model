import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")):
    raise unittest.SkipTest("Install training dependencies")

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.conversation import THINK_END, TURN_END
from pretrain.model import LanguageModel, ModelConfig
from pretrain.reinforcement import (
    ReinforcementConfig,
    advantages,
    code_reward,
    math_reward,
    policy_loss,
    reward,
    train_reinforcement,
)
from pretrain.training import atomic_save, load_model

# The special tokens, eight reserved tokens that mark conversations, and bytes.
VOCABULARY = 266


def byte_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCABULARY,
        show_progress=False,
        special_tokens=["<|bos|>", "<|eos|>", *(f"<|reserved_{index}|>" for index in range(8))],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


class ReinforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = byte_tokenizer()

    def checkpoint(self):
        torch.manual_seed(0)
        config = ModelConfig(width=24, heads=4, layers=1, context=64, architecture="llama", document_start=0)
        model = LanguageModel(VOCABULARY, config)
        path = self.root / "start.pt"
        atomic_save(
            {"version": 1, "model_config": asdict(config), "model": model.state_dict(),
             "tokenizer": self.tokenizer.to_str(), "step": 0},
            path,
        )
        return path

    def test_数学の答えは最後の囲みか最後の数を参照と数として比べる(self):
        """数学の答えは、最後の \\boxed{} の中身か最後の数を取り出し、参照と数として等しいときに報酬を与える。"""
        self.assertEqual(math_reward("途中で 3 を足して \\boxed{1,234} です。", "1234"), 1.0)
        self.assertEqual(math_reward("半分なので 3/6", "1/2"), 1.0)
        self.assertEqual(math_reward("答えは 41", "42"), 0.0)
        self.assertEqual(math_reward("わかりません", "42"), 0.0)
        self.assertEqual(math_reward("\\boxed{x^{2}}", "x^{2}"), 1.0)

    def test_コードはテストを通ったときだけ報酬を与える(self):
        """回答の最後の Python のコードがテストを通ったときだけ報酬を与える。"""
        passing = "考えたとおりです。\n```python\ndef f(x):\n    return x + 1\n```"
        self.assertEqual(code_reward(passing, "assert f(1) == 2"), 1.0)
        self.assertEqual(code_reward("def f(x):\n    return x", "assert f(1) == 2"), 0.0)
        self.assertEqual(code_reward("def f(x):\n    raise ValueError", "assert f(1) == 2"), 0.0)

    def test_考える過程を閉じないか終わらない回答には報酬を与えない(self):
        """考える過程を閉じない回答や、上限まで終わらなかった回答には、答えが合っていても報酬を与えない。"""
        verifier = {"type": "math", "answer": "4"}
        self.assertEqual(reward("2 と 2 で" + THINK_END + "答えは 4", verifier, True, True), 1.0)
        self.assertEqual(reward("2 と 2 で答えは 4", verifier, True, True), 0.0)
        self.assertEqual(reward("2 と 2 で" + THINK_END + "答えは 4", verifier, True, False), 0.0)
        self.assertEqual(reward("答えは 4", verifier, False, True), 1.0)

    def test_同じ組の平均との差を利点とし全員同じ点の組は除く(self):
        """利点は同じ組の報酬の平均との差で、全員が同じ点だった組は学習から除く。"""
        centered, informative = advantages([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3)
        torch.testing.assert_close(centered, torch.tensor([2 / 3, -1 / 3, -1 / 3, 0.0, 0.0, 0.0]))
        self.assertEqual(informative.tolist(), [True, True, True, False, False, False])

    def test_利点が正の回答の確率を上げ負の回答の確率を下げる(self):
        """利点が正の回答の確率を上げ、負の回答の確率を下げるように重みを更新する。"""
        model, _, _ = load_model(self.checkpoint(), torch.device("cpu"))
        prompt = [0, 10, 11]
        better, worse = prompt + [20, 21], prompt + [30, 31]

        def score(sequence):
            with torch.no_grad():
                logits = model(torch.tensor([sequence[:-1]]))[0].log_softmax(-1)
            return sum(logits[position - 1, sequence[position]].item() for position in range(len(prompt), len(sequence)))

        before = (score(better), score(worse))
        config = ReinforcementConfig(steps=2, warmup_steps=0, max_reply_tokens=4, micro_batch=1, group_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        policy_loss(model, [better, worse], [3, 3], torch.tensor([1.0, -1.0]), config, torch.device("cpu"), "float32")
        optimizer.step()
        self.assertGreater(score(better), before[0])
        self.assertLess(score(worse), before[1])

    def test_検証できる問題で学び学習と同じ形式で保存する(self):
        """検証できる問題で学習し、評価や配信が読む形式で、強化学習の記録とともに保存する。"""
        prompts = self.root / "prompts.jsonl"
        prompts.write_text(
            "".join(
                json.dumps({"messages": [{"role": "user", "content": f"{a}+{a}?"}],
                            "verifier": {"type": "math", "answer": str(2 * a)}}) + "\n"
                for a in range(4)
            )
        )
        config = ReinforcementConfig(
            steps=2, prompts_per_step=2, group_size=2, max_reply_tokens=4, warmup_steps=0, learning_rate=1e-3
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_reinforcement(prompts, self.root / "rl", self.checkpoint(), config, torch.device("cpu"))
        self.assertEqual(result["step"], 2)
        model, tokenizer, saved = load_model(self.root / "rl/last.pt", torch.device("cpu"))
        self.assertEqual(saved["initialization"]["method"], "grpo")
        self.assertEqual(tokenizer.to_str(), self.tokenizer.to_str())
        log = [json.loads(line) for line in (self.root / "rl/log.jsonl").read_text().splitlines()]
        self.assertEqual([line["step"] for line in log], [1, 2])
        self.assertTrue(all(0 <= line["reward"] <= 1 for line in log))
        with self.assertRaisesRegex(ValueError, "group_size"):
            ReinforcementConfig(group_size=1)

    def test_強化学習をコマンドから行う(self):
        """設定ファイルの値で、強化学習をコマンドから行う。"""
        from pretrain.__main__ import main

        prompts = self.root / "prompts.jsonl"
        prompts.write_text(
            json.dumps({"messages": [{"role": "user", "content": "1+1?"}], "verifier": {"type": "math", "answer": "2"}}) + "\n"
        )
        config = self.root / "rl.toml"
        config.write_text(
            "[reinforcement]\nsteps = 1\nprompts_per_step = 1\ngroup_size = 2\nmax_reply_tokens = 4\nwarmup_steps = 0\n"
        )
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            main(["train-reinforcement", "--config", str(config), "--prompts", str(prompts),
                  "--initialize-from", str(self.checkpoint()), "--output", str(self.root / "rl"), "--device", "cpu"])
        self.assertEqual(json.loads(shown.getvalue())["step"], 1)
        self.assertTrue((self.root / "rl/last.pt").exists())

    def test_確かめ方のない問題を断る(self):
        """答えの確かめ方を持たない問題や、利用者の発言で終わらない会話を読み込まない。"""
        prompts = self.root / "prompts.jsonl"
        prompts.write_text(json.dumps({"messages": [{"role": "user", "content": "?"}], "verifier": {"type": "math"}}) + "\n")
        with self.assertRaisesRegex(ValueError, "math answer or code tests"):
            train_reinforcement(prompts, self.root / "rl", self.checkpoint(), ReinforcementConfig(), torch.device("cpu"))
        prompts.write_text(
            json.dumps({"messages": [{"role": "assistant", "content": "?"}], "verifier": {"type": "math", "answer": "1"}}) + "\n"
        )
        with self.assertRaisesRegex(ValueError, "end with the user"):
            train_reinforcement(prompts, self.root / "rl", self.checkpoint(), ReinforcementConfig(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
