import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers", "ahocorasick")):
    raise unittest.SkipTest("Install training and corpus dependencies")

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.data import file_digest
from pretrain.model import LanguageModel, ModelConfig
from pretrain.preference import (
    PreferenceConfig,
    PreferenceCorpus,
    encode_reply,
    preference_loss,
    prepare_preferences,
    train_preferences,
)
from pretrain.training import atomic_save, load_model


# The special tokens, six reserved tokens that mark conversations, and bytes.
VOCABULARY = 264


def byte_tokenizer():
    """A tokenizer whose vocabulary holds only the special and reserved tokens and bytes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCABULARY,
        show_progress=False,
        special_tokens=["<|bos|>", "<|eos|>", *(f"<|reserved_{index}|>" for index in range(6))],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


def pair(question, chosen, rejected):
    return {"prompt": [{"role": "user", "content": question}], "chosen": chosen, "rejected": rejected}


class PreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = byte_tokenizer()
        self.tokenizer.save(str(self.root / "tokenizer.json"))
        benchmarks = self.root / "benchmarks"
        benchmarks.mkdir()
        self.question = "Describe rain without using commas."
        (benchmarks / "ifeval.jsonl").write_text(json.dumps({"id": "1", "context": self.question}) + "\n")
        (benchmarks / "sources.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "tasks": {
                        "ifeval": {
                            "language": "en",
                            "filename": "ifeval.jsonl",
                            "sha256": file_digest(benchmarks / "ifeval.jsonl"),
                        }
                    },
                }
            )
        )
        self.benchmarks = benchmarks

    def write_source(self, pairs_by_language):
        source = self.root / "source"
        source.mkdir()
        for language, pairs in pairs_by_language.items():
            (source / f"{language}.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in pairs), encoding="utf-8"
            )
        return source

    def toy_pairs(self, count):
        return {
            "ja": [pair(f"質問{index}は？", "はい、そうです。", "いいえ。") for index in range(count)],
            "en": [pair(f"Question {index}?", "Yes, it is.", "No.") for index in range(count)],
        }

    def checkpoint(self, config):
        torch.manual_seed(0)
        model = LanguageModel(VOCABULARY, config)
        path = self.root / "start.pt"
        atomic_save(
            {
                "version": 1,
                "model_config": asdict(config),
                "model": model.state_dict(),
                "tokenizer": self.tokenizer.to_str(),
                "step": 0,
            },
            path,
        )
        return path

    def test_最後の回答とその終わりの記号だけを学習の対象にする(self):
        """会話の最後の回答とその終わりの記号だけを学習の対象にし、それより前の発言は対象にしない。"""
        turns = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "See you"},
        ]
        tokens, mask = encode_reply(self.tokenizer, turns)
        learned = self.tokenizer.decode([int(t) for t, m in zip(tokens, mask) if m], skip_special_tokens=True)
        self.assertEqual(learned, "See you")
        self.assertEqual(tokens[-1], self.tokenizer.token_to_id("<|eos|>"))
        self.assertEqual(mask[-1], 1)

    def test_重複や評価の問題や長すぎる組を除いて組を用意する(self):
        """重複した組、評価の問題を含む組、文脈に収まらない組、二つの回答が同じ組を除き、残りを検証用と学習用に分ける。"""
        pairs = self.toy_pairs(6)
        pairs["en"] += [
            pairs["en"][0],
            pair(self.question, "Soft drops fall.", "It rains."),
            pair("Long?", "x" * 200, "No."),
            pair("Same?", "Yes.", "Yes."),
        ]
        manifest = prepare_preferences(
            self.write_source(pairs), self.root / "pairs", self.root / "tokenizer.json",
            context=64, validation_pairs=2, benchmarks=self.benchmarks,
        )
        english = manifest["pairs"]["en"]
        self.assertEqual(
            {key: english[key] for key in ("train", "validation", "duplicates", "contaminated", "too_long", "malformed")},
            {"train": 4, "validation": 2, "duplicates": 1, "contaminated": 1, "too_long": 1, "malformed": 1},
        )
        corpus = PreferenceCorpus(self.root / "pairs")
        inputs, targets = corpus.load("train", corpus.pick("train", 0, 3, 42), torch.device("cpu"))
        # Three chosen replies, then the three rejected ones.
        self.assertEqual(tuple(inputs.shape), (6, 64))
        self.assertTrue(all((row != -100).sum() > 0 for row in targets))

    def test_学習の始めは基準と同じなので損失がlog2になる(self):
        """学習を始める時点の方策は基準と同じなので、どの組でも損失は log 2、差は 0 になる。"""
        prepare_preferences(
            self.write_source(self.toy_pairs(4)), self.root / "pairs", self.root / "tokenizer.json",
            context=64, validation_pairs=1,
        )
        corpus = PreferenceCorpus(self.root / "pairs")
        torch.manual_seed(0)
        policy = LanguageModel(VOCABULARY, ModelConfig(width=24, heads=4, layers=1, context=64, architecture="llama"))
        inputs, targets = corpus.load("train", corpus.pick("train", 0, 4, 42), torch.device("cpu"))
        for length_normalized in (False, True):
            config = PreferenceConfig(steps=2, warmup_steps=0, length_normalized=length_normalized)
            loss, accuracy, margin = preference_loss(policy, policy, inputs, targets, config)
            self.assertAlmostEqual(loss.item(), math.log(2), places=5)
            self.assertAlmostEqual(margin.item(), 0.0, places=5)

    def test_好ましい回答を選ぶように学び学習と同じ形式で保存する(self):
        """好ましい回答の確率が基準より上がるように学び、評価や配信が読む形式で保存する。"""
        prepare_preferences(
            self.write_source(self.toy_pairs(8)), self.root / "pairs", self.root / "tokenizer.json",
            context=64, validation_pairs=2,
        )
        start = self.checkpoint(ModelConfig(width=24, heads=4, layers=1, context=64, architecture="llama"))
        config = PreferenceConfig(
            steps=30, batch_size=4, accumulation_steps=1, learning_rate=3e-3, warmup_steps=2, eval_every=30
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train_preferences(self.root / "pairs", self.root / "dpo", start, config, torch.device("cpu"))
        self.assertAlmostEqual(result["initial_validation"]["loss"], math.log(2), places=4)
        self.assertLess(result["validation"]["loss"], math.log(2) * 0.5)
        self.assertEqual(result["validation"]["accuracy"], 1.0)
        model, tokenizer, saved = load_model(self.root / "dpo/last.pt", torch.device("cpu"))
        self.assertEqual(saved["initialization"]["method"], "dpo")
        self.assertEqual(saved["step"], 30)
        self.assertEqual(tokenizer.to_str(), self.tokenizer.to_str())
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            PreferenceConfig(steps=2, warmup_steps=2)

    def test_組の用意と学習をコマンドから行う(self):
        """組の用意と DPO の学習をコマンドから行い、設定ファイルの値で学習する。"""
        from pretrain.__main__ import main

        source = self.write_source(self.toy_pairs(6))
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            main(["prepare-preferences", "--source", str(source), "--tokenizer", str(self.root / "tokenizer.json"),
                  "--output", str(self.root / "pairs"), "--context", "64", "--validation-pairs", "2"])
        self.assertEqual(json.loads(shown.getvalue())["ja"]["train"], 4)
        start = self.checkpoint(ModelConfig(width=24, heads=4, layers=1, context=64, architecture="llama"))
        config = self.root / "dpo.toml"
        config.write_text("[preference]\nsteps = 2\nbatch_size = 2\naccumulation_steps = 1\nwarmup_steps = 1\n")
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            main(["train-preferences", "--config", str(config), "--data", str(self.root / "pairs"),
                  "--initialize-from", str(start), "--output", str(self.root / "dpo"), "--device", "cpu"])
        self.assertEqual(json.loads(shown.getvalue())["step"], 2)
        self.assertTrue((self.root / "dpo/last.pt").exists())


if __name__ == "__main__":
    unittest.main()
