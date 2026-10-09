import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")
):
    raise unittest.SkipTest(
        "Install training dependencies with uv sync --extra train-cpu"
    )

import numpy as np
import torch

from pretrain.data import Corpus, prepare_data
from pretrain.model import LanguageModel, ModelConfig
from pretrain.training import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    load_model,
    select_device,
    train,
)


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.japanese = self.root / "ja.jsonl"
        self.english = self.root / "en.jsonl"
        self.documents = {
            "ja": ["あいう" * 30 + f" 終わり{i}" for i in range(12)],
            "en": ["abc " * 40 + f" ending{i}" for i in range(12)],
        }
        for language, path in (("ja", self.japanese), ("en", self.english)):
            with path.open("w", encoding="utf-8") as destination:
                for text in self.documents[language] + [self.documents[language][0]]:
                    destination.write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    )
        self.data = self.root / "data"
        prepare_data(
            self.japanese,
            self.english,
            self.data,
            vocabulary_size=258,
            validation_fraction=0.25,
        )
        self.corpus = Corpus(self.data)
        self.model_config = ModelConfig(width=24, heads=4, layers=1, context=16)

    def config(self, steps=8):
        return TrainConfig(
            steps=steps,
            batch_size=4,
            accumulation_steps=2,
            learning_rate=0.01,
            min_learning_rate=0.001,
            warmup_steps=1,
            eval_every=4,
            save_every=4,
            eval_batches=2,
            log_every=4,
        )

    def run_training(self, name, config=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(
                self.data,
                self.root / name,
                self.model_config,
                config or self.config(),
                torch.device("cpu"),
                **kwargs,
            )

    def test_byte_tokenizer_round_trips_unseen_japanese_and_english(self):
        """学習にない漢字・絵文字・英単語も元の文章に復元する。"""
        text = "初めての漢字と🌱。An Unseen English Sentence!\n改行も維持する。"
        encoded = self.corpus.tokenizer.encode(text)
        self.assertEqual(self.corpus.tokenizer.decode(encoded.ids), text)

    def test_preparation_deduplicates_documents_and_holds_out_each_language(self):
        """日英それぞれの文書を重複除去して学習用と検証用に分割する。"""
        self.assertEqual(self.corpus.manifest["duplicates_removed"], 2)
        eos = self.corpus.tokenizer.token_to_id("<|eos|>")
        for language in ("ja", "en"):
            splits = {}
            for split in ("train", "validation"):
                stream = self.corpus.streams[split][language]
                boundaries = np.flatnonzero(stream == eos)
                decoded, start = set(), 0
                for end in boundaries:
                    decoded.add(
                        self.corpus.tokenizer.decode(stream[start : end + 1].tolist())
                    )
                    start = end + 1
                splits[split] = decoded
            self.assertTrue(splits["train"])
            self.assertTrue(splits["validation"])
            self.assertTrue(splits["train"].isdisjoint(splits["validation"]))
            self.assertEqual(
                splits["train"] | splits["validation"], set(self.documents[language])
            )

    def test_prediction_uses_only_the_preceding_tokens(self):
        """未来の入力が変わっても、それより前の次トークン予測を維持する。"""
        model = LanguageModel(258, self.model_config).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 200, 201]])
        torch.testing.assert_close(
            model(first)[:, :3], model(second)[:, :3], rtol=0, atol=1e-6
        )

    def test_training_improves_held_out_predictions_in_both_languages(self):
        """日英の繰り返しパターンを学習して、両言語の検証損失を下げる。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        before = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        after = self.run_training("learn", self.config(steps=24))["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_resume_continues_the_same_training_sequence(self):
        """途中保存から再開しても、連続して学習したときと同じ重み・学習量を得る。"""
        self.run_training("continuous")
        partial = self.run_training("resumed", max_steps=3)
        self.assertEqual(partial["step"], 3)
        self.run_training("resumed", resume=self.root / "resumed" / "last.pt")
        continuous = load_checkpoint(self.root / "continuous" / "last.pt")
        resumed = load_checkpoint(self.root / "resumed" / "last.pt")
        self.assertEqual(resumed["step"], 8)
        self.assertEqual(resumed["tokens_seen"], 8 * 4 * 2 * 16)
        self.assertEqual(continuous["validation"], resumed["validation"])
        for name, parameter in continuous["model"].items():
            torch.testing.assert_close(
                parameter, resumed["model"][name], rtol=0, atol=0
            )

    def test_checkpoint_restores_predictions_and_seeded_long_prompt_generation(self):
        """保存した重みと語彙を復元し、長い日英の接頭辞を保って再現可能に生成する。"""
        self.run_training("generation", max_steps=2)
        path = self.root / "generation" / "last.pt"
        first, tokenizer, _ = load_model(path, torch.device("cpu"))
        second, restored_tokenizer, _ = load_model(path, torch.device("cpu"))
        inputs = torch.tensor([[10, 20, 30]])
        torch.testing.assert_close(first(inputs), second(inputs), rtol=0, atol=0)
        prompt = "日本語とEnglishの長い文章です。" * 5
        output = first.generate(tokenizer, prompt, max_new_tokens=12, seed=9)
        self.assertTrue(output.startswith(prompt))
        self.assertEqual(
            output,
            second.generate(restored_tokenizer, prompt, max_new_tokens=12, seed=9),
        )

    def test_rotary_grouped_attention_learns_both_languages_causally(self):
        self.model_config = ModelConfig(
            width=24,
            heads=4,
            kv_heads=2,
            layers=2,
            context=16,
            architecture="llama",
            intermediate_size=64,
        )
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 200, 201]])
        torch.testing.assert_close(
            model(first)[:, :3], model(second)[:, :3], rtol=0, atol=1e-6
        )
        before = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        after = self.run_training("modern", self.config(steps=24))["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_rotary_model_resumes_the_same_updates_and_restores_generation(self):
        self.model_config = ModelConfig(
            width=24,
            heads=4,
            kv_heads=2,
            layers=2,
            context=16,
            architecture="llama",
            intermediate_size=64,
        )
        self.run_training("modern-continuous")
        self.run_training("modern-resumed", max_steps=3)
        self.run_training("modern-resumed", resume=self.root / "modern-resumed/last.pt")
        continuous, tokenizer, first = load_model(
            self.root / "modern-continuous/last.pt", torch.device("cpu")
        )
        resumed, _, second = load_model(
            self.root / "modern-resumed/last.pt", torch.device("cpu")
        )
        self.assertEqual(first["tokens_seen"], second["tokens_seen"])
        for name, parameter in continuous.state_dict().items():
            torch.testing.assert_close(
                parameter, resumed.state_dict()[name], rtol=0, atol=0
            )
        prompt = "日本語とEnglish"
        self.assertEqual(
            continuous.generate(tokenizer, prompt, max_new_tokens=8, seed=7),
            resumed.generate(tokenizer, prompt, max_new_tokens=8, seed=7),
        )

    def test_new_training_phase_keeps_learned_predictions_on_additional_data(self):
        self.run_training("phase-one", max_steps=3)
        source = self.root / "phase-one/last.pt"
        old = load_checkpoint(source)
        new_data = self.root / "additional-data"
        prepare_data(
            self.japanese,
            self.english,
            new_data,
            vocabulary_size=258,
            seed=99,
            tokenizer_path=self.data / "tokenizer.json",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                new_data,
                self.root / "phase-two",
                self.model_config,
                self.config(steps=12),
                torch.device("cpu"),
                initialize_from=source,
                stop_requested=lambda: True,
            )
        initialized = load_checkpoint(self.root / "phase-two/last.pt")
        self.assertEqual(initialized["step"], 0)
        self.assertEqual(
            initialized["initialization"]["source_tokens_seen"], old["tokens_seen"]
        )
        self.assertEqual(
            initialized["dataset_fingerprint"], Corpus(new_data).fingerprint
        )
        for name, parameter in old["model"].items():
            torch.testing.assert_close(
                parameter, initialized["model"][name], rtol=0, atol=0
            )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                new_data,
                self.root / "phase-two",
                self.model_config,
                self.config(steps=12),
                torch.device("cpu"),
                resume=self.root / "phase-two/last.pt",
                max_steps=2,
            )
        self.assertEqual(result["step"], 2)
        self.assertEqual(
            Corpus(new_data).tokenizer.to_str(), self.corpus.tokenizer.to_str()
        )

    def test_prepared_data_reports_corruption_before_training(self):
        """改変されたトークンファイルを検出して、データの破損を通知する。"""
        path = self.data / "train_ja.bin"
        with path.open("r+b") as source:
            source.write(b"\xff\xff\xff\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Corpus(self.data)

    def test_resume_validates_the_original_dataset(self):
        """別のデータを指定した再開には、元のデータが必要なことを通知する。"""
        self.run_training("original", max_steps=2)
        other_data = self.root / "other"
        prepare_data(
            self.japanese, self.english, other_data, vocabulary_size=258, seed=99
        )
        with self.assertRaisesRegex(ValueError, "same prepared dataset"):
            train(
                other_data,
                self.root / "original",
                self.model_config,
                self.config(),
                torch.device("cpu"),
                resume=self.root / "original" / "last.pt",
            )

    def test_stop_request_saves_the_last_completed_update(self):
        """停止要求で完了済みの学習を保存し、そこから残りの学習を再開する。"""
        config = TrainConfig(**{**self.config().__dict__, "log_every": 1})
        metrics = self.root / "stopped" / "metrics.jsonl"

        def stop_requested():
            return metrics.exists() and metrics.stat().st_size > 0

        stopped = self.run_training("stopped", config, stop_requested=stop_requested)
        saved = load_checkpoint(self.root / "stopped" / "last.pt")
        self.assertEqual(stopped["step"], 1)
        self.assertEqual(saved["step"], stopped["step"])
        self.assertEqual(saved["validation_step"], stopped["step"])
        self.run_training("stopped", config, resume=self.root / "stopped" / "last.pt")
        self.run_training("uninterrupted", config)
        resumed = load_checkpoint(self.root / "stopped" / "last.pt")
        continuous = load_checkpoint(self.root / "uninterrupted" / "last.pt")
        for name, parameter in continuous["model"].items():
            torch.testing.assert_close(
                parameter, resumed["model"][name], rtol=0, atol=0
            )

    def test_explicit_gpu_training_reports_unavailable_hardware(self):
        """GPU指定時にCUDAが使えなければ、環境の確認方法を通知する。"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "CUDA GPU is unavailable"),
        ):
            select_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_trains_and_evaluates_on_cuda(self):
        """CUDA上で学習し、日英の有限な検証損失を返す。"""
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                self.data,
                self.root / "cuda",
                self.model_config,
                self.config(),
                torch.device("cuda"),
                max_steps=2,
            )
        for language in ("ja", "en"):
            self.assertTrue(np.isfinite(result["validation"][language]["loss"]))


if __name__ == "__main__":
    unittest.main()
