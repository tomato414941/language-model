import contextlib
import importlib.util
import io
import itertools
import json
import math
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

from pretrain.data import Corpus, prepare_data, token_bytes
from pretrain.model import LanguageModel, ModelConfig
from pretrain.training import (
    TrainConfig,
    evaluate,
    evaluate_windows,
    learning_rate,
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

    def test_batched_greedy_continuations_match_generating_each_prompt_alone(self):
        """長さの違う入力をまとめて貪欲生成しても、1件ずつ生成したときと同じトークンを返す。"""
        tokenizer = self.corpus.tokenizer
        bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
        prompts = ["日本語とEnglish", "abc", "あいう終わり"]
        sequences = [
            [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            for prompt in prompts
        ]
        for config in (
            ModelConfig(width=24, heads=4, layers=2, context=64),
            ModelConfig(
                width=24,
                heads=4,
                kv_heads=2,
                layers=2,
                context=64,
                architecture="llama",
                intermediate_size=64,
            ),
        ):
            torch.manual_seed(3)
            model = LanguageModel(258, config).eval()
            # Larger weights make each prediction depend on the earlier positions.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.normal_(std=0.3)
            alone = [
                [
                    token
                    for token in model.sample_tokens(
                        tokenizer, prompt, 20, temperature=0, top_k=0
                    )
                    if token != eos
                ]
                for prompt in prompts
            ]
            self.assertEqual(
                model.continue_greedily(sequences, 20, eos, banned=[bos]), alone
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

    def wsd_config(self, steps=8, **changes):
        return TrainConfig(
            **{
                **self.config(steps).__dict__,
                "schedule": "wsd",
                "decay_steps": 2,
                "sampling": "sequential",
                **changes,
            }
        )

    def test_wsd_schedule_holds_the_peak_and_decays_over_the_last_steps(self):
        config = self.wsd_config(steps=10, decay_steps=4)
        rates = [learning_rate(config, step) for step in range(10)]
        self.assertEqual(rates[1:6], [0.01] * 5)
        self.assertTrue(all(a > b for a, b in itertools.pairwise(rates[5:])))
        self.assertAlmostEqual(rates[-1], 0.001)
        longer = self.wsd_config(steps=20, decay_steps=4)
        self.assertEqual(learning_rate(longer, 5), learning_rate(config, 5))

    def test_square_root_decay_falls_faster_at_first_and_reaches_zero(self):
        """1-sqrtの減衰は直線より先に大きく下がり、最後のステップで0にする。"""
        linear = self.wsd_config(steps=10, decay_steps=4, min_learning_rate=0.0)
        shaped = self.wsd_config(
            steps=10, decay_steps=4, min_learning_rate=0.0, decay_shape="1-sqrt"
        )
        straight = [learning_rate(linear, step) for step in range(10)]
        rates = [learning_rate(shaped, step) for step in range(10)]
        self.assertEqual(rates[:6], straight[:6])
        self.assertAlmostEqual(rates[6], 0.005)
        self.assertTrue(all(a < b for a, b in zip(rates[6:9], straight[6:9])))
        self.assertTrue(all(a > b for a, b in itertools.pairwise(rates[5:])))
        self.assertEqual(rates[-1], 0.0)

    def test_every_window_evaluation_predicts_each_held_out_token_once(self):
        """学習データなしで検証データを窓に区切り、各トークンを一度ずつ予測した損失を返す。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        context = self.model_config.context
        for path in self.data.glob("train_*.bin"):
            path.unlink()
        corpus = Corpus(self.data, ("validation",))
        result = evaluate_windows(model, corpus, torch.device("cpu"), batch_size=3)
        for language in ("ja", "en"):
            stream = corpus.streams["validation"][language]
            count = (len(stream) - 1) // context
            tokens = torch.from_numpy(
                np.asarray(stream[: count * context + 1], dtype=np.int64)
            )
            with torch.no_grad():
                expected = model(
                    tokens[:-1].view(count, context), tokens[1:].view(count, context)
                ).item()
            self.assertAlmostEqual(result[language]["loss"], expected, places=5)
            self.assertEqual(result[language]["evaluated_tokens"], count * context)
        first = evaluate_windows(
            model, corpus, torch.device("cpu"), batch_size=3, limit=1
        )
        self.assertEqual(first["ja"]["evaluated_tokens"], context)

    def test_token_bytes_count_the_text_each_token_stands_for(self):
        """各トークンが表す文章のバイト数を数え、特別なトークンは0バイトとする。"""
        data = self.root / "merged"
        prepare_data(
            self.japanese,
            self.english,
            data,
            vocabulary_size=300,
            validation_fraction=0.25,
        )
        tokenizer = Corpus(data).tokenizer
        sizes = token_bytes(tokenizer)
        text = "あいうあいう 終わり1 abc abc ending1"
        ids = tokenizer.encode(text).ids
        self.assertGreater(sizes[ids].max(), 1)
        self.assertEqual(sizes[ids].sum(), len(text.encode()))
        for special in ("<|bos|>", "<|eos|>"):
            self.assertEqual(sizes[tokenizer.token_to_id(special)], 0)

    def test_validation_reports_bits_per_byte_of_the_predicted_text(self):
        """検証で、予測したトークンのビット数を、それらが表す文章のバイト数で割って返す。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        corpus = Corpus(self.data, ("validation",))
        sizes = token_bytes(corpus.tokenizer)
        every = evaluate_windows(model, corpus, torch.device("cpu"), batch_size=3)
        sampled = evaluate(model, corpus, torch.device("cpu"), batches=2, batch_size=2)
        for language in ("ja", "en"):
            stream = corpus.streams["validation"][language]
            tokens = every[language]["evaluated_tokens"]
            targets = np.asarray(stream[1 : tokens + 1], dtype=np.int64)
            self.assertEqual(every[language]["evaluated_bytes"], sizes[targets].sum())
            for result in (every, sampled):
                measured = result[language]
                self.assertAlmostEqual(
                    measured["bits_per_byte"],
                    measured["loss"]
                    * measured["evaluated_tokens"]
                    / math.log(2)
                    / measured["evaluated_bytes"],
                )
        for result in (every, sampled):
            self.assertAlmostEqual(
                result["bits_per_byte"],
                (result["ja"]["bits_per_byte"] + result["en"]["bits_per_byte"]) / 2,
            )

    def test_compiled_training_saves_checkpoints_that_load_without_compilation(self):
        """torch.compileで学習しても、通常のモデルで同じ予測をするチェックポイントを保存する。"""
        result = self.run_training("compiled", self.config(steps=4), compiled=True)
        model, _, payload = load_model(
            self.root / "compiled/last.pt", torch.device("cpu")
        )
        self.assertEqual(payload["step"], 4)
        again = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        for language in ("ja", "en"):
            self.assertAlmostEqual(
                again[language]["loss"],
                result["validation"][language]["loss"],
                places=4,
            )

    def test_sequential_reading_uses_each_window_once_per_epoch(self):
        context = 16
        streams = self.corpus.streams["train"]
        # Distinct tokens make every window recognizable.
        streams["ja"], streams["en"] = (
            np.arange(10 * context + 1),
            np.arange(7 * context + 5),
        )
        counts = {"ja": 10, "en": 7}
        x, _ = self.corpus.sequential_batch(0, 20, context, 42, torch.device("cpu"))
        for offset, language in enumerate(("ja", "en")):
            seen = x[offset::2][: counts[language]]
            windows = {
                tuple(np.asarray(streams[language][i * context : (i + 1) * context]))
                for i in range(counts[language])
            }
            self.assertEqual({tuple(row.tolist()) for row in seen}, windows)
            self.assertEqual(
                len({tuple(row.tolist()) for row in seen}), counts[language]
            )
        again, _ = self.corpus.sequential_batch(3, 4, context, 42, torch.device("cpu"))
        torch.testing.assert_close(again, x[3:7], rtol=0, atol=0)

    def test_sequential_training_resumes_the_same_updates(self):
        self.run_training("sequential-continuous", self.wsd_config())
        self.run_training("sequential-resumed", self.wsd_config(), max_steps=3)
        self.run_training(
            "sequential-resumed",
            self.wsd_config(),
            resume=self.root / "sequential-resumed/last.pt",
        )
        first, second = (
            load_checkpoint(self.root / name / "last.pt")
            for name in ("sequential-continuous", "sequential-resumed")
        )
        for name, parameter in first["model"].items():
            torch.testing.assert_close(parameter, second["model"][name], rtol=0, atol=0)

    def test_continuation_carries_the_run_onto_new_data_without_a_rate_jump(self):
        self.run_training("stable", self.wsd_config(), max_steps=6)
        stable = load_checkpoint(self.root / "stable/last.pt")
        new_data = self.root / "decay-data"
        prepare_data(
            self.japanese,
            self.english,
            new_data,
            vocabulary_size=258,
            seed=99,
            tokenizer_path=self.data / "tokenizer.json",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                new_data,
                self.root / "decay",
                self.model_config,
                self.wsd_config(),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
            )
        decayed = load_checkpoint(self.root / "decay/last.pt")
        self.assertEqual(result["step"], 8)
        self.assertEqual(decayed["phase_start_step"], 6)
        self.assertEqual(decayed["tokens_seen"], stable["tokens_seen"] + 2 * 4 * 2 * 16)
        self.assertEqual(decayed["dataset_fingerprint"], Corpus(new_data).fingerprint)
        self.assertEqual(decayed["initialization"]["continued_from"]["step"], 6)
        with contextlib.redirect_stdout(io.StringIO()):
            extended = train(
                self.data,
                self.root / "extended",
                self.model_config,
                self.wsd_config(steps=12),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
                max_steps=7,
            )
        self.assertEqual(extended["step"], 7)
        with self.assertRaisesRegex(ValueError, "keep the learning rate"):
            train(
                new_data,
                self.root / "restarted",
                self.model_config,
                self.config(),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
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
