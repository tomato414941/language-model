import contextlib
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None
    for name in ("numpy", "torch", "tokenizers", "pyarrow")
):
    raise unittest.SkipTest("Install training and corpus dependencies")

import pyarrow as arrow
import torch
from pyarrow import parquet
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.__main__ import main
from pretrain.benchmarks import (
    TASKS,
    choice_log_likelihoods,
    evaluate_benchmarks,
    fetch_benchmarks,
    load_benchmarks,
)
from pretrain.data import file_digest, prepare_data
from pretrain.model import LanguageModel, ModelConfig
from pretrain.training import TrainConfig, train
from scripts import inspect_pretrain

ROWS = {
    "hellaswag": [
        {
            "ind": index,
            "activity_label": "Cooking eggs",
            "ctx_a": "A man cracks eggs into a bowl.",
            "ctx_b": "he",
            "ctx": "A man cracks eggs into a bowl. he",
            "endings": [
                "[step] whisks them with a fork.",
                "throws the bowl away.",
                "reads a  newspaper.",
                "paints the wall.",
            ],
            "source_id": "activitynet~cooking",
            "split": "val",
            "split_type": "indomain",
            "label": "0",
        }
        for index in range(3)
    ],
    "arc_easy": [
        {
            "id": "letters",
            "question": "Which gas do plants take in?",
            "choices": {
                "text": ["oxygen", "carbon dioxide", "helium", "neon"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "numbers",
            "question": "Which is a liquid at room temperature?",
            "choices": {"text": ["iron", "water", "salt"], "label": ["1", "2", "3"]},
            "answerKey": "2",
        },
    ],
    "jcommonsenseqa": [
        {
            "q_id": index,
            "question": "子ども向けの絵が多い本は？",
            "choice0": "論文",
            "choice1": "辞書",
            "choice2": "絵本",
            "choice3": "新聞",
            "choice4": "地図",
            "label": 2,
        }
        for index in range(2)
    ],
}
ID_FIELDS = {"hellaswag": "ind", "arc_easy": "id", "jcommonsenseqa": "q_id"}


def byte_tokenizer():
    """A tokenizer whose vocabulary holds only the special tokens and bytes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=258,
        show_progress=False,
        special_tokens=["<|bos|>", "<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


class FixedModel(torch.nn.Module):
    """Predict the same next-token distribution at every position."""

    def __init__(self, probabilities):
        super().__init__()
        self.config = SimpleNamespace(context=16)
        self.register_buffer("logits", probabilities.log())

    def forward(self, tokens):
        return self.logits.expand(*tokens.shape, -1)


class CpuTorch:
    """Stand in for torch so the GPU inspection script runs on the CPU."""

    cuda = SimpleNamespace(get_device_name=lambda: "CPU")

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, name):
        return torch.device("cpu")


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files, self.tasks = {}, {}
        for name, task in TASKS.items():
            files = {}
            for offset, path in enumerate(task["files"]):
                # Each split holds different questions, as in the real sources.
                field = ID_FIELDS[name]
                rows = [
                    {
                        **row,
                        field: row[field] + 10 * offset
                        if isinstance(row[field], int)
                        else f"{row[field]}{offset}",
                    }
                    for row in ROWS[name]
                ]
                buffer = io.BytesIO()
                parquet.write_table(arrow.Table.from_pylist(rows), buffer)
                url = f"{task['repository']}/resolve/{task['revision']}/{path}"
                self.files[url] = buffer.getvalue()
                files[path] = hashlib.sha256(buffer.getvalue()).hexdigest()
            self.tasks[name] = {**task, "files": files}

    def fetch(self, output, files=None):
        def download(url):
            return (files or self.files)[url.split("/datasets/", 1)[1]]

        with (
            patch("pretrain.benchmarks.TASKS", self.tasks),
            patch("pretrain.benchmarks.download", side_effect=download),
        ):
            return fetch_benchmarks(output)

    def train_checkpoint(self):
        for language, text in (("ja", "あいう" * 30), ("en", "abc " * 40)):
            with (self.root / f"{language}.jsonl").open("w", encoding="utf-8") as file:
                for index in range(6):
                    file.write(json.dumps({"text": f"{text}{index}"}) + "\n")
        prepare_data(
            self.root / "ja.jsonl", self.root / "en.jsonl", self.root / "data", 258
        )
        config = TrainConfig(
            steps=2,
            batch_size=2,
            accumulation_steps=1,
            warmup_steps=1,
            eval_batches=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                self.root / "data",
                self.root / "run",
                ModelConfig(width=24, heads=4, layers=1, context=64),
                config,
                torch.device("cpu"),
                max_steps=1,
            )

    def test_fetch_converts_each_benchmark_to_shared_items_with_provenance(self):
        """取得したファイルを検証し、各ベンチマークを共通の問題形式に変換して出典とともに保存する。"""
        output = self.root / "benchmarks"
        manifest = self.fetch(output)
        benchmarks = load_benchmarks(output)
        hellaswag = benchmarks["hellaswag"][0]
        self.assertEqual(
            hellaswag["context"], "Cooking eggs: A man cracks eggs into a bowl. He"
        )
        self.assertEqual(hellaswag["choices"][0], " whisks them with a fork.")
        self.assertEqual(hellaswag["choices"][2], " reads a newspaper.")
        self.assertEqual({item["label"] for item in benchmarks["arc_easy"]}, {1})
        self.assertEqual(
            benchmarks["arc_easy"][1]["choices"], [" iron", " water", " salt"]
        )
        self.assertEqual(benchmarks["jcommonsenseqa"][0]["choices"][2], "絵本")
        for name, task in self.tasks.items():
            metadata = manifest["tasks"][name]
            self.assertEqual(metadata["revision"], task["revision"])
            self.assertEqual(metadata["license"], task["license"])
            self.assertEqual(metadata["sha256"], file_digest(output / f"{name}.jsonl"))
            self.assertEqual(metadata["items"], len(ROWS[name]) * len(task["files"]))

    def test_fetch_reports_a_source_whose_checksum_differs(self):
        """取得したファイルのチェックサムが記録と異なれば、何も保存せずに通知する。"""
        output = self.root / "benchmarks"
        url = next(iter(self.files))
        files = {**self.files, url: self.files[url] + b"changed"}
        with self.assertRaisesRegex(ValueError, "source checksum"):
            self.fetch(output, files)
        self.assertFalse(output.exists())

    def test_loading_reports_modified_benchmark_items(self):
        """保存後に書き換えられた問題ファイルを、チェックサム不一致として通知する。"""
        output = self.root / "benchmarks"
        self.fetch(output)
        with (output / "arc_easy.jsonl").open("a", encoding="utf-8") as destination:
            destination.write("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_benchmarks(output, ["arc_easy"])

    def test_choice_likelihoods_match_whether_scored_together_or_alone(self):
        """選択肢の尤度を、まとめて計算しても1件ずつ計算しても同じ値で返す。"""
        torch.manual_seed(0)
        config = ModelConfig(
            width=24, heads=4, kv_heads=2, layers=2, context=32, architecture="llama"
        )
        model = LanguageModel(258, config)
        items = [
            {
                "context": "Question: why?\nAnswer:",
                "choices": [" a", " bcd"],
                "label": 0,
            },
            {
                "context": "質問：なぜ？\n回答：",
                "choices": ["絵本", "地図", "辞書"],
                "label": 2,
            },
        ]
        device = torch.device("cpu")
        together = choice_log_likelihoods(model, byte_tokenizer(), items, device)
        alone = choice_log_likelihoods(
            model, byte_tokenizer(), items, device, batch_tokens=1
        )
        for expected, actual in zip(together, alone):
            for a, b in zip(expected, actual):
                self.assertAlmostEqual(a, b, places=4)

    def test_counts_the_most_likely_choice_and_reports_chance_and_error(self):
        """最も確からしい選択肢を回答として数え、偶然の正答率と誤差を添えて報告する。"""
        tokenizer = byte_tokenizer()
        probabilities = torch.full((258,), 0.2 / 256)
        probabilities[tokenizer.token_to_id("a")] = 0.5
        probabilities[tokenizer.token_to_id("b")] = 0.3
        items = [
            {"context": "x", "choices": ["a", "b"], "label": 0},
            {"context": "x", "choices": ["a", "b"], "label": 1},
            # Per character "aa" is more likely than "b", but not in total.
            {"context": "x", "choices": ["aa", "b", "c"], "label": 0},
        ]
        result = evaluate_benchmarks(
            FixedModel(probabilities), tokenizer, {"toy": items}, torch.device("cpu")
        )["toy"]
        self.assertEqual(result["items"], 3)
        self.assertAlmostEqual(result["chance"], (1 / 2 + 1 / 2 + 1 / 3) / 3)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["normalized_accuracy"], 2 / 3)
        self.assertAlmostEqual(
            result["normalized_accuracy_stderr"], math.sqrt(2 / 3 * 1 / 3 / 3)
        )

    def test_benchmark_command_saves_accuracy_for_a_trained_checkpoint(self):
        """学習済みチェックポイントの正答率を、各ベンチマークについて保存する。"""
        self.train_checkpoint()
        self.fetch(self.root / "benchmarks")
        output = self.root / "results.json"
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                [
                    "benchmark",
                    "--checkpoint",
                    str(self.root / "run/last.pt"),
                    "--benchmarks",
                    str(self.root / "benchmarks"),
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--output",
                    str(output),
                ]
            )
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["step"], 1)
        self.assertEqual(set(result["results"]), set(TASKS))
        for name, metrics in result["results"].items():
            self.assertEqual(
                metrics["items"], len(ROWS[name]) * len(TASKS[name]["files"])
            )
            self.assertTrue(0 <= metrics["normalized_accuracy"] <= 1)

    def test_inspection_saves_benchmark_accuracy_in_the_summary(self):
        """学習後の確認で、選んだチェックポイントの正答率を要約に保存する。"""
        self.train_checkpoint()
        self.fetch(self.root / "benchmarks")
        with (
            patch("scripts.inspect_pretrain.torch", CpuTorch()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            inspect_pretrain.main(
                [
                    "--output",
                    str(self.root / "run"),
                    "--data",
                    str(self.root / "data"),
                    "--results",
                    str(self.root / "results"),
                    "--benchmarks",
                    str(self.root / "benchmarks"),
                ]
            )
        summary = json.loads(
            (self.root / "results/summary.json").read_text(encoding="utf-8")
        )
        benchmarks = summary["benchmarks"]
        self.assertEqual(
            benchmarks["sha256"], file_digest(self.root / "benchmarks/sources.json")
        )
        self.assertEqual(set(benchmarks["results"]), set(TASKS))
        for name, metrics in benchmarks["results"].items():
            self.assertEqual(
                metrics["items"], len(ROWS[name]) * len(TASKS[name]["files"])
            )


if __name__ == "__main__":
    unittest.main()
