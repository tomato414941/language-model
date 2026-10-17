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
    for name in ("numpy", "torch", "tokenizers", "pyarrow", "lm_eval")
):
    raise unittest.SkipTest("Install training, corpus and evaluation dependencies")

import pyarrow as arrow
import torch
from datasets import Dataset, DatasetDict
from pyarrow import parquet
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.__main__ import main
from pretrain.benchmarks import (
    TASKS,
    fetch_benchmarks,
    load_benchmarks,
)
from pretrain.data import file_digest, prepare_data
from pretrain.evaluation import DEFAULT_TASKS, NativeLM, evaluate_benchmarks
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
IFEVAL_ROWS = [
    {
        "key": 1,
        "prompt": "Describe rain without using commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{"num_words": None}],
    }
]
ROWS["ifeval"] = IFEVAL_ROWS
ID_FIELDS["ifeval"] = "key"


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


def benchmark_dataset(path, name=None, **kwargs):
    """Supply source records while using the harness's real task definitions."""
    if path == "google/IFEval":
        return DatasetDict(train=Dataset.from_list(IFEVAL_ROWS))
    task = {
        "Rowan/hellaswag": "hellaswag",
        "allenai/ai2_arc": "arc_easy",
        "json": "jcommonsenseqa",
    }[path]
    rows = ROWS[task]
    train = [{**rows[index % len(rows)], ID_FIELDS[task]: index} for index in range(6)]
    return DatasetDict(
        train=Dataset.from_list(train),
        validation=Dataset.from_list(rows if task != "arc_easy" else rows[:1]),
        test=Dataset.from_list(rows),
    )


class FixedModel(torch.nn.Module):
    """Predict the same next-token distribution at every position."""

    def __init__(self, probabilities, context=256):
        super().__init__()
        self.config = SimpleNamespace(context=context)
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
                if path.endswith(".jsonl"):
                    buffer.write(
                        "".join(json.dumps(row) + "\n" for row in rows).encode()
                    )
                else:
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
        self.assertEqual(benchmarks["ifeval"][0]["context"], IFEVAL_ROWS[0]["prompt"])
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
        """Returns the same likelihoods when requests share a batch or run alone."""
        torch.manual_seed(0)
        config = ModelConfig(
            width=24, heads=4, kv_heads=2, layers=2, context=32, architecture="llama"
        )
        model = LanguageModel(258, config)
        requests = [
            SimpleNamespace(args=(context, choice))
            for context, choices in (
                ("Question: why?\nAnswer:", [" a", " bcd"]),
                ("質問：なぜ？\n回答：", ["絵本", "地図", "辞書"]),
            )
            for choice in choices
        ]
        device = torch.device("cpu")
        together = NativeLM(model, byte_tokenizer(), device).loglikelihood(
            requests, disable_tqdm=True
        )
        alone = NativeLM(model, byte_tokenizer(), device, batch_tokens=1).loglikelihood(
            requests, disable_tqdm=True
        )
        for expected, actual in zip(together, alone):
            self.assertAlmostEqual(expected[0], actual[0], places=4)
            self.assertEqual(expected[1], actual[1])

    def test_scores_continuations_and_rolling_documents_with_native_probabilities(self):
        """Scores every requested token and reports whether it is greedy."""
        tokenizer = byte_tokenizer()
        probabilities = torch.full((258,), 0.2 / 256)
        probabilities[tokenizer.token_to_id("a")] = 0.5
        probabilities[tokenizer.token_to_id("b")] = 0.3
        backend = NativeLM(
            FixedModel(probabilities, context=16), tokenizer, torch.device("cpu")
        )
        result = backend.loglikelihood(
            [SimpleNamespace(args=("", "a")), SimpleNamespace(args=("x", "bb"))],
            disable_tqdm=True,
        )
        self.assertAlmostEqual(result[0][0], math.log(0.5), places=5)
        self.assertTrue(result[0][1])
        self.assertAlmostEqual(result[1][0], 2 * math.log(0.3), places=5)
        self.assertFalse(result[1][1])
        documents = backend.loglikelihood_rolling(
            [SimpleNamespace(args=("a" * 35,)), SimpleNamespace(args=("",))]
        )
        self.assertAlmostEqual(documents[0], 35 * math.log(0.5), places=4)
        self.assertEqual(documents[1], 0)

    def test_generation_stops_at_the_requested_sequence_and_model_end_token(self):
        """Returns native greedy continuations up to the requested stop or EOS."""
        tokenizer = byte_tokenizer()
        model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=32))
        backend = NativeLM(model, tokenizer, torch.device("cpu"))

        def continuations(sequences, *args, **kwargs):
            return [tokenizer.encode("日本語 STOP suffix").ids for _ in sequences]

        with patch.object(model, "continue_greedily", side_effect=continuations):
            stopped, ended = backend.generate_until(
                [
                    SimpleNamespace(args=("x", {"until": ["STOP"]})),
                    SimpleNamespace(args=("x", {"until": []})),
                ]
            )
        self.assertEqual(stopped, "日本語 ")
        self.assertEqual(ended, "日本語 STOP suffix")

    def test_standard_tasks_use_held_out_splits_and_published_fewshot_counts(self):
        """Uses published evaluation splits, examples and metrics."""
        with patch(
            "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
        ):
            result = evaluate_benchmarks(
                FixedModel(torch.full((258,), 1 / 258)),
                byte_tokenizer(),
                torch.device("cpu"),
            )
        self.assertEqual(result["framework"]["name"], "lm-evaluation-harness")
        self.assertEqual(result["framework"]["version"], "0.4.13")
        self.assertEqual(result["evaluation_scope"], "full")
        self.assertEqual(result["n-samples"]["arc_easy"]["original"], 2)
        self.assertEqual(result["n-samples"]["hellaswag"]["original"], 3)
        self.assertEqual(
            result["n-samples"]["ja_leaderboard_jcommonsenseqa"]["original"], 2
        )
        configs = result["configs"]
        self.assertEqual(configs["arc_easy"]["test_split"], "test")
        self.assertEqual(configs["hellaswag"]["validation_split"], "validation")
        self.assertEqual(
            configs["ja_leaderboard_jcommonsenseqa"]["validation_split"], "validation"
        )
        self.assertEqual(result["n-shot"]["ja_leaderboard_jcommonsenseqa"], 3)
        for name in DEFAULT_TASKS:
            self.assertIn("acc,none", result["results"][name])
            self.assertEqual(result["versions"][name], 1.0)

    def test_instruction_following_asks_in_the_conversation_format(self):
        """Asks IFEval in the conversation format and keeps published choice prompts."""
        tokenizer = byte_tokenizer()
        model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=64))
        asked, limits = [], []

        def reply(sequences, max_new_tokens, *args, **kwargs):
            asked.extend(
                tokenizer.decode(sequence, skip_special_tokens=True)
                for sequence in sequences
            )
            limits.append(max_new_tokens)
            return [tokenizer.encode("Rain falls softly.").ids for _ in sequences]

        encode = NativeLM.tok_encode
        with (
            patch(
                "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
            ),
            # Keep the test offline; the replies here need no sentence splitter.
            patch("nltk.download"),
            patch.object(model, "continue_greedily", side_effect=reply),
            patch.object(
                NativeLM, "tok_encode", autospec=True, side_effect=encode
            ) as encoded,
        ):
            result = evaluate_benchmarks(
                model, tokenizer, torch.device("cpu"), tasks=["arc_easy", "ifeval"]
            )
        prompt = "User: Describe rain without using commas.\n\nAssistant:"
        self.assertEqual(asked, [prompt])
        self.assertEqual(limits, [1280])
        texts = [call.args[1] for call in encoded.call_args_list]
        self.assertEqual([text for text in texts if "User:" in text], [prompt])
        self.assertIn("Question: Which gas do plants take in?\nAnswer:", texts)
        self.assertEqual(result["results"]["ifeval"]["prompt_level_strict_acc,none"], 1)
        self.assertIn("acc,none", result["results"]["arc_easy"])
        self.assertEqual(result["conversation_format_tasks"], ["ifeval"])

    def test_limited_standard_evaluation_is_reported_as_a_smoke_check(self):
        """Records the limited scope and item count for a smoke check."""
        with patch(
            "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
        ):
            result = evaluate_benchmarks(
                FixedModel(torch.full((258,), 1 / 258)),
                byte_tokenizer(),
                torch.device("cpu"),
                tasks=["arc_easy"],
                limit=1,
            )
        self.assertEqual(result["evaluation_scope"], "smoke")
        self.assertEqual(
            result["n-samples"]["arc_easy"], {"original": 2, "effective": 1}
        )

    def test_benchmark_command_saves_standard_results_for_a_native_checkpoint(self):
        """Saves standard task results and the evaluated checkpoint's identity."""
        self.train_checkpoint()
        output = self.root / "results.json"
        with (
            patch(
                "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "benchmark",
                    "--checkpoint",
                    str(self.root / "run/last.pt"),
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--output",
                    str(output),
                ]
            )
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["checkpoint"]["step"], 1)
        self.assertEqual(
            result["checkpoint"]["sha256"], file_digest(self.root / "run/last.pt")
        )
        self.assertEqual(set(result["results"]), set(DEFAULT_TASKS))
        for metrics in result["results"].values():
            self.assertTrue(0 <= metrics["acc,none"] <= 1)

    def test_inspection_saves_standard_benchmark_results_in_the_summary(self):
        """Exports the selected checkpoint with its standard benchmark results."""
        self.train_checkpoint()
        with (
            patch("scripts.inspect_pretrain.torch", CpuTorch()),
            patch(
                "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
            ),
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
                    "--benchmark-tasks",
                    *DEFAULT_TASKS,
                ]
            )
        summary = json.loads(
            (self.root / "results/summary.json").read_text(encoding="utf-8")
        )
        benchmarks = summary["benchmarks"]
        self.assertEqual(benchmarks["framework"]["version"], "0.4.13")
        self.assertEqual(set(benchmarks["results"]), set(DEFAULT_TASKS))
        for metrics in benchmarks["results"].values():
            self.assertTrue(0 <= metrics["acc,none"] <= 1)


if __name__ == "__main__":
    unittest.main()
