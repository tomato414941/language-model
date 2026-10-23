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
    source_url,
)
from pretrain.comparison import compare_benchmarks
from pretrain.data import file_digest, prepare_data
from pretrain.evaluation import DEFAULT_TASKS, STANDARD_TASKS, NativeLM, evaluate_benchmarks
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
ROWS["jnli"] = [
    {
        "sentence_pair_id": "0",
        "yjcaptions_id": "1-2-3",
        "sentence1": "犬が芝生の上を走っている。",
        "sentence2": "動物が外で動いている。",
        "label": "entailment",
    },
    {
        "sentence_pair_id": "1",
        "yjcaptions_id": "4-5-6",
        "sentence1": "電車が駅に止まっている。",
        "sentence2": "電車が走っている。",
        "label": "contradiction",
    },
]
ID_FIELDS["jnli"] = "sentence_pair_id"
ROWS["lambada"] = [
    {"text": "The cat slept on the warm mat"},
    {"text": "Rain fell softly on the green hills"},
]
ROWS["xwinograd"] = [
    {
        "sentence": "ケーキが箱に入らなかった。_が大きすぎたからだ。",
        "option1": "ケーキ",
        "option2": "箱",
        "answer": "1",
    }
]
ROWS["xwinograd_ja"] = [
    {
        "sentence1": "箱にケーキが入らない。箱が大きい。",
        "sentence2": "箱にケーキが入らない。ケーキが大きい。",
        "answer": "2",
    }
]
ROWS["sciq"] = [
    {
        "question": "Which organ pumps blood through the body?",
        "distractor1": "lungs",
        "distractor2": "liver",
        "distractor3": "kidney",
        "correct_answer": "heart",
        "support": "The heart pumps blood through the body.",
    }
]
ROWS["piqa"] = [
    {
        "goal": "How do I keep bread fresh?",
        "sol1": "Store it in a sealed bag.",
        "sol2": "Leave it in the sun.",
        "label": 0,
    }
]
ROWS["kuci"] = [
    {
        "id": 0,
        "context": "電池 の 減り は 早い ので 、",
        "choice_a": "海 で 泳ぐ",
        "choice_b": "本 を 読む",
        "choice_c": "雨 が 降る",
        "choice_d": "予備 の 電池 を 買う",
        "label": "d",
    }
]
ID_FIELDS["kuci"] = "id"
ROWS["blimp"] = [
    {
        "sentence_good": "Raymond is selling this sketch.",
        "sentence_bad": "Raymond is selling this sketches.",
    }
]


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
    if path == "nyu-mll/blimp":
        # Each phenomenon gets sentences of its own lengths.
        rows = [
            {"sentence_good": "a" * (len(name) % k + 1), "sentence_bad": "bb"}
            for k in (3, 5)
        ]
        return DatasetDict(train=Dataset.from_list(rows))
    if path in ("allenai/sciq", "baber/piqa"):
        rows = Dataset.from_list(ROWS[path.split("/")[1]])
        return DatasetDict(train=rows, validation=rows, test=rows)
    tested = {
        "EleutherAI/lambada_openai": "lambada",
        "Muennighoff/xwinograd": "xwinograd",
        "polm-stability/xwinograd-ja": "xwinograd_ja",
    }
    if path in tested:
        return DatasetDict(test=Dataset.from_list(ROWS[tested[path]]))
    if path == "json" and "KUCI" in str(kwargs["data_files"]):
        return DatasetDict(test=Dataset.from_list(ROWS["kuci"]))
    if path == "json":
        task = "jnli" if "jnli" in kwargs["data_files"]["train"] else "jcommonsenseqa"
    else:
        task = {"Rowan/hellaswag": "hellaswag", "allenai/ai2_arc": "arc_easy"}[path]
    rows = ROWS[task]
    train = [{**rows[index % len(rows)], ID_FIELDS[task]: index} for index in range(6)]
    return DatasetDict(
        train=Dataset.from_list(train),
        validation=Dataset.from_list(rows if task != "arc_easy" else rows[:1]),
        test=Dataset.from_list(rows),
    )


def continuation_bits(tokenizer, probabilities, text):
    """Bits per byte of text under a model that predicts the same tokens everywhere."""
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    nats = -sum(math.log(probabilities[token]) for token in tokens)
    return nats / (math.log(2) * len(text.encode("utf-8")))


def skewed_probabilities():
    probabilities = torch.linspace(1, 3, 258, dtype=torch.float64)
    return probabilities / probabilities.sum()


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
                field = ID_FIELDS.get(name)
                rows = [
                    row
                    if field is None
                    else {
                        **row,
                        field: row[field] + 10 * offset
                        if isinstance(row[field], int)
                        else f"{row[field]}{offset}",
                    }
                    for row in ROWS[name]
                ]
                buffer = io.BytesIO()
                if path.endswith((".json", ".jsonl")):
                    buffer.write(
                        "".join(json.dumps(row) + "\n" for row in rows).encode()
                    )
                else:
                    parquet.write_table(arrow.Table.from_pylist(rows), buffer)
                self.files[source_url(task, path)] = buffer.getvalue()
                files[path] = hashlib.sha256(buffer.getvalue()).hexdigest()
            self.tasks[name] = {**task, "files": files}

    def fetch(self, output, files=None):
        def download(url):
            return (files or self.files)[url]

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
        self.assertEqual(
            benchmarks["sciq"][0]["context"],
            "Question: Which organ pumps blood through the body?\nAnswer:",
        )
        self.assertEqual(
            benchmarks["piqa"][0]["context"],
            "Question: How do I keep bread fresh?\nAnswer:",
        )
        self.assertEqual(
            benchmarks["blimp"][0]["context"], "Raymond is selling this sketch."
        )
        self.assertEqual(
            benchmarks["xwinograd_ja"][0]["context"], "箱にケーキが入らない。ケーキが大きい。"
        )
        self.assertEqual(benchmarks["jnli"][0]["context"], "犬が芝生の上を走っている。")
        self.assertEqual(benchmarks["kuci"][0]["context"], "電池の減りは早いので、予備の電池を買う")
        # Questions without identifiers of their own are numbered by file and row.
        self.assertEqual(benchmarks["sciq"][1]["id"], "data/validation-00000-of-00001.parquet:0")
        self.assertEqual(
            manifest["tasks"]["jnli"]["dataset_url"], "https://github.com/yahoojapan/JGLUE"
        )
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

    def test_採点した要求の数を進み具合として表示する(self):
        """Shows how many requests have been scored, for long evaluations' logs."""
        backend = NativeLM(
            FixedModel(torch.full((258,), 1 / 258)),
            byte_tokenizer(),
            torch.device("cpu"),
        )
        shown = io.StringIO()
        with contextlib.redirect_stderr(shown):
            backend._loglikelihood_tokens([(None, [0, 5], [6]), (None, [0, 7], [8, 9])])
        self.assertIn("Scoring requests", shown.getvalue())
        self.assertIn("2/2", shown.getvalue())

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
        self.assertEqual(configs["sciq"]["test_split"], "test")
        self.assertEqual(configs["piqa"]["validation_split"], "validation")
        self.assertEqual(result["n-shot"]["ja_leaderboard_jnli"], 3)
        self.assertEqual(configs["kuci"]["test_split"], "test")
        for name in DEFAULT_TASKS:
            self.assertIn("acc,none", result["results"][name])
        # BLiMP's group carries a version of its own.
        self.assertEqual(
            {name: float(result["versions"][name]) for name in DEFAULT_TASKS},
            {name: 2.0 if name == "blimp" else 1.0 for name in DEFAULT_TASKS},
        )

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
            contextlib.redirect_stdout(io.StringIO()) as shown,
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
        # The command shows the summary by language; the file keeps every question.
        self.assertEqual(json.loads(shown.getvalue()), result["by_language"])
        self.assertEqual(result["checkpoint"]["step"], 1)
        self.assertEqual(
            result["checkpoint"]["sha256"], file_digest(self.root / "run/last.pt")
        )
        # The results also list each of BLiMP's phenomena.
        tasks = {name for name in result["results"] if not name.startswith("blimp_")}
        self.assertEqual(tasks, set(DEFAULT_TASKS))
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
        tasks = {name for name in benchmarks["results"] if not name.startswith("blimp_")}
        self.assertEqual(tasks, set(DEFAULT_TASKS))
        for metrics in benchmarks["results"].values():
            self.assertTrue(0 <= metrics["acc,none"] <= 1)

    def evaluate_skewed(self, tasks):
        probabilities = skewed_probabilities()
        with patch(
            "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
        ):
            result = evaluate_benchmarks(
                FixedModel(probabilities),
                byte_tokenizer(),
                torch.device("cpu"),
                tasks=tasks,
            )
        return result, lambda text: continuation_bits(
            byte_tokenizer(), probabilities, text
        )

    def evaluate_two_models(self, tasks):
        with patch(
            "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
        ):
            return tuple(
                evaluate_benchmarks(
                    FixedModel(probabilities),
                    byte_tokenizer(),
                    torch.device("cpu"),
                    tasks=tasks,
                )
                for probabilities in (torch.full((258,), 1 / 258), skewed_probabilities())
            )

    def test_正解の選択肢を予測できた度合いを1バイトあたりのビット数で測る(self):
        """Reports the correct choice's bits per byte for each question and on average."""
        result, bits = self.evaluate_skewed(["arc_easy"])
        expected = [bits(" carbon dioxide"), bits(" water")]
        items = result["items"]["arc_easy"]
        self.assertEqual(items["doc_id"], [0, 1])
        for value, wanted in zip(items["gold_bits_per_byte"], expected, strict=True):
            self.assertAlmostEqual(value, wanted, places=4)
        metrics = result["results"]["arc_easy"]
        self.assertAlmostEqual(
            metrics["gold_bits_per_byte,none"], sum(expected) / 2, places=4
        )
        self.assertAlmostEqual(
            metrics["gold_bits_per_byte_stderr,none"],
            abs(expected[0] - expected[1]) / 2,
            places=4,
        )
        # Every accuracy the task reports is kept for each question.
        for metric in ("acc", "acc_norm"):
            self.assertEqual(sum(items["correct"][metric]) / 2, metrics[f"{metric},none"])
        self.assertFalse(result["higher_is_better"]["arc_easy"]["gold_bits_per_byte"])

    def test_KUCIは形態素の区切りの空白を除いた文で正解の続きを測る(self):
        """Scores KUCI's right continuation as Japanese text, without spaces between morphemes."""
        result, bits = self.evaluate_skewed(["kuci"])
        self.assertAlmostEqual(
            result["items"]["kuci"]["gold_bits_per_byte"][0], bits("予備の電池を買う"), places=4
        )

    def test_課題ごとに決めた正答率を言語別にまとめる(self):
        """Reads each standard task's chosen accuracy and groups the tasks by language."""
        result, _ = self.evaluate_skewed(["arc_easy", "sciq", "kuci", "ja_leaderboard_xwinograd"])
        summary = result["by_language"]
        self.assertEqual(set(summary), {"en", "ja"})
        self.assertEqual(set(summary["en"]), {"arc_easy", "sciq"})
        self.assertEqual(set(summary["ja"]), {"kuci", "ja_leaderboard_xwinograd"})
        for language, name, metric in (
            ("en", "arc_easy", "acc_norm"),
            ("en", "sciq", "acc"),
            ("ja", "kuci", "acc_norm"),
            ("ja", "ja_leaderboard_xwinograd", "acc"),
        ):
            read = summary[language][name]
            self.assertEqual(read["metric"], metric)
            self.assertEqual(read["accuracy"], result["results"][name][f"{metric},none"])

    def test_二つの評価を同じ問題ごとに比べて言語別に差を示す(self):
        """Pairs two evaluations' questions to show each task's difference by language."""
        base, other = self.evaluate_two_models(["arc_easy", "blimp"])
        paths = [self.root / "base.json", self.root / "other.json"]
        for path, result in zip(paths, (base, other)):
            path.write_text(json.dumps(result), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            main(["compare-benchmarks", "--base", str(paths[0]), "--other", str(paths[1])])
        comparison = json.loads(shown.getvalue())
        self.assertEqual(comparison, json.loads(json.dumps(compare_benchmarks(base, other))))
        arc = comparison["en"]["arc_easy"]
        self.assertEqual((arc["metric"], arc["questions"]), ("acc_norm", 2))
        self.assertAlmostEqual(arc["difference"], arc["other"] - arc["base"])
        changes = [
            b - a
            for a, b in zip(base["items"]["arc_easy"]["gold_bits_per_byte"], other["items"]["arc_easy"]["gold_bits_per_byte"])
        ]
        self.assertAlmostEqual(arc["bits_difference"], sum(changes) / 2, places=6)
        # A group's phenomena are paired question by question too.
        blimp = comparison["en"]["blimp"]
        self.assertEqual(blimp["questions"], 2 * 67)
        self.assertAlmostEqual(blimp["difference"], blimp["other"] - blimp["base"])

    def test_指標を見直しても前の評価を評価し直さずに比べる(self):
        """Reads a revised accuracy from each question's record, without evaluating again."""
        base, other = self.evaluate_two_models(["arc_easy"])
        with patch.dict(STANDARD_TASKS, {"arc_easy": ("en", "acc")}):
            arc = compare_benchmarks(base, other)["en"]["arc_easy"]
        self.assertEqual(arc["metric"], "acc")
        self.assertEqual(arc["base"], base["results"]["arc_easy"]["acc,none"])
        self.assertAlmostEqual(arc["difference"], arc["other"] - arc["base"])

    def test_片方にしかない課題や違う問題を解いた課題は理由を示して残りを比べる(self):
        """Lists the tasks it cannot pair, with the reason, and compares the rest."""
        with patch(
            "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
        ):
            whole, part = (
                evaluate_benchmarks(
                    FixedModel(torch.full((258,), 1 / 258)),
                    byte_tokenizer(),
                    torch.device("cpu"),
                    tasks=tasks,
                    limit=limit,
                )
                for tasks, limit in ((["arc_easy", "sciq"], None), (["arc_easy", "sciq", "kuci"], 1))
            )
        comparison = compare_benchmarks(whole, part)
        self.assertEqual(
            comparison["en"]["arc_easy"], {"not_compared": "the evaluations asked different questions"}
        )
        self.assertEqual(comparison["ja"]["kuci"], {"not_compared": "only one of the evaluations asked it"})
        self.assertEqual(comparison["en"]["sciq"]["questions"], 1)

    def test_文章の最後の語を当てる課題では最後の語を正解として測る(self):
        """Measures the passage's last word, as the task's continuation."""
        result, bits = self.evaluate_skewed(["lambada_openai"])
        values = result["items"]["lambada_openai"]["gold_bits_per_byte"]
        for value, word in zip(values, (" mat", " hills"), strict=True):
            self.assertAlmostEqual(value, bits(word), places=4)

    def test_選択肢ごとに文脈が変わる課題では正解の文脈に続く部分を測る(self):
        """Measures the shared ending after the correct choice's context."""
        scored = []

        def score(requests, **kwargs):
            # A context's length stands in for its likelihood, to tell the contexts apart.
            scored.extend(context for (context, _), _, _ in requests)
            return [(-float(len(context)), False) for (context, _), _, _ in requests]

        with (
            patch(
                "lm_eval.api.task.datasets.load_dataset", side_effect=benchmark_dataset
            ),
            patch.object(NativeLM, "_loglikelihood_tokens", side_effect=score),
        ):
            result = evaluate_benchmarks(
                FixedModel(torch.full((258,), 1 / 258)),
                byte_tokenizer(),
                torch.device("cpu"),
                tasks=["xwinograd_jp"],
            )
        correct = "ケーキが箱に入らなかった。ケーキ"
        self.assertIn(correct, scored)
        ending = " が大きすぎたからだ。".encode()
        self.assertAlmostEqual(
            result["items"]["xwinograd_jp"]["gold_bits_per_byte"][0],
            len(correct) / (math.log(2) * len(ending)),
            places=4,
        )

    def test_課題のまとまりでは正答率と同じ重みで各課題の値をまとめる(self):
        """Combines a group's tasks with the weights the group gives their accuracy."""
        result, _ = self.evaluate_skewed(["blimp"])
        members = result["group_subtasks"]["blimp"]
        self.assertEqual(len(members), 67)
        values = [result["results"][m]["gold_bits_per_byte,none"] for m in members]
        errors = [
            result["results"][m]["gold_bits_per_byte_stderr,none"] for m in members
        ]
        group = result["results"]["blimp"]
        # BLiMP averages its phenomena without weighting them by size.
        self.assertAlmostEqual(
            group["gold_bits_per_byte,none"], sum(values) / len(values), places=6
        )
        self.assertAlmostEqual(
            group["gold_bits_per_byte_stderr,none"],
            math.sqrt(sum(error**2 for error in errors)) / len(errors),
            places=6,
        )

    def test_JNLIは公開されたJSONの文字のラベルを選択肢の順に読む(self):
        """Reads JNLI's published labels as the task's choices in their order."""
        result, bits = self.evaluate_skewed(["ja_leaderboard_jnli"])
        self.assertEqual(result["n-shot"]["ja_leaderboard_jnli"], 3)
        values = result["items"]["ja_leaderboard_jnli"]["gold_bits_per_byte"]
        for value, label in zip(values, ("\n含意", "\n矛盾"), strict=True):
            self.assertAlmostEqual(value, bits(label), places=4)


if __name__ == "__main__":
    unittest.main()
