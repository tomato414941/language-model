import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None
    for name in ("numpy", "torch", "tokenizers", "pyarrow", "fastapi")
):
    raise unittest.SkipTest(
        "Install training and corpus dependencies with uv sync --extra train-cpu --extra corpus"
    )

import numpy as np
import pyarrow
from pyarrow import parquet
import torch
from tokenizers import Tokenizer

from pretrain.__main__ import main
from pretrain.data import LANGUAGES, file_digest, prepare_data
from pretrain.instructions import (
    InstructionCorpus,
    chat_prompt,
    conversation_text,
    encode_conversations,
    fetch_instructions,
    prepare_instructions,
)
from pretrain.model import ModelConfig
from pretrain.training import TrainConfig, evaluate, load_model, train
from responses_api import ResponseRequest


def exchange(question, answer):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def conversations(count=12):
    return {
        "ja": [exchange(f"質問{i}は？", f"答え{i}です。") for i in range(count)],
        "en": [
            exchange(f"What is item {i}?", f"It is answer {i}.") for i in range(count)
        ],
    }


class InstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Transcripts in the text let the tokenizer merge labels and blank lines.
        paths = {}
        for language, items in conversations(16).items():
            paths[language] = self.root / f"{language}.jsonl"
            paths[language].write_text(
                "".join(
                    json.dumps(
                        {"text": conversation_text(messages)[0] * 2}, ensure_ascii=False
                    )
                    + "\n"
                    for messages in items
                ),
                encoding="utf-8",
            )
        self.pretraining = self.root / "pretraining"
        prepare_data(
            paths["ja"],
            paths["en"],
            self.pretraining,
            vocabulary_size=300,
            validation_fraction=0.25,
        )
        self.tokenizer_path = self.pretraining / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.model_config = ModelConfig(
            width=32, heads=4, layers=2, context=64, architecture="llama", kv_heads=2
        )

    def write_source(self, items, name="source"):
        source = self.root / name
        source.mkdir()
        files = {}
        for language in LANGUAGES:
            path = source / f"{language}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {"source": "test", "messages": messages}, ensure_ascii=False
                    )
                    + "\n"
                    for messages in items[language]
                ),
                encoding="utf-8",
            )
            files[path.name] = {
                "conversations": len(items[language]),
                "sha256": file_digest(path),
            }
        (source / "sources.json").write_text(
            json.dumps({"version": 1, "sources": {}, "files": files}), encoding="utf-8"
        )
        return source

    def prepare(self, items, name="tuning", **kwargs):
        options = {"context": 64, "validation_conversations": 2, **kwargs}
        data = self.root / name
        prepare_instructions(
            self.write_source(items, f"{name}-source"),
            data,
            self.tokenizer_path,
            **options,
        )
        return data

    def config(self, steps):
        return TrainConfig(
            steps=steps,
            batch_size=4,
            accumulation_steps=1,
            learning_rate=0.01,
            min_learning_rate=0.001,
            warmup_steps=1,
            eval_every=steps,
            save_every=steps,
            eval_batches=2,
            log_every=steps,
        )

    def rows(self, corpus, split, language):
        """Return each packed conversation with the text of its learned tokens."""
        tokens, masks = corpus.rows[split][language]
        bos = self.tokenizer.token_to_id("<|bos|>")
        found = []
        for row, mask in zip(np.asarray(tokens), np.asarray(masks)):
            starts = list(np.flatnonzero(row == bos)) + [len(row)]
            for start, end in zip(starts, starts[1:]):
                found.append(
                    (
                        self.tokenizer.decode(row[start:end].tolist()),
                        self.tokenizer.decode(
                            row[start:end][mask[start:end] == 1].tolist(),
                            skip_special_tokens=False,
                        ),
                    )
                )
        return found

    def test_transcript_matches_the_api_prompt_before_each_reply(self):
        """会話をAPIと同じ形式で並べ、回答の直前までの文字とトークンがAPIの入力と一致する。"""
        messages = [
            {"role": "system", "content": "丁寧に答える。"},
            *exchange("色は？", "赤です。"),
            *exchange("What about the sky?", "It is blue."),
        ]
        request = ResponseRequest.model_validate(
            {"model": "test", "input": messages[:-1]}
        )
        text, replies = conversation_text(messages)
        self.assertEqual(text[: replies[-1][0]], request.prompt())
        self.assertEqual(chat_prompt(messages[:-1]), request.prompt())
        tokens, mask = encode_conversations(self.tokenizer, [messages])[0]
        reply_starts = np.flatnonzero((mask[1:] == 1) & (mask[:-1] == 0)) + 1
        self.assertEqual(len(reply_starts), 2)
        self.assertEqual(
            tokens[: reply_starts[-1]].tolist(),
            [
                self.tokenizer.token_to_id("<|bos|>"),
                *self.tokenizer.encode(request.prompt(), add_special_tokens=False).ids,
            ],
        )

    def test_preparation_learns_only_replies_and_the_final_end_token(self):
        """利用者の発言は学習対象から外し、回答と会話の最後の終了記号だけを学習する。"""
        corpus = InstructionCorpus(self.prepare(conversations()))
        for language in LANGUAGES:
            packed = self.rows(corpus, "train", language)
            self.assertGreater(
                len(packed), corpus.manifest["files"][f"train_{language}.bin"]["rows"]
            )
            for text, learned in packed:
                reply = text.split("Assistant:")[1].removesuffix("<|eos|>")
                self.assertEqual(learned, reply + "<|eos|>")
        x, y = corpus.batch(
            "train", 3, 64, torch.Generator().manual_seed(1), torch.device("cpu")
        )
        self.assertEqual(x.shape, (3, 64))
        learned = y[y != -100]
        self.assertTrue(len(learned))
        self.assertNotIn("User", self.tokenizer.decode(learned.tolist()))

    def test_preparation_holds_out_validation_conversations_and_applies_the_budget(
        self,
    ):
        """日英それぞれで検証用の会話を学習用と重ならないように取り分け、学習用は上限で打ち切る。"""
        data = self.prepare(conversations(), tokens_per_language=60)
        corpus = InstructionCorpus(data)
        for language in LANGUAGES:
            stats = corpus.manifest["conversations"][language]
            self.assertEqual(stats["validation"], 2)
            self.assertGreater(stats["unused"], 0)
            self.assertEqual(stats["validation"] + stats["train"] + stats["unused"], 12)
            validation = {text for text, _ in self.rows(corpus, "validation", language)}
            train_texts = {text for text, _ in self.rows(corpus, "train", language)}
            self.assertEqual(len(validation), 2)
            self.assertTrue(train_texts)
            self.assertTrue(validation.isdisjoint(train_texts))

    def test_preparation_drops_duplicates_benchmark_questions_and_long_conversations(
        self,
    ):
        """重複した会話、ベンチマークの問題を含む会話、文脈長に収まらない会話を除く。"""
        question = "電車に乗るときに必要なものはどれでしょう"
        benchmarks = self.root / "benchmarks"
        benchmarks.mkdir()
        items = benchmarks / "jcommonsenseqa.jsonl"
        items.write_text(
            json.dumps(
                {
                    "id": "1",
                    "context": f"質問：{question}\n回答：",
                    "choices": ["切符", "傘"],
                    "label": 0,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (benchmarks / "sources.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "tasks": {
                        "jcommonsenseqa": {
                            "language": "ja",
                            "filename": items.name,
                            "sha256": file_digest(items),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        items = conversations()
        items["ja"] += [
            items["ja"][0],
            exchange(f"次の問題に答えて。{question}？", "切符です。"),
            exchange("長い答えをください。", "とても" * 40),
        ]
        corpus = InstructionCorpus(self.prepare(items, benchmarks=benchmarks))
        stats = corpus.manifest["conversations"]["ja"]
        self.assertEqual(
            (stats["duplicates"], stats["contaminated"], stats["too_long"]), (1, 1, 1)
        )
        self.assertEqual(stats["train"] + stats["validation"], 12)
        texts = [
            text
            for split in ("train", "validation")
            for text, _ in self.rows(corpus, split, "ja")
        ]
        self.assertFalse(
            any("電車" in text or "とてもとても" in text for text in texts)
        )

    def test_training_requires_the_context_the_rows_were_packed_for(self):
        """詰めたときと違う文脈長のモデルでは学習を始めない。"""
        data = self.prepare(conversations())
        with self.assertRaisesRegex(ValueError, "64-token context"):
            train(
                data,
                self.root / "short",
                ModelConfig(width=32, heads=4, layers=1, context=32),
                self.config(2),
                torch.device("cpu"),
            )

    def test_tuning_from_pretrained_weights_learns_replies_that_end(self):
        """事前学習の重みから指示データで学習し、回答の形を覚えて終了記号で止める。"""
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                self.pretraining,
                self.root / "base",
                self.model_config,
                self.config(4),
                torch.device("cpu"),
            )
        data = self.prepare(conversations())
        base, _, _ = load_model(self.root / "base/last.pt", torch.device("cpu"))
        before = evaluate(
            base, InstructionCorpus(data), torch.device("cpu"), batches=2, batch_size=4
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                data,
                self.root / "tuned",
                self.model_config,
                self.config(80),
                torch.device("cpu"),
                initialize_from=self.root / "base/last.pt",
            )
        self.assertLess(result["validation"]["loss"], before["loss"] * 0.5)
        output = self.root / "chat.json"
        prompts = {"ja": ["質問3は？"], "en": ["What is item 3?"]}
        with (
            patch("pretrain.instructions.CHAT_PROMPTS", prompts),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "chat-samples",
                    "--checkpoint",
                    str(self.root / "tuned/last.pt"),
                    "--device",
                    "cpu",
                    "--max-new-tokens",
                    "24",
                    "--output",
                    str(output),
                ]
            )
        samples = json.loads(output.read_text(encoding="utf-8"))["samples"]
        self.assertEqual(
            [sample["question"] for sample in samples], ["質問3は？", "What is item 3?"]
        )
        self.assertRegex(samples[0]["reply"], r"^答え\d+です。$")
        self.assertRegex(samples[1]["reply"], r"^It is answer \d+\.$")
        self.assertTrue(all(sample["finished"] for sample in samples))


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        table = io.BytesIO()
        parquet.write_table(
            pyarrow.Table.from_pylist([{"messages": exchange("猫は？", "動物です。")}]),
            table,
        )
        self.content = {
            "chat.jsonl": "".join(
                json.dumps({"conversations": messages}, ensure_ascii=False) + "\n"
                for messages in (
                    [
                        {"role": "system", "content": "短く答える。"},
                        *exchange("空は？", "青です。"),
                    ],
                    [
                        *exchange("海は？", "広いです。"),
                        {"role": "user", "content": "続けて"},
                    ],
                    exchange("空白", " "),
                )
            ).encode(),
            "dolly.jsonl": "".join(
                json.dumps(record) + "\n"
                for record in (
                    {"instruction": "Name a color.", "context": "", "response": "Red."},
                    {
                        "instruction": "Summarize.",
                        "context": "Text here.",
                        "response": "Short.",
                    },
                    {"instruction": " ", "context": "", "response": "x"},
                )
            ).encode(),
            "data/train.parquet": table.getvalue(),
        }

        def source(language, layout, path):
            return {
                "language": language,
                "repository": f"test/{layout}",
                "revision": "0" * 40,
                "license": "Apache-2.0",
                "origin": "test",
                "layout": layout,
                "files": {path: hashlib.sha256(self.content[path]).hexdigest()},
            }

        self.sources = {
            "chat-ja": source("ja", "conversations", "chat.jsonl"),
            "pairs-en": source("en", "dolly", "dolly.jsonl"),
            "table-ja": source("ja", "messages", "data/train.parquet"),
        }

    def download(self, url, destination):
        data = next(value for path, value in self.content.items() if url.endswith(path))
        Path(destination).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_fetch_converts_sources_to_shared_conversations_with_provenance(self):
        """固定した出典から取得して会話を共通の形式にそろえ、壊れた会話は数えて除く。"""
        output = self.root / "source"
        with (
            patch("pretrain.instructions.SOURCES", self.sources),
            patch("pretrain.instructions.download_file", side_effect=self.download),
        ):
            manifest = fetch_instructions(output)
        read = {
            language: [
                json.loads(line)
                for line in (output / f"{language}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for language in LANGUAGES
        }
        self.assertEqual(
            [record["messages"] for record in read["ja"]],
            [
                [
                    {"role": "system", "content": "短く答える。"},
                    *exchange("空は？", "青です。"),
                ],
                exchange("海は？", "広いです。"),
                exchange("猫は？", "動物です。"),
            ],
        )
        self.assertEqual(
            [record["messages"] for record in read["en"]],
            [
                exchange("Name a color.", "Red."),
                exchange("Summarize.\n\nText here.", "Short."),
            ],
        )
        self.assertEqual(
            {
                name: (item["conversations"], item["malformed"])
                for name, item in manifest["sources"].items()
            },
            {"chat-ja": (2, 1), "pairs-en": (2, 1), "table-ja": (1, 0)},
        )
        for language in LANGUAGES:
            self.assertEqual(
                manifest["files"][f"{language}.jsonl"]["sha256"],
                file_digest(output / f"{language}.jsonl"),
            )
        self.assertEqual(json.loads((output / "sources.json").read_text()), manifest)

    def test_fetch_stops_when_a_source_checksum_differs(self):
        """出典のチェックサムが固定した値と違うときは、取得をやめて何も残さない。"""
        self.sources["pairs-en"]["files"]["dolly.jsonl"] = "0" * 64
        with (
            patch("pretrain.instructions.SOURCES", self.sources),
            patch("pretrain.instructions.download_file", side_effect=self.download),
            self.assertRaisesRegex(ValueError, "checksum does not match"),
        ):
            fetch_instructions(self.root / "source")
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
