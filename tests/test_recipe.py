import contextlib
import importlib.util
import io
import json
import math
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None
    for name in (
        "numpy",
        "torch",
        "tokenizers",
        "ahocorasick",
        "pyarrow",
        "huggingface_hub",
    )
):
    raise unittest.SkipTest("Install training and corpus dependencies")

import numpy as np
import pyarrow as arrow
import torch
from pyarrow import parquet

from pretrain.__main__ import main
from pretrain.data import Corpus, file_digest
from pretrain.instructions import InstructionCorpus, prepare_instructions
from pretrain.recipe import (
    BenchmarkFilter,
    accepted_documents,
    document_digest,
    inspect_recipe,
    iter_source,
    prepare_corpus,
    prepare_tokenizer,
    read_recipe,
    validate_corpus,
    validation_document,
    write_sft_config,
)
from pretrain.training import load_checkpoint, read_config, train

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe_path = self.root / "recipe.toml"
        text = (ROOT / "configs/pretrain-1b.toml").read_text()
        for old, new in (
            ("bilingual-1b.toml", "training.toml"),
            ("30_000_000_000", "2000"),
            ("32768", "258"),
            ("67_108_864", "1000"),
            ("5_000_000", "200"),
            ("0.001", "0.25"),
            ("minimum_characters = 300", "minimum_characters = 50"),
        ):
            text = text.replace(old, new)
        self.recipe_path.write_text(text)
        (self.root / "training.toml").write_text(
            '[model]\narchitecture="llama"\nwidth=24\nheads=4\nkv_heads=2\nlayers=1\nintermediate_size=64\ncontext=128\n'
            "[training]\nsteps=8\nbatch_size=2\naccumulation_steps=1\nwarmup_steps=1\neval_batches=1\n"
        )
        self.recipe = read_recipe(self.recipe_path)
        self.benchmarks = self.root / "benchmarks"
        self.benchmarks.mkdir()
        self.question = "Describe rain without using commas."
        questions = self.benchmarks / "ifeval.jsonl"
        questions.write_text(json.dumps({"id": "1", "context": self.question}) + "\n")
        (self.benchmarks / "sources.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "tasks": {
                        "ifeval": {
                            "language": "en",
                            "filename": questions.name,
                            "sha256": file_digest(questions),
                        }
                    },
                }
            )
        )
        self.rows = {}
        for source in self.recipe["sources"]:
            language = source["language"]
            text = (
                "自然科学と歴史を説明する文章です。"
                if language == "ja"
                else "An explanation of natural science and history. "
            ) * 20
            rows = []
            for index in range(100):
                row = {
                    "text": f"{text} {source['id']}:{index}",
                    "url": f"https://example.org/{source['id']}/{index}",
                    "id": f"{source['id']}:{index}",
                    "source_file": source["prefix"] + "fixture.parquet",
                    "language": source.get("source_language", language),
                    "in_language": language,
                    "language_score": 0.99,
                }
                if "quality_field" in source:
                    row[source["quality_field"]] = 0.9 if language == "ja" else 4.0
                rows.append(row)
            self.rows[source["id"]] = rows
        self.source_patch = patch(
            "pretrain.recipe.iter_source",
            side_effect=lambda source, seed: iter(self.rows[source["id"]]),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def prepare(self):
        tokenizer = self.root / "tokenizer"
        prepare_tokenizer(self.recipe_path, tokenizer, self.benchmarks)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(
                self.recipe_path, tokenizer, self.root / "corpus", self.benchmarks
            )
        return Corpus(self.root / "corpus")

    def test_学習計画のモデル規模と混合比と実行手順を表示する(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-1b.toml", "data/pretrain-1b")
        self.assertEqual(plan["parameters"], 1_036_085_248)
        self.assertEqual(plan["nominal_training_tokens"], 30_000_000_000)
        self.assertEqual(plan["scheduled_training_tokens"], 30_000_283_648)
        self.assertEqual(plan["minimum_packed_bytes"], 60_020_000_000)
        self.assertEqual(
            sum(s["training_tokens"] for s in plan["sources"] if s["language"] == "ja"),
            15_000_000_000,
        )
        self.assertEqual(
            plan["commands"][-1][-4:],
            ["hellaswag", "arc_easy", "ja_leaderboard_jcommonsenseqa", "ifeval"],
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(
                [
                    "recipe",
                    "--config",
                    str(self.recipe_path),
                    "--workdir",
                    str(self.root),
                ]
            )
        self.assertEqual(json.loads(output.getvalue())["nominal_training_tokens"], 2000)

    def test_各データ源の割当を満たす日英コーパスを作成する(self):
        corpus = self.prepare()
        result = validate_corpus(self.recipe_path, self.root / "corpus")
        self.assertTrue(result["ready"])
        for source in corpus.manifest["sources"].values():
            for split, target in source["targets"].items():
                self.assertGreaterEqual(source["statistics"][f"{split}_tokens"], target)
        for split in ("train", "validation"):
            for language in ("ja", "en"):
                stream = corpus.streams[split][language]
                self.assertLess(int(stream.max()), corpus.tokenizer.get_vocab_size())
                self.assertEqual(
                    (self.root / "corpus" / f"{split}_{language}.bin").stat().st_size,
                    len(stream) * 2,
                )
        records = [
            json.loads(line)
            for line in (self.root / "corpus/documents.jsonl").read_text().splitlines()
        ]
        digest_sets = {
            split: {r["digest"] for r in records if r["split"] == split}
            for split in ("train", "validation")
        }
        self.assertTrue(digest_sets["train"].isdisjoint(digest_sets["validation"]))
        self.assertEqual(
            sum(corpus.manifest["documents"][lang]["train"] for lang in ("ja", "en")),
            len(digest_sets["train"]),
        )
        x, y = corpus.batch(
            "train", 2, 128, torch.Generator().manual_seed(1), torch.device("cpu")
        )
        self.assertEqual(tuple(x.shape), (2, 128))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_評価問題と重複文書を除外し語彙学習を学習用文書に限定する(self):
        rows = self.rows["en-web"]
        self.rows["en-web"] = [
            {**rows[0], "text": rows[0]["text"] + self.question},
            rows[0],
            {**rows[0], "text": rows[0]["text"].replace(" ", "  ")},
            *rows[1:],
        ]
        corpus = self.prepare()
        stats = corpus.manifest["sources"]["en-web"]["statistics"]
        self.assertGreaterEqual(stats["contaminated"], 1)
        self.assertGreaterEqual(stats["duplicates"], 1)
        filter_ = BenchmarkFilter(self.benchmarks)
        with sqlite3.connect(":memory:") as database:
            database.execute("CREATE TABLE documents (digest BLOB PRIMARY KEY)")
            source = next(s for s in self.recipe["sources"] if s["id"] == "ja-web")
            documents = list(
                accepted_documents(
                    source,
                    self.recipe,
                    filter_,
                    database,
                    Counter(),
                    training_only=True,
                )
            )
        self.assertTrue(documents)
        self.assertTrue(
            all(
                split == "train" and not validation_document(digest, 0.25)
                for _, digest, split, _ in documents
            )
        )

    def test_同じ設定から同じ語彙とトークン列を再現する(self):
        first = self.prepare()
        second_tokenizer = self.root / "second-tokenizer"
        prepare_tokenizer(self.recipe_path, second_tokenizer, self.benchmarks)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(
                self.recipe_path,
                second_tokenizer,
                self.root / "second-corpus",
                self.benchmarks,
            )
        second = Corpus(self.root / "second-corpus")
        self.assertEqual(first.fingerprint, second.fingerprint)
        for split in ("train", "validation"):
            for language in ("ja", "en"):
                np.testing.assert_array_equal(
                    first.streams[split][language], second.streams[split][language]
                )

    def test_必要量に満たないデータ源を通知する(self):
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.benchmarks)
        self.rows["ja-web"] = []
        with self.assertRaisesRegex(ValueError, "short of the tokenizer sample budget"):
            prepare_tokenizer(
                self.recipe_path, self.root / "second-tokenizer", self.benchmarks
            )
        with self.assertRaisesRegex(ValueError, "short of its clean corpus budget"):
            prepare_corpus(
                self.recipe_path,
                self.root / "tokenizer",
                self.root / "corpus",
                self.benchmarks,
            )

    def test_変更されたコーパスの出典を検知する(self):
        self.prepare()
        with (self.root / "corpus/documents.jsonl").open("a") as output:
            output.write("changed\n")
        with self.assertRaisesRegex(ValueError, "provenance checksum"):
            validate_corpus(self.recipe_path, self.root / "corpus")

    def test_圧縮したトークン列で学習を保存し同じ更新から再開する(self):
        self.prepare()
        model, config = read_config(self.root / "training.toml")
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                self.root / "corpus",
                self.root / "continuous",
                model,
                config,
                torch.device("cpu"),
                max_steps=3,
            )
            train(
                self.root / "corpus",
                self.root / "resumed",
                model,
                config,
                torch.device("cpu"),
                max_steps=1,
            )
            train(
                self.root / "corpus",
                self.root / "resumed",
                model,
                config,
                torch.device("cpu"),
                resume=self.root / "resumed/last.pt",
                max_steps=3,
            )
        first, second = (
            load_checkpoint(self.root / name / "last.pt")
            for name in ("continuous", "resumed")
        )
        self.assertEqual(first["tokens_seen"], second["tokens_seen"])
        for name in first["model"]:
            torch.testing.assert_close(
                first["model"][name], second["model"][name], rtol=0, atol=0
            )

    def test_指示データの量から同じモデルの追加学習設定を作成する(self):
        self.prepare()
        source = self.root / "instruction-source"
        source.mkdir()
        files = {}
        for language in ("ja", "en"):
            file = source / f"{language}.jsonl"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": f"{language} Q{index}?"},
                        {"role": "assistant", "content": f"{language} A{index}."},
                    ]
                }
                for index in range(20)
            ]
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            files[file.name] = {"sha256": file_digest(file)}
        (source / "sources.json").write_text(json.dumps({"files": files}))
        prepare_instructions(
            source,
            self.root / "instructions",
            self.root / "tokenizer/tokenizer.json",
            context=128,
            validation_conversations=2,
            benchmarks=self.benchmarks,
        )
        result = write_sft_config(
            self.recipe_path,
            self.root / "instructions",
            self.root / "tokenizer",
            self.root / "sft.toml",
        )
        model, config = read_config(self.root / "sft.toml")
        self.assertEqual(model, read_config(self.root / "training.toml")[0])
        self.assertEqual(config.learning_rate, 0.00002)
        corpus = InstructionCorpus(self.root / "instructions")
        target = (
            2
            * 2
            * 128
            * min(len(corpus.rows["train"][lang][0]) for lang in ("ja", "en"))
        )
        self.assertGreaterEqual(result["scheduled_tokens"], target)
        self.assertEqual(config.steps, max(2, math.ceil(target / (4 * 8 * 128))))

    def test_固定した版の必要な列だけをParquetから読み取る(self):
        source = self.recipe["sources"][0]
        rows = self.rows[source["id"]][:5]
        file = self.root / "source.parquet"
        parquet.write_table(
            arrow.Table.from_pylist(
                [{**row, "embeddings": [1.0] * 1000} for row in rows]
            ),
            file,
            row_group_size=2,
        )
        from types import SimpleNamespace

        with (
            patch("huggingface_hub.HfApi") as api,
            patch("huggingface_hub.HfFileSystem") as filesystem,
        ):
            api.return_value.dataset_info.return_value = SimpleNamespace(
                sha=source["revision"],
                siblings=[
                    SimpleNamespace(rfilename=source["prefix"] + "fixture.parquet")
                ],
            )
            filesystem.return_value.open.side_effect = lambda *args, **kwargs: (
                file.open("rb")
            )
            loaded = list(iter_source(source, 42))
        self.assertEqual({row["url"] for row in loaded}, {row["url"] for row in rows})
        self.assertEqual(
            document_digest(rows[0]["text"], 42),
            document_digest("  " + rows[0]["text"].replace(" ", "\n ") + "  ", 42),
        )


if __name__ == "__main__":
    unittest.main()
