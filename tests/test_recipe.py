import contextlib
import gzip
import importlib.util
import io
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from pretrain.instructions import BenchmarkFilter, InstructionCorpus, prepare_instructions
from pretrain.recipe import (
    accepted_documents,
    document_digest,
    document_text,
    inspect_recipe,
    iter_source,
    prepare_corpus,
    prepare_tokenizer,
    read_recipe,
    recipe_fingerprint,
    validate_corpus,
    validate_source,
    validation_document,
    validation_fraction,
    write_sft_config,
)
from pretrain.training import load_checkpoint, read_config, train

ROOT = Path(__file__).resolve().parents[1]


def nested(name, value):
    for part in reversed(name.split(".")[1:]):
        value = {part: value}
    return name.split(".")[0], value


def passing_row(source, **row):
    """Build a document that every filter of the source accepts."""
    conditions = [
        *source.get("equal", {}).items(),
        *source.get("minimum", {}).items(),
        *list(source.get("any_minimum", {}).items())[:1],
    ]
    for name, value in conditions:
        key, value = nested(name, value)
        if isinstance(value, dict) and isinstance(row.get(key), dict):
            value = {**row[key], **value}
        row[key] = value
    return row


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe_path = self.root / "recipe.toml"
        for name in ("sources.toml", "previous-shards.toml"):
            (self.root / name).write_text((ROOT / "configs" / name).read_text())
        shutil.copytree(
            ROOT / "configs/continue-v4-shards", self.root / "continue-v4-shards"
        )
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
            self.rows[source["id"]] = [
                passing_row(
                    source,
                    text=f"{text} {source['id']}:{index}",
                    url=f"https://example.org/{source['id']}/{index}",
                    id=f"{source['id']}:{index}",
                    source_file=source["prefix"] + "fixture.parquet",
                )
                for index in range(100)
            ]
        self.source_patch = patch(
            "pretrain.recipe.iter_source",
            side_effect=lambda source, seed, part=None: iter(
                self.rows[source["id"]][
                    slice(part["index"], None, part["count"]) if part else slice(None)
                ]
            ),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def prepare(self):
        tokenizer = self.root / "tokenizer"
        prepare_tokenizer(self.recipe_path, tokenizer, self.benchmarks)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(self.recipe_path, tokenizer, self.root, self.benchmarks)
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
            plan["commands"][-1][-11:],
            [
                "hellaswag",
                "arc_easy",
                "ja_leaderboard_jcommonsenseqa",
                "sciq",
                "piqa",
                "lambada_openai",
                "blimp",
                "kuci",
                "ja_leaderboard_xwinograd",
                "ja_leaderboard_jnli",
                "ifeval",
            ],
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

    def test_縮小した学習計画を既定として学習量と保存先を表示する(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["recipe"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["parameters"], 537_982_464)
        self.assertEqual(plan["nominal_training_tokens"], 15_000_000_000)
        self.assertEqual(plan["scheduled_training_tokens"], 15_000_403_968)
        self.assertEqual(plan["minimum_packed_bytes"], 30_020_000_000)
        self.assertEqual(
            sum(s["training_tokens"] for s in plan["sources"] if s["language"] == "ja"),
            7_500_000_000,
        )
        self.assertEqual(plan["model"]["context"], 4096)
        training = [command for command in plan["commands"] if command[3] == "train"]
        self.assertEqual(
            training[0][training[0].index("--config") + 1],
            "configs/bilingual-500m.toml",
        )
        self.assertEqual(
            training[0][training[0].index("--data") + 1], "data/pretrain-500m/corpus"
        )
        self.assertEqual(
            training[0][training[0].index("--output") + 1], "runs/pretrain-500m"
        )
        self.assertEqual(
            training[1][training[1].index("--initialize-from") + 1],
            "runs/pretrain-500m/last.pt",
        )
        self.assertEqual(
            training[1][training[1].index("--output") + 1], "runs/instructions-500m"
        )

    def test_指定した学習計画に合わせてデータと重みの保存先を選択する(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["recipe", "--config", str(ROOT / "configs/pretrain-1b.toml")])
        plan = json.loads(output.getvalue())
        training = [command for command in plan["commands"] if command[3] == "train"]
        self.assertEqual(
            training[0][training[0].index("--data") + 1], "data/pretrain-1b/corpus"
        )
        self.assertEqual(
            training[0][training[0].index("--output") + 1], "runs/pretrain-1b"
        )
        self.assertEqual(
            training[1][training[1].index("--initialize-from") + 1],
            "runs/pretrain-1b/last.pt",
        )

    def test_各データ源の割当を満たす日英コーパスを作成する(self):
        corpus = self.prepare()
        result = validate_corpus(self.recipe_path, self.root)
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
            source = next(s for s in self.recipe["sources"] if s["id"] == "ja-hq")
            documents = list(
                accepted_documents(
                    source,
                    self.recipe,
                    filter_,
                    database,
                    Counter(),
                    training_fraction=0.25,
                )
            )
        self.assertTrue(documents)
        self.assertTrue(
            all(not validation_document(digest, 0.25) for _, digest, _ in documents)
        )

    def test_同じ設定から同じ語彙とトークン列を再現する(self):
        first = self.prepare()
        second_root = self.root / "second"
        second_tokenizer = second_root / "tokenizer"
        prepare_tokenizer(self.recipe_path, second_tokenizer, self.benchmarks)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(
                self.recipe_path, second_tokenizer, second_root, self.benchmarks
            )
        second = Corpus(second_root / "corpus")
        self.assertEqual(first.fingerprint, second.fingerprint)
        for split in ("train", "validation"):
            for language in ("ja", "en"):
                np.testing.assert_array_equal(
                    first.streams[split][language], second.streams[split][language]
                )

    def test_データ源を分けて並列に読んでも同じコーパスを作り重複を除く(self):
        text = self.recipe_path.read_text().replace(
            "[corpus]\n", "[corpus]\npart_tokens = 300\n"
        )
        self.recipe_path.write_text(text)
        rows = self.rows["ja-hq"]
        for index, row in enumerate(rows):
            row["text"] = "自然科学と歴史を説明する短い文章です。" * 3 + str(index)
        rows[1]["text"] = rows[0]["text"]
        corpora = []
        for workers in (1, 2):
            root = self.root / f"workers-{workers}"
            prepare_tokenizer(self.recipe_path, root / "tokenizer", self.benchmarks)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_corpus(
                    self.recipe_path,
                    root / "tokenizer",
                    root,
                    self.benchmarks,
                    workers,
                )["main"]
            corpora.append(Corpus(root / "corpus"))
            self.assertEqual(
                manifest["sources"]["ja-hq"]["statistics"]["duplicates"], 1
            )
        self.assertEqual(corpora[0].fingerprint, corpora[1].fingerprint)

    def test_並列に作る部分のプロセスが終わってしまうとコーパス作りを止める(self):
        """Stops at once when a part's process dies, instead of waiting for the part."""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.benchmarks)
        with (
            patch("pretrain.recipe.prepare_part", side_effect=lambda *_: os._exit(1)),
            self.assertRaisesRegex(RuntimeError, "exit code 1"),
        ):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root, self.benchmarks, 2
            )

    def test_必要量に満たないデータ源を通知する(self):
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.benchmarks)
        self.rows["ja-hq"] = []
        with self.assertRaisesRegex(ValueError, "short of the tokenizer sample budget"):
            prepare_tokenizer(
                self.recipe_path, self.root / "second-tokenizer", self.benchmarks
            )
        with self.assertRaisesRegex(ValueError, "short of its clean corpus budget"):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root, self.benchmarks
            )

    def test_変更されたコーパスの出典を検知する(self):
        self.prepare()
        with (self.root / "corpus/documents.jsonl").open("a") as output:
            output.write("changed\n")
        with self.assertRaisesRegex(ValueError, "provenance checksum"):
            validate_corpus(self.recipe_path, self.root)

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

    def phase_recipe(self, extra=""):
        """A two-phase recipe that reuses the fixture's tokenizer and sources."""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.benchmarks)
        (self.root / "phases.toml").write_text(
            '[model]\narchitecture="llama"\nwidth=24\nheads=4\nkv_heads=2\nlayers=1\nintermediate_size=64\ncontext=128\n'
            "[training]\nsteps=6\nbatch_size=2\naccumulation_steps=1\nwarmup_steps=1\neval_batches=1\n"
            'schedule="wsd"\ndecay_steps=2\nsampling="sequential"\n'
        )
        path = self.root / "phased.toml"
        digest = file_digest(self.root / "tokenizer/tokenizer.json")
        path.write_text(
            f"""training_config = "phases.toml"
sources = "sources.toml"
seed = 42
{extra}
[tokenizer]
vocabulary_size = 258
reuse_from = "{self.root / "tokenizer/tokenizer.json"}"
reuse_sha256 = "{digest}"

[corpus]
validation_tokens_per_language = 200
minimum_characters = 50
maximum_characters = 100000

[evaluation]
framework = "lm-eval"
base_tasks = ["arc_easy"]

[[corpora]]
name = "stable"
training_tokens = 1000
mix = {{ ja-hq = 45, ja-wiki = 5, en-web = 50 }}

[[corpora]]
name = "decay"
training_tokens = 500
mix = {{ ja-hq = 25, ja-wiki = 25, en-web = 25, en-wiki = 25 }}
"""
        )
        return path

    def test_データ源ごとの条件に合う文書だけを残す(self):
        source = {
            "id": "fixture",
            "language": "ja",
            "equal": {"meta.kind": "modern"},
            "minimum": {"quality": 4},
            "maximum": {"advertisement": 2},
            "any_minimum": {"educational": 4, "cultural": 4},
            "maximum_characters": 400,
        }
        corpus = self.recipe["corpus"]
        text = "日本語の文章です。" * 20
        row = {
            "text": text,
            "meta": {"kind": "modern"},
            "quality": 4,
            "advertisement": 1,
            "educational": 1,
            "cultural": 5,
        }
        self.assertEqual(document_text(row, source, corpus), text)
        for change in (
            {"meta": {"kind": "old"}},
            {"meta": None},
            {"quality": 3},
            {"quality": float("nan")},
            {"advertisement": 3},
            {"cultural": 1},
            {"text": text * 3},
        ):
            self.assertIsNone(document_text({**row, **change}, source, corpus))

    def test_カタログに足したデータ源は既存の計画を変えない(self):
        before = recipe_fingerprint(self.recipe_path)
        with (self.root / "sources.toml").open("a") as catalog:
            catalog.write(
                '\n[[sources]]\nid = "en-new"\nlanguage = "en"\nrepository = "org/new"\n'
                f'revision = "{"0" * 40}"\nprefix = "data/"\nlicense = "CC0-1.0"\n'
                'terms_url = "https://example.org/terms"\n'
            )
        self.assertEqual(recipe_fingerprint(self.recipe_path), before)
        with (self.root / "sources.toml").open("a") as catalog:
            catalog.write("minimun = { score = 3 }\n")
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            read_recipe(self.recipe_path)

    def test_記録した語彙を再利用し異なる語彙を拒否する(self):
        recipe = self.phase_recipe()
        prepare_tokenizer(recipe, self.root / "reused", self.benchmarks)
        self.assertEqual(
            file_digest(self.root / "reused/tokenizer.json"),
            file_digest(self.root / "tokenizer/tokenizer.json"),
        )
        text = recipe.read_text()
        start = text.index('reuse_sha256 = "') + len('reuse_sha256 = "')
        recipe.write_text(text[:start] + "0" * 64 + text[start + 64 :])
        with self.assertRaisesRegex(ValueError, "does not match its checksum"):
            prepare_tokenizer(recipe, self.root / "rejected", self.benchmarks)

    def test_段階ごとのコーパスを重複なく作り段階をつなぐ手順を示す(self):
        recipe = self.phase_recipe('initialize_from = "runs/base/model.pt"\n')
        prepare_tokenizer(recipe, self.root / "phased/tokenizer", self.benchmarks)
        with contextlib.redirect_stdout(io.StringIO()):
            manifests = prepare_corpus(
                recipe,
                self.root / "phased/tokenizer",
                self.root / "phased",
                self.benchmarks,
            )
        self.assertEqual(set(manifests), {"stable", "decay"})
        self.assertTrue(validate_corpus(recipe, self.root / "phased")["ready"])
        digests = {
            name: {
                json.loads(line)["digest"]
                for line in (self.root / f"phased/corpus-{name}/documents.jsonl")
                .read_text()
                .splitlines()
            }
            for name in manifests
        }
        self.assertTrue(digests["stable"].isdisjoint(digests["decay"]))
        # The decay corpus continues each shared source instead of rereading it.
        for source in ("ja-hq", "en-web"):
            self.assertNotIn(
                "duplicates", manifests["decay"]["sources"][source]["statistics"]
            )
        commands = [
            command
            for command in inspect_recipe(recipe, self.root / "phased")["commands"]
            if command[3] == "train"
        ]
        self.assertEqual(
            commands[0][commands[0].index("--initialize-from") + 1],
            "runs/base/model.pt",
        )
        self.assertEqual(commands[0][commands[0].index("--max-steps") + 1], "4")
        self.assertEqual(
            commands[1][commands[1].index("--continue-from") + 1],
            "runs/phased-stable/last.pt",
        )
        self.assertNotIn("--max-steps", commands[1])

    def test_保存したチェックポイントから段階を続け終了位置をその続きから数える(self):
        recipe = self.phase_recipe(
            'continue_from = { checkpoint = "runs/earlier-stable/last.pt", step = 10 }\n'
        )
        config = self.root / "phases.toml"
        config.write_text(config.read_text().replace("\nsteps=6\n", "\nsteps=16\n"))
        inspected = inspect_recipe(recipe, self.root / "phased")
        commands = [
            command for command in inspected["commands"] if command[3] == "train"
        ]
        self.assertEqual(
            commands[0][commands[0].index("--continue-from") + 1],
            "runs/earlier-stable/last.pt",
        )
        self.assertNotIn("--initialize-from", commands[0])
        self.assertEqual(commands[0][commands[0].index("--max-steps") + 1], "14")
        self.assertEqual(
            commands[1][commands[1].index("--continue-from") + 1],
            "runs/phased-stable/last.pt",
        )
        self.assertEqual(inspected["scheduled_training_tokens"], 6 * 256)
        config.write_text(config.read_text().replace("\nsteps=16\n", "\nsteps=6\n"))
        with self.assertRaisesRegex(ValueError, "token budget"):
            read_recipe(recipe)

    def test_検証用の割合を学習量と検証量の比から決める(self):
        recipe = read_recipe(self.phase_recipe())
        stable, decay = recipe["corpora"]
        larger = {**stable, "training_tokens": 100_000}
        self.assertAlmostEqual(
            validation_fraction(recipe, larger), 1.25 * 0.004 / 1.004
        )
        self.assertEqual(validation_fraction(recipe, decay), 0.25)
        self.assertEqual(
            validation_fraction(recipe, {**stable, "validation_fraction": 0.1}), 0.1
        )

    def test_学習率を下げ始める位置と最後の段階がずれた計画を拒否する(self):
        recipe = self.phase_recipe()
        (self.root / "phases.toml").write_text(
            (self.root / "phases.toml")
            .read_text()
            .replace("decay_steps=2", "decay_steps=3")
        )
        with self.assertRaisesRegex(ValueError, "begins to decay"):
            read_recipe(recipe)

    def test_JSON行から必要な項目だけを読み取る(self):
        import zstandard

        rows = [
            {"text": f"text {index}", "meta": {"kind": "modern"}, "extra": "x" * 100}
            for index in range(3)
        ]
        lines = "".join(json.dumps(row) + "\n" for row in rows).encode()
        from types import SimpleNamespace

        for form, data in (
            ("jsonl", lines),
            ("jsonl.gz", gzip.compress(lines)),
            ("jsonl.zst", zstandard.ZstdCompressor().compress(lines)),
        ):
            source = {
                "id": "fixture",
                "repository": "org/data",
                "revision": "0" * 40,
                "prefix": "part",
                "format": form,
                "equal": {"meta.kind": "modern"},
            }
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.HfFileSystem") as filesystem,
            ):
                api.return_value.dataset_info.return_value = SimpleNamespace(
                    sha=source["revision"],
                    siblings=[
                        SimpleNamespace(rfilename="part-0." + form),
                        SimpleNamespace(rfilename="part-1.parquet"),
                    ],
                )
                filesystem.return_value.open.side_effect = lambda *a, data=data, **k: (
                    io.BytesIO(data)
                )
                loaded = list(iter_source(source, 42))
            self.assertEqual(
                sorted(row["text"] for row in loaded), [r["text"] for r in rows]
            )
            self.assertTrue(all(row["meta"] == {"kind": "modern"} for row in loaded))
            self.assertTrue(all("extra" not in row for row in loaded))
            self.assertEqual({row["source_file"] for row in loaded}, {"part-0." + form})

    def test_GitLabの固定した版から読み途中で切れた応答は続きから読む(self):
        revision = "a" * 40
        rows = [{"text": f"数学の問題 {index}", "category": "math"} for index in range(400)]
        data = gzip.compress("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())
        files = {
            "ja/math/questions.jsonl.gz": data,
            "ja/math/excluded.jsonl.gz": gzip.compress(b'{"text": "excluded"}\n'),
            "ja/coding/programs.jsonl.gz": gzip.compress(b'{"text": "code"}\n'),
        }
        project = "/api/v4/projects/" + urllib.parse.quote("datasets/corpus", safe="")
        ranges, cut = [], [True]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == f"{project}/repository/commits/{revision}":
                    body, status, headers = json.dumps({"id": revision}).encode(), 200, {}
                elif path == f"{project}/repository/tree":
                    asked = urllib.parse.parse_qs(query)
                    self.server_test.assertEqual(asked["ref"], [revision])
                    entries = [
                        {"path": name, "type": "blob"}
                        for name in files
                        if name.startswith(asked["path"][0] + "/")
                    ]
                    body, status, headers = json.dumps(entries).encode(), 200, {}
                else:
                    name = path.removeprefix(f"/datasets/corpus/-/raw/{revision}/")
                    content = files[urllib.parse.unquote(name)]
                    start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
                    ranges.append(start)
                    body, status = content[start:], 206
                    headers = {"Content-Range": f"bytes {start}-{len(content) - 1}/{len(content)}"}
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if status == 206 and cut[0]:
                    # The first response for a file stops partway through.
                    body, cut[0] = body[: len(body) // 2], False
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        Handler.server_test = self
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        source = {
            "id": "fixture",
            "host": f"http://127.0.0.1:{server.server_port}",
            "repository": "datasets/corpus",
            "revision": revision,
            "prefix": "ja/math/",
            "format": "jsonl.gz",
            "exclude_files": ["ja/math/excluded.jsonl.gz"],
        }
        with patch("pretrain.recipe.time.sleep"):
            loaded = list(iter_source(source, 42))
        self.assertEqual(sorted(row["text"] for row in loaded), sorted(row["text"] for row in rows))
        self.assertEqual({row["source_file"] for row in loaded}, {"ja/math/questions.jsonl.gz"})
        # The second request asks for the rest after the bytes that arrived.
        self.assertEqual(ranges, [0, len(data) // 2])

    def test_GitLabのデータ源はhttpsの接続先だけを受け付ける(self):
        source = {
            "id": "fixture",
            "language": "ja",
            "repository": "datasets/corpus",
            "revision": "a" * 40,
            "prefix": "ja/math/",
            "license": "Apache-2.0",
            "terms_url": "https://example.org/terms",
        }
        validate_source({**source, "host": "https://gitlab.example.org"})
        with self.assertRaisesRegex(ValueError, "https address"):
            validate_source({**source, "host": "http://gitlab.example.org"})


if __name__ == "__main__":
    unittest.main()
