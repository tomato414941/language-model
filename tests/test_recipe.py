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
        "language_model_evaluation",
        "pyarrow",
        "huggingface_hub",
    )
):
    raise unittest.SkipTest("Install training and corpus dependencies")

import numpy as np
import pyarrow as arrow
import torch
from pyarrow import parquet
from tokenizers import Tokenizer

from pretrain.__main__ import main
from pretrain.data import Corpus, file_digest
from pretrain.conversation import conversation_text
from pretrain.instructions import InstructionCorpus, prepare_instructions
from pretrain.overlap import EvaluationOverlap, artifacts
from pretrain.recipe import (
    accepted_documents,
    data_fingerprint,
    document_digest,
    document_text,
    inspect_recipe,
    iter_source,
    prepare_corpus,
    prepare_part,
    prepare_tokenizer,
    read_recipe,
    source_columns,
    source_sizes,
    source_text,
    validate_corpus,
    validate_source,
    validation_document,
    validation_fraction,
    write_sft_config,
)
from pretrain.training import load_checkpoint, read_config, train
from tests.helpers import evaluation

ROOT = Path(__file__).resolve().parents[1]


def commands(plan, kind):
    """The argument lists of the plan's experiment commands of one kind."""
    return [command["argv"] for command in plan["experiment"] if command["kind"] == kind]


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
        # An instruction prompt of the evaluation, which no training text may hold.
        self.question = "Describe rain without using commas."
        self.evaluation = evaluation(self, self.root / "evaluation", prompts={"en": [self.question]})
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
        self.layouts = {}
        self.source_patch = patch(
            "pretrain.recipe.iter_source",
            side_effect=lambda source, seed, part=None, extra=(): (
                row
                for index, row in enumerate(self.rows[source["id"]])
                if part is None or self.file(source, index) in part["files"]
            ),
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        sizes = patch(
            "pretrain.recipe.source_sizes",
            side_effect=lambda source: {
                name: 1000 * count
                for name, count in Counter(
                    self.file(source, index)
                    for index in range(len(self.rows[source["id"]]))
                ).items()
            },
        )
        sizes.start()
        self.addCleanup(sizes.stop)

    def file(self, source, index):
        """The file holding a source's row, of 1,000 bytes a row.

        The rows take turns in four files unless the test lays them out.
        """
        layout = self.layouts.get(source["id"], lambda index: f"fixture-{index % 4}")
        return f"{source['prefix']}{layout(index)}.parquet"

    def prepare(self):
        tokenizer = self.root / "tokenizer"
        prepare_tokenizer(self.recipe_path, tokenizer, self.evaluation)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(self.recipe_path, tokenizer, self.root, self.evaluation)
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
            plan["experiment"][-1]["argv"][3:],
            [
                "knowledge",
                "--checkpoint",
                "runs/instructions-1b/last.pt",
                "--device",
                "cuda",
                "--artifacts",
                "data/evaluation",
                "--output",
                "runs/instructions-1b/knowledge.json",
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

    def test_事前学習の後のモデルの知識と日英の釣り合い_指示学習の後のモデルの指示に従う力と知識を測る手順にする(self):
        """事前学習の後は最後の重みと平均した重みの知識と、最後の重みの日英の釣り合いを測り、指示学習の後は指示に従う力と、
        参考の知識を測る。"""
        plan = inspect_recipe(ROOT / "configs/pretrain-v6.toml", "data/pretrain-v6")
        measured = [(command["kind"], command["argv"][command["argv"].index("--checkpoint") + 1],
                     command["leaves"]) for command in plan["experiment"]
                    if command["kind"] in ("knowledge", "balance", "instructions")]
        self.assertEqual(measured, [
            ("knowledge", "runs/pretrain-v6-decay/last.pt", ["runs/pretrain-v6-decay/knowledge.json"]),
            ("balance", "runs/pretrain-v6-decay/last.pt", ["runs/pretrain-v6-decay/balance.json"]),
            ("knowledge", "runs/pretrain-v6-decay/averaged.pt", ["runs/pretrain-v6-decay/knowledge-averaged.json"]),
            ("instructions", "runs/instructions-v6/last.pt", ["runs/instructions-v6/instructions.json"]),
            ("knowledge", "runs/instructions-v6/last.pt", ["runs/instructions-v6/knowledge.json"]),
        ])
        for command in commands(plan, "knowledge") + commands(plan, "balance"):
            self.assertEqual(command[command.index("--artifacts") + 1], "data/evaluation")
        for command in commands(plan, "instructions"):
            self.assertEqual(command[command.index("--work") + 1], "work/instructions")

    def test_減衰の間に残した重みの平均も最後の重みと別に測る手順にする(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v6.toml", "data/pretrain-v6")
        experiment = [command["argv"] for command in plan["experiment"]]
        measured = commands(plan, "knowledge")
        [average] = commands(plan, "average")
        self.assertEqual(average[average.index("--run") + 1], "runs/pretrain-v6-decay")
        self.assertEqual(average[average.index("--count") + 1], "10")
        self.assertEqual(measured[1][measured[1].index("--checkpoint") + 1], average[average.index("--output") + 1])
        self.assertLess(experiment.index(average), experiment.index(measured[1]))
        tuning = [command for command in commands(plan, "train") if "sft.toml" in command[command.index("--config") + 1]]
        self.assertEqual(tuning[0][tuning[0].index("--initialize-from") + 1], "runs/pretrain-v6-decay/last.pt")

    def test_語彙とコーパスと指示データの準備は評価と重なる文章を教える成果物を読む(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v7.toml", "data/pretrain-v7")
        overlaps = [f"data/evaluation/{name}" for name in artifacts()]
        reads = {command["name"]: command["reads"] for command in plan["preparation"]}
        self.assertEqual(reads["tokenizer"], overlaps)
        self.assertEqual(reads["corpus"][1:], overlaps)
        self.assertEqual(reads["prepare-instructions"][2:], overlaps)

    def test_実験の各手順は前の手順か準備が残したものだけを読む(self):
        """実験の手順が読むものは、準備が残したデータか、それより前の手順が残したものに限る。"""
        plan = inspect_recipe(ROOT / "configs/pretrain-v7.toml", "data/pretrain-v7")
        left = {path for command in plan["preparation"] for path in command["leaves"]}
        for command in plan["experiment"]:
            for path in command["reads"]:
                self.assertTrue(any(path == made or path.startswith(f"{made}/") for made in left),
                                (command["name"], path))
            left.update(command["leaves"])
        self.assertEqual(len({command["name"] for command in plan["experiment"]}), len(plan["experiment"]))

    def test_学習の手順は設定と始めと終わりの段を示す(self):
        """続けて学ぶ段階は前の段階の終わりの段から始め、最後の段階は設定の最後の段まで学ぶ。"""
        plan = inspect_recipe(ROOT / "configs/pretrain-v7.toml", "data/pretrain-v7")
        trainings = {command["name"]: command["training"] for command in plan["experiment"] if "training" in command}
        stable, late, decay = (trainings[f"pretrain-v7-{name}"] for name in ("stable", "late", "decay"))
        self.assertEqual((stable["from_step"], late["from_step"], decay["from_step"]),
                         (0, stable["to_step"], late["to_step"]))
        self.assertIsNone(decay["to_step"])
        self.assertEqual(trainings["instructions-v7"]["settings"], "data/pretrain-v7/sft.toml")

    def test_重みを残さない計画は平均を作らない(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v5.toml", "data/pretrain-v5")
        self.assertFalse(commands(plan, "average"))

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
        training = commands(plan, "train")
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
        training = commands(plan, "train")
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

    def test_コンパイルする計画は指示学習を含むすべての学習をコンパイルする(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v6.toml", "data/pretrain-v6")
        training = commands(plan, "train")
        self.assertEqual(["--compile" in command for command in training], [True, True, True, True])

    def test_コンパイルを指定しない計画はコンパイルしない(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v5.toml", "data/pretrain-v5")
        self.assertFalse(any("--compile" in command["argv"] for command in [*plan["preparation"], *plan["experiment"]]))

    def test_パッチで読む段階の重みから次の段階を始める(self):
        plan = inspect_recipe(ROOT / "configs/pretrain-v6.toml", "data/pretrain-v6")
        patches, stable, decay = commands(plan, "train")[:3]
        self.assertTrue(patches[patches.index("--config") + 1].endswith("model-v6-patch.toml"))
        self.assertEqual(patches[patches.index("--data") + 1], "data/pretrain-v6/corpus-patch")
        self.assertEqual(
            stable[stable.index("--initialize-from") + 1], "runs/pretrain-v6-patch/last.pt"
        )
        self.assertEqual(decay[decay.index("--continue-from") + 1], "runs/pretrain-v6-stable/last.pt")
        self.assertEqual(
            plan["nominal_training_tokens"], 5333 * 32 * (8192 * 4 + 3) + 2667 * 1_048_576
        )

    def test_パッチの設定が合わない計画は断る(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "configs"
            shutil.copytree(ROOT / "configs", root)
            recipe = root / "pretrain-v6.toml"
            text = recipe.read_text()
            recipe.write_text(text.replace('patch_config = "model-v6-patch.toml"\n', ""))
            with self.assertRaisesRegex(ValueError, "patch_config"):
                read_recipe(recipe)
            recipe.write_text(text)
            settings = root / "model-v6-patch.toml"
            patches = settings.read_text()
            settings.write_text(patches.replace("steps = 5333", "steps = 5300"))
            with self.assertRaisesRegex(ValueError, "cover the patch corpus"):
                read_recipe(recipe)
            settings.write_text(patches.replace("layers = 32", "layers = 24"))
            with self.assertRaisesRegex(ValueError, "same model"):
                read_recipe(recipe)

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

    def test_質の欄を指定したデータ源は読み順で選ぶのと同じ文書を点の低い順に書く(self):
        """Writes a scored source's training documents, the same ones as without the order, unscored first and then from the lowest score up."""
        for index, row in enumerate(self.rows["en-web"]):
            row["score"] = 3 + index * 37 % 100 / 100
        for index, row in enumerate(self.rows["en-wiki"]):
            row["rank"] = None if index % 5 == 0 else index * 13 % 50
        # About ten documents of each English source.
        text = self.recipe_path.read_text().replace(
            "mix = { ja-hq = 45, ja-wiki = 5, en-web = 40, en-math = 5, en-wiki = 5 }",
            "corpus_tokens = 40000\nmix = { ja-hq = 45, ja-wiki = 5, en-web = 25, en-math = 5, en-wiki = 20 }",
        )
        self.recipe_path.write_text(text)
        unordered = self.prepare()
        ordered_path = self.root / "ordered.toml"
        ordered_path.write_text(
            text.replace('"training.toml"', '"ordered-training.toml"').replace(
                "corpus_tokens = 40000\n", 'corpus_tokens = 40000\nquality = { en-web = "score", en-wiki = "rank" }\n'
            )
        )
        (self.root / "ordered-training.toml").write_text(
            (self.root / "training.toml").read_text() + 'sampling = "ordered"\n'
        )
        prepare_tokenizer(ordered_path, self.root / "ordered-tokenizer", self.evaluation)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(ordered_path, self.root / "ordered-tokenizer", self.root / "ordered", self.evaluation)
        ordered = Corpus(self.root / "ordered/corpus")

        def records(corpus, source_id=None, split="train"):
            lines = (corpus.path / "documents.jsonl").read_text().splitlines()
            return [
                record for record in map(json.loads, lines)
                if record["split"] == split and source_id in (None, record["source"])
            ]

        def values(source_id, field):
            return [
                self.rows[source_id][int(record["id"].split(":")[1])][field]
                for record in records(ordered, source_id)
            ]

        for source_id in self.rows:
            for split in ("train", "validation"):
                self.assertEqual(
                    sorted(record["digest"] for record in records(unordered, source_id, split)),
                    sorted(record["digest"] for record in records(ordered, source_id, split)),
                )
        self.assertEqual(records(ordered, "en-math"), records(unordered, "en-math"))
        scores = values("en-web", "score")
        self.assertGreater(len(scores), 5)
        self.assertEqual(scores, sorted(scores))
        ranks = values("en-wiki", "rank")
        unscored = ranks.count(None)
        self.assertTrue(0 < unscored < len(ranks) - 2)
        self.assertEqual(ranks[unscored:], sorted(ranks[unscored:]))
        self.assertEqual(ranks[:unscored], [None] * unscored)
        self.assertEqual(
            ordered.manifest["sources"]["en-wiki"]["statistics"]["unscored_documents"], unscored
        )
        self.assertEqual(ordered.manifest["quality"], {"en-web": "score", "en-wiki": "rank"})
        # Each line of the provenance still describes the tokens at its place in the stream.
        languages = {source["id"]: source["language"] for source in self.recipe["sources"]}
        offsets = Counter()
        for record in records(ordered):
            language = languages[record["source"]]
            tokens = ordered.streams["train"][language][offsets[language] : offsets[language] + record["tokens"]]
            offsets[language] += record["tokens"]
            self.assertTrue(ordered.tokenizer.decode(tokens.tolist()).endswith(record["id"]))
        for language in ("ja", "en"):
            self.assertEqual(offsets[language], len(ordered.streams["train"][language]))

    def test_並べる欄を持たないデータ源はコーパスを作る途中で知らせる(self):
        """Stops building when a source's rows hold none of the field its corpus is ordered by."""
        self.recipe_path.write_text(
            self.recipe_path.read_text().replace("mix = {", 'quality = { en-web = "scroe" }\nmix = {', 1)
        )
        (self.root / "training.toml").write_text(
            (self.root / "training.toml").read_text() + 'sampling = "ordered"\n'
        )
        with self.assertRaisesRegex(ValueError, "en-web rows hold no scroe"):
            self.prepare()

    def test_質の順に書くコーパスと質の順に読む学習は対にして受け付ける(self):
        """Accepts a corpus written in quality order only with training that reads in that order, and the reverse."""
        text = self.recipe_path.read_text()
        cases = (
            (text.replace("mix = {", 'quality = { en-web = "score" }\nmix = {', 1), ""),
            (text, 'sampling = "ordered"\n'),
            (text.replace("mix = {", 'quality = { en-edu-top = "score" }\nmix = {', 1), 'sampling = "ordered"\n'),
            (text.replace("mix = {", 'quality = { en-web = 3 }\nmix = {', 1), 'sampling = "ordered"\n'),
        )
        for recipe, sampling in cases:
            self.recipe_path.write_text(recipe)
            (self.root / "training.toml").write_text(
                (self.root / "training.toml").read_text().replace('sampling = "ordered"\n', "") + sampling
            )
            with self.assertRaisesRegex(ValueError, "quality"):
                read_recipe(self.recipe_path)
        self.recipe_path.write_text(text.replace("mix = {", 'quality = { en-web = "score" }\nmix = {', 1))
        self.assertEqual(read_recipe(self.recipe_path)["corpora"][0]["quality"], {"en-web": "score"})

    def test_言語の割合を配合から決めどの言語にも同じ量の検証用文書を割り当てる(self):
        """日英の割合は配合で決まり、どの言語も同じ量の検証用の文書を持つ。配合のない言語があるレシピや、合計が100%でないレシピは断る。"""
        text = self.recipe_path.read_text()
        mix = "mix = { ja-hq = 45, ja-wiki = 5, en-web = 40, en-math = 5, en-wiki = 5 }"
        self.assertIn(mix, text)
        self.recipe_path.write_text(
            text.replace(
                mix, "mix = { ja-hq = 60, ja-wiki = 10, en-web = 20, en-math = 5, en-wiki = 5 }"
            )
        )
        corpus = self.prepare()
        train, validation = Counter(), Counter()
        for source in corpus.manifest["sources"].values():
            train[source["language"]] += source["targets"]["train"]
            validation[source["language"]] += source["targets"]["validation"]
        self.assertEqual(train, {"ja": 1400, "en": 600})
        # Each source's held-out tokens round down, so a language may hold a few fewer.
        for language in ("ja", "en"):
            self.assertLessEqual(200 - validation[language], 2)
        self.assertTrue(validate_corpus(self.recipe_path, self.root)["ready"])
        for wrong in (
            "mix = { ja-hq = 60, ja-wiki = 10, en-web = 20, en-math = 5 }",
            "mix = { ja-hq = 90, ja-wiki = 10 }",
        ):
            self.recipe_path.write_text(text.replace(mix, wrong))
            with self.assertRaisesRegex(ValueError, "100 percent, with a share for each language"):
                read_recipe(self.recipe_path)

    def test_作ったあとで学習の設定や読むトークン数を変えても同じコーパスを使う(self):
        """コーパスを作ったあとで学習率やモデル、評価、学習で読むトークン数を変えても、同じコーパスが確認を通る。作るトークン数を変えると別のコーパスとして断る。"""
        self.prepare()
        training = self.root / "training.toml"
        training.write_text(
            training.read_text().replace("width=24", "width=32").replace("steps=8", "steps=4")
            + "learning_rate=1e-3\n"
        )
        text = self.recipe_path.read_text().replace('"hellaswag", ', "")
        self.recipe_path.write_text(
            text.replace("training_tokens = 2000", "training_tokens = 1000\ncorpus_tokens = 2000")
        )
        self.assertTrue(validate_corpus(self.recipe_path, self.root)["ready"])
        self.recipe_path.write_text(
            text.replace("training_tokens = 2000", "training_tokens = 1000\ncorpus_tokens = 3000")
        )
        with self.assertRaisesRegex(ValueError, "different recipe"):
            validate_corpus(self.recipe_path, self.root)
        self.recipe_path.write_text(
            text.replace("training_tokens = 2000", "training_tokens = 1000\ncorpus_tokens = 900")
        )
        with self.assertRaisesRegex(ValueError, "corpus_tokens"):
            read_recipe(self.recipe_path)

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
        filter_ = EvaluationOverlap(self.evaluation)
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

    def test_GPT4の区切りで数字を一桁ずつにし空きの記号を確保した語彙を作る(self):
        """GPT-4 の区切りの規則で数字を一桁ずつに分け、文書の始まりと終わりの後ろに空きの記号を置いた語彙を作る。"""
        recipe = self.recipe_path.read_text()
        self.assertIn("vocabulary_size = 258", recipe)
        self.recipe_path.write_text(
            recipe.replace(
                "vocabulary_size = 258",
                'vocabulary_size = 300\npretokenizer = "gpt4"\nsingle_digits = true\nreserved_tokens = 4',
            )
        )
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        tokenizer = Tokenizer.from_file(str(self.root / "tokenizer/tokenizer.json"))
        self.assertEqual(tokenizer.get_vocab_size(), 300)
        self.assertEqual(
            [tokenizer.token_to_id(token) for token in ("<|bos|>", "<|eos|>", "<|reserved_0|>", "<|reserved_3|>")],
            [0, 1, 2, 5],
        )
        text = "歴史の説明は1400年前から。\n    return x"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(tokenizer.decode(ids), text)
        pieces = [tokenizer.decode([index]) for index in ids]
        self.assertEqual([piece for piece in pieces if any(c.isdigit() for c in piece)], ["1", "4", "0", "0"])
        # The indentation is not joined to the line break before it, as GPT-2's split joins them.
        self.assertFalse(any("\n " in piece for piece in pieces))
        self.recipe_path.write_text(recipe.replace("vocabulary_size = 258", 'vocabulary_size = 258\npretokenizer = "gpt3"'))
        with self.assertRaisesRegex(ValueError, "pretokenizer"):
            read_recipe(self.recipe_path)

    def test_同じ設定から同じ語彙とトークン列を再現する(self):
        first = self.prepare()
        second_root = self.root / "second"
        second_tokenizer = second_root / "tokenizer"
        prepare_tokenizer(self.recipe_path, second_tokenizer, self.evaluation)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(
                self.recipe_path, second_tokenizer, second_root, self.evaluation
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
            prepare_tokenizer(self.recipe_path, root / "tokenizer", self.evaluation)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_corpus(
                    self.recipe_path,
                    root / "tokenizer",
                    root,
                    self.evaluation,
                    workers,
                )["main"]
            corpora.append(Corpus(root / "corpus"))
            self.assertEqual(
                manifest["sources"]["ja-hq"]["statistics"]["duplicates"], 1
            )
        self.assertEqual(corpora[0].fingerprint, corpora[1].fingerprint)

    def test_並列に作る部分のプロセスが終わってしまうとコーパス作りを止める(self):
        """Stops at once when a part's process dies, instead of waiting for the part."""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        with (
            patch("pretrain.recipe.prepare_part", side_effect=lambda *_: os._exit(1)),
            self.assertRaisesRegex(RuntimeError, "exit code 1"),
        ):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root, self.evaluation, 2
            )

    def test_必要量に満たないとわかった時点で残りを読まずに止めコーパスを残さない(self):
        """部分を小さい順に読み、データ源が割当に届かないと確かになった時点で、残りの部分を読まずに止める。"""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        rows, self.rows["ja-hq"] = self.rows["ja-hq"], []
        with self.assertRaisesRegex(ValueError, "short of the tokenizer sample budget"):
            prepare_tokenizer(
                self.recipe_path, self.root / "second-tokenizer", self.evaluation
            )
        self.rows["ja-hq"] = rows
        self.rows["en-math"] = self.rows["en-math"][:1]
        read = []
        with (
            patch(
                "pretrain.recipe.prepare_part",
                side_effect=lambda *arguments: read.append(arguments[3]["source"])
                or prepare_part(*arguments),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, r"main en-math lacks at least"),
        ):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root, self.evaluation
            )
        # The parts after en-math's, ja-hq's largest of all, are never read.
        self.assertEqual(read, ["ja-wiki", "en-math"])
        self.assertFalse((self.root / "corpus").exists())

    def test_止めたコーパス作りは終わった部分を読み直さずに続きから作る(self):
        """途中で止まったコーパス作りは終わった部分を残し、次は残りの部分だけを読んで、一度に作ったときと同じコーパスを作る。"""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        read = []

        def interrupted(*arguments):
            if len(read) == 2:
                raise RuntimeError("interrupted")
            read.append(arguments[3]["source"])
            return prepare_part(*arguments)

        with (
            patch("pretrain.recipe.prepare_part", side_effect=interrupted),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            prepare_corpus(self.recipe_path, self.root / "tokenizer", self.root, self.evaluation)
        self.assertFalse((self.root / "corpus").exists())
        resumed = []
        with (
            patch(
                "pretrain.recipe.prepare_part",
                side_effect=lambda *arguments: resumed.append(arguments[3]["source"])
                or prepare_part(*arguments),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            prepare_corpus(self.recipe_path, self.root / "tokenizer", self.root, self.evaluation)
        self.assertEqual(read, ["ja-wiki", "en-math"])
        self.assertEqual(resumed, ["en-wiki", "en-web", "ja-hq"])
        self.assertFalse((self.root / ".corpus-parts").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root / "whole", self.evaluation
            )
        self.assertEqual(
            Corpus(self.root / "corpus").fingerprint, Corpus(self.root / "whole/corpus").fingerprint
        )

    def test_レシピが変わったら止めたときの部分を使わずに読み直す(self):
        """途中で止まったあとにレシピが変わったら、残っていた部分は使わず、すべての部分を読み直す。"""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        read = []

        def interrupted(*arguments):
            if read:
                raise RuntimeError("interrupted")
            read.append(arguments[3]["source"])
            return prepare_part(*arguments)

        with (
            patch("pretrain.recipe.prepare_part", side_effect=interrupted),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            prepare_corpus(self.recipe_path, self.root / "tokenizer", self.root, self.evaluation)
        text = self.recipe_path.read_text()
        self.assertIn("minimum_characters = 50", text)
        self.recipe_path.write_text(text.replace("minimum_characters = 50", "minimum_characters = 60"))
        prepare_tokenizer(self.recipe_path, self.root / "second-tokenizer", self.evaluation)
        reread = []
        with (
            patch(
                "pretrain.recipe.prepare_part",
                side_effect=lambda *arguments: reread.append(arguments[3]["source"])
                or prepare_part(*arguments),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            prepare_corpus(
                self.recipe_path, self.root / "second-tokenizer", self.root, self.evaluation
            )
        self.assertEqual(len(reread), 5)

    def test_ほかのデータ源と重なって足りなくなったデータ源をまとめて通知する(self):
        """部分ごとには足りていても、まとめるときにほかのデータ源と重なる文書を除いて足りなくなったデータ源を、すべて通知する。"""
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
        for copy, original in (("ja-wiki", "ja-hq"), ("en-wiki", "en-web")):
            self.rows[copy] = [
                {**row, "text": source["text"]}
                for row, source in zip(self.rows[copy], self.rows[original])
            ]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, r"main ja-wiki lacks .*main en-wiki lacks"),
        ):
            prepare_corpus(
                self.recipe_path, self.root / "tokenizer", self.root, self.evaluation
            )
        self.assertFalse((self.root / "corpus").exists())

    def test_前に作ったコーパスの文書を除いて続きのコーパスを作る(self):
        """前に作ったコーパスが記録した文書を除き、まだ使っていない文書だけで割当を満たすコーパスを作る。"""
        self.prepare()
        documents = self.root / "corpus/documents.jsonl"
        recipe = self.root / "later.toml"
        text = self.recipe_path.read_text()
        recipe.write_text(
            text.replace(
                "[corpus]\n",
                f'[corpus]\nexcluded_corpora = [{{ path = "{self.root / "corpus"}", '
                f'documents_sha256 = "{file_digest(documents)}" }}]\n',
            )
        )
        later = self.root / "later"
        prepare_tokenizer(recipe, later / "tokenizer", self.evaluation)
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = prepare_corpus(recipe, later / "tokenizer", later, self.evaluation)[
                "main"
            ]

        def digests(path):
            return {json.loads(line)["digest"] for line in path.read_text().splitlines()}

        self.assertTrue(digests(documents).isdisjoint(digests(later / "corpus/documents.jsonl")))
        for source in manifest["sources"].values():
            self.assertGreater(source["statistics"]["used_earlier"], 0)
            for split, target in source["targets"].items():
                self.assertGreaterEqual(source["statistics"][f"{split}_tokens"], target)
        recipe.write_text(recipe.read_text().replace(file_digest(documents), "0" * 64))
        prepare_tokenizer(recipe, later / "second-tokenizer", self.evaluation)
        with self.assertRaisesRegex(ValueError, "do not match their checksum"):
            prepare_corpus(recipe, later / "second-tokenizer", later / "second", self.evaluation)
        recipe.write_text(text.replace("[corpus]\n", '[corpus]\nexcluded_corpora = [{ path = "corpus" }]\n'))
        with self.assertRaisesRegex(ValueError, "excluded corpus"):
            read_recipe(recipe)

    def test_大きさの違うファイルに分かれたデータ源からも必要量を集める(self):
        """A part dealt only small files is not asked for what the large one holds."""
        self.recipe_path.write_text(
            self.recipe_path.read_text().replace("[corpus]\n", "[corpus]\npart_tokens = 600\n")
        )
        rows = self.rows["ja-hq"]
        for index, row in enumerate(rows):
            # Short documents, so the budget takes several and none overshoots it.
            row["text"] = "自然科学と歴史を説明する短い文章です。" * 3 + str(index)
        for row in rows[:3]:
            row["text"] = "短い。"
        # Three small files hold only rejected rows; one large file holds the rest.
        self.layouts["ja-hq"] = lambda index: f"small-{index}" if index < 3 else "large"
        corpus = self.prepare()
        stats = corpus.manifest["sources"]["ja-hq"]["statistics"]
        targets = corpus.manifest["sources"]["ja-hq"]["targets"]
        self.assertGreaterEqual(stats["train_tokens"], targets["train"])
        self.assertGreaterEqual(stats["validation_tokens"], targets["validation"])

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

    def tune(self, sft):
        """Build instruction data for the recipe, with sft added to its [sft] table, and write its tuning settings."""
        # The conversation format marks turns with reserved tokens.
        self.recipe_path.write_text(
            self.recipe_path.read_text()
            .replace("vocabulary_size = 258", "vocabulary_size = 266\nreserved_tokens = 8")
            .replace("[sft]\n", "[sft]\n" + sft)
        )
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
            evaluation=self.evaluation,
        )
        result = write_sft_config(
            self.recipe_path,
            self.root / "instructions",
            self.root / "tokenizer",
            self.root / "sft.toml",
        )
        return (result, *read_config(self.root / "sft.toml"))

    def test_指示データの量から同じモデルの追加学習設定を作成する(self):
        result, model, config = self.tune(
            'optimizer = "normuon"\nhyperball_rate = 10.0\nsampling = "sequential"\neval_batches = 3\n'
        )
        self.assertEqual(model, read_config(self.root / "training.toml")[0])
        self.assertEqual(config.learning_rate, 0.00002)
        self.assertEqual(
            (config.optimizer, config.hyperball_rate, config.sampling), ("normuon", 10.0, "sequential")
        )
        self.assertEqual(config.eval_batches, 3)
        # The residual scalars fall by the share that this rate is of pretraining's
        # peak at this batch.
        _, pretraining = read_config(self.root / "training.toml")
        self.assertAlmostEqual(
            config.scalar_rate,
            0.00002
            / (
                pretraining.learning_rate
                * math.sqrt(
                    config.batch_size
                    * config.accumulation_steps
                    / (pretraining.batch_size * pretraining.accumulation_steps)
                )
            ),
        )
        corpus = InstructionCorpus(self.root / "instructions")
        target = (
            2
            * 2
            * 128
            * min(len(corpus.rows["train"][lang][0]) for lang in ("ja", "en"))
        )
        self.assertGreaterEqual(result["scheduled_tokens"], target)
        self.assertEqual(config.steps, max(2, math.ceil(target / (4 * 8 * 128))))

    def test_事前学習が行列の大きさを保つなら指示学習も同じ率で保つ(self):
        training = self.root / "training.toml"
        training.write_text(training.read_text() + 'optimizer="normuon"\nhyperball_rate=7.0\n')
        _, _, config = self.tune('optimizer = "normuon"\n')
        self.assertEqual((config.optimizer, config.hyperball_rate), ("normuon", 7.0))

    def test_リストの中の項目から本文を読む(self):
        rewritten = {"id": "rewritten", "text_field": "rollout_results.0.text"}
        row = {"text": "元の文書", "rollout_results": [{"text": "書き直した文書"}]}
        self.assertEqual(source_text(row, rewritten), "書き直した文書")
        self.assertIn("rollout_results", source_columns(rewritten))
        self.assertNotIn("text", source_columns(rewritten))
        self.assertIsNone(source_text({"rollout_results": []}, rewritten))
        asked = {"id": "asked", "template": "User: {messages.0.content}\n\nAssistant: {messages.1.content}"}
        talk = {"messages": [{"content": "問い"}, {"content": "答え"}]}
        self.assertEqual(source_text(talk, asked), "User: 問い\n\nAssistant: 答え")
        self.assertIn("messages", source_columns(asked))
        self.assertIsNone(source_text({"messages": [{"content": "問い"}]}, asked))

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
                    SimpleNamespace(
                        rfilename=source["prefix"] + "fixture.parquet",
                        size=file.stat().st_size,
                    )
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
        prepare_tokenizer(self.recipe_path, self.root / "tokenizer", self.evaluation)
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

    def test_欄を当てはめた文章や会話を文書として読む(self):
        """型に欄を当てはめた文章や会話の発言の並びを、指示チューニングと同じ書式の文書として読む。"""
        corpus = self.recipe["corpus"]
        answer = "Two and three make five. " * 4
        templated = {"id": "math", "template": "User: {problem}\n\nAssistant: {solution}"}
        self.assertEqual(
            document_text({"problem": "What is 2 + 3?", "solution": answer}, templated, corpus),
            f"User: What is 2 + 3?\n\nAssistant: {answer}".strip(),
        )
        self.assertIsNone(document_text({"problem": "What is 2 + 3?"}, templated, corpus))
        talk = {"id": "talk", "conversation_field": "conversations"}
        turns = [{"from": "human", "value": "Why is the sky blue?"}, {"from": "gpt", "value": answer}]
        self.assertEqual(
            document_text({"conversations": turns}, talk, corpus),
            conversation_text(
                [{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": answer}]
            )[0].strip(),
        )
        unknown = [{"from": "narrator", "value": answer}]
        self.assertIsNone(document_text({"conversations": unknown}, talk, corpus))
        source = {
            "id": "both",
            "language": "en",
            "repository": "org/data",
            "revision": "0" * 40,
            "prefix": "data/",
            "license": "CC-BY-4.0",
            "terms_url": "https://example.org/terms",
        }
        validate_source({**source, "template": "{question} {answer}"})
        with self.assertRaisesRegex(ValueError, "one way"):
            validate_source({**source, "template": "{question}", "conversation_field": "messages"})

    def test_カタログに足したデータ源は既存のデータを変えない(self):
        before = data_fingerprint(self.recipe_path)
        with (self.root / "sources.toml").open("a") as catalog:
            catalog.write(
                '\n[[sources]]\nid = "en-new"\nlanguage = "en"\nrepository = "org/new"\n'
                f'revision = "{"0" * 40}"\nprefix = "data/"\nlicense = "CC0-1.0"\n'
                'terms_url = "https://example.org/terms"\n'
            )
        self.assertEqual(data_fingerprint(self.recipe_path), before)
        with (self.root / "sources.toml").open("a") as catalog:
            catalog.write("minimun = { score = 3 }\n")
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            read_recipe(self.recipe_path)

    def test_記録した語彙を再利用し異なる語彙を拒否する(self):
        recipe = self.phase_recipe()
        prepare_tokenizer(recipe, self.root / "reused", self.evaluation)
        self.assertEqual(
            file_digest(self.root / "reused/tokenizer.json"),
            file_digest(self.root / "tokenizer/tokenizer.json"),
        )
        text = recipe.read_text()
        start = text.index('reuse_sha256 = "') + len('reuse_sha256 = "')
        recipe.write_text(text[:start] + "0" * 64 + text[start + 64 :])
        with self.assertRaisesRegex(ValueError, "does not match its checksum"):
            prepare_tokenizer(recipe, self.root / "rejected", self.evaluation)

    def test_段階ごとのコーパスを重複なく作り段階をつなぐ手順を示す(self):
        recipe = self.phase_recipe('initialize_from = "runs/base/model.pt"\n')
        prepare_tokenizer(recipe, self.root / "phased/tokenizer", self.evaluation)
        with contextlib.redirect_stdout(io.StringIO()):
            manifests = prepare_corpus(
                recipe,
                self.root / "phased/tokenizer",
                self.root / "phased",
                self.evaluation,
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
        trainings = commands(inspect_recipe(recipe, self.root / "phased"), "train")
        self.assertEqual(
            trainings[0][trainings[0].index("--initialize-from") + 1],
            "runs/base/model.pt",
        )
        self.assertEqual(trainings[0][trainings[0].index("--max-steps") + 1], "4")
        self.assertEqual(
            trainings[1][trainings[1].index("--continue-from") + 1],
            "runs/phased-stable/last.pt",
        )
        self.assertNotIn("--max-steps", trainings[1])

    def test_保存したチェックポイントから段階を続け終了位置をその続きから数える(self):
        recipe = self.phase_recipe(
            'continue_from = { checkpoint = "runs/earlier-stable/last.pt", step = 10 }\n'
        )
        config = self.root / "phases.toml"
        config.write_text(config.read_text().replace("\nsteps=6\n", "\nsteps=16\n"))
        inspected = inspect_recipe(recipe, self.root / "phased")
        trainings = commands(inspected, "train")
        self.assertEqual(
            trainings[0][trainings[0].index("--continue-from") + 1],
            "runs/earlier-stable/last.pt",
        )
        self.assertNotIn("--initialize-from", trainings[0])
        self.assertEqual(trainings[0][trainings[0].index("--max-steps") + 1], "14")
        self.assertEqual(
            trainings[1][trainings[1].index("--continue-from") + 1],
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
                        SimpleNamespace(rfilename="part-0." + form, size=len(data)),
                        SimpleNamespace(rfilename="part-1.parquet", size=0),
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

    def test_GitLabの固定した版の大きさを中身を読まずに調べ途中で切れた応答は続きから読む(self):
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

            def do_POST(self):
                # GraphQL answers with the size of the content behind each pointer.
                asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server_test.assertEqual(asked["variables"]["ref"], revision)
                nodes = [
                    {"path": name, "rawSize": str(len(files[name]))}
                    for name in asked["variables"]["paths"]
                ]
                body = json.dumps({"data": {"project": {"repository": {"blobs": {"nodes": nodes}}}}}).encode()
                self.send_response(200 if self.path == "/api/graphql" else 404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
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
            sizes = source_sizes(source)
            loaded = list(iter_source(source, 42))
        self.assertEqual(sizes, {"ja/math/questions.jsonl.gz": len(data)})
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
