import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
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
import torch
import torch.nn.functional as F

from pretrain.data import Corpus, file_digest, prepare_data
from pretrain.model import LanguageModel, ModelConfig
from pretrain.recipe import (
    prepare_corpus,
    prepare_tokenizer,
    read_recipe,
    validate_corpus,
)
from pretrain.selection import document_losses, select_corpora, train_classifier
from pretrain.training import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]
TEXTS = {
    "ja": {
        "rain": "雨が降ると川の水が増え、田畑が潤います。",
        "stone": "石は硬くて重く、山の岩から砕けてできます。",
    },
    "en": {
        "rain": "Rain falls on the hills and the rivers rise. ",
        "stone": "Stones are hard and heavy pieces of old rock. ",
    },
}


def write_documents(path, texts):
    with path.open("w", encoding="utf-8") as destination:
        for text in texts:
            destination.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_文書ごとに最初の記号を除いた予測の損失を平均する(self):
        torch.manual_seed(0)
        model = LanguageModel(32, ModelConfig(width=16, heads=2, layers=1, context=8))
        tokens = np.random.default_rng(0).integers(0, 32, 30)
        starts = [0, 5, 17]
        losses = document_losses(
            model, tokens, starts, torch.device("cpu"), batch_size=2
        )
        totals, counts = np.zeros(3), np.zeros(3)
        with torch.inference_mode():
            for begin in range(0, len(tokens) - 1, 8):
                window = torch.as_tensor(tokens[begin : begin + 9])[None]
                values = F.cross_entropy(
                    model(window[:, :-1])[0], window[0, 1:], reduction="none"
                )
                for offset, value in enumerate(values.tolist()):
                    position = begin + 1 + offset
                    owner = max(i for i, start in enumerate(starts) if start <= position)
                    if position != starts[owner]:
                        totals[owner] += value
                        counts[owner] += 1
        np.testing.assert_allclose(losses, totals / counts, rtol=1e-5)

    def test_同じ記号でも並び方の違いで文書を見分ける分類器を学ぶ(self):
        generator = np.random.default_rng(0)

        def document(pattern):
            return np.array(
                [*generator.integers(4, 30, 8), *pattern * 20], dtype=np.int64
            )

        positives = [document([1, 2, 3]) for _ in range(60)]
        negatives = [document([2, 1, 3]) for _ in range(60)]
        classifier = train_classifier(
            positives[:40] + negatives[:40],
            [True] * 40 + [False] * 40,
            32,
            torch.device("cpu"),
            learning_rate=0.03,
        )
        scores = classifier.score(positives[40:] + negatives[40:], torch.device("cpu"))
        self.assertGreater((scores[:20, None] > scores[None, 20:]).mean(), 0.95)

    def train_models(self):
        """Train a prior on both kinds of text, then tune it on rain only."""
        kinds = {}
        for name, chosen in (("mixed", ("rain", "stone")), ("rain", ("rain",))):
            paths = {}
            for language in TEXTS:
                paths[language] = self.root / f"{name}-{language}.jsonl"
                write_documents(
                    paths[language],
                    [
                        TEXTS[language][kind] * 4 + f" {index}"
                        for index in range(40)
                        for kind in chosen
                    ],
                )
            kinds[name] = paths
        prepare_data(
            kinds["mixed"]["ja"],
            kinds["mixed"]["en"],
            self.root / "mixed",
            vocabulary_size=258,
        )
        tokenizer = self.root / "mixed/tokenizer.json"
        prepare_data(
            kinds["rain"]["ja"],
            kinds["rain"]["en"],
            self.root / "rain",
            vocabulary_size=258,
            tokenizer_path=tokenizer,
        )
        model = ModelConfig(width=24, heads=4, layers=1, context=64)
        config = TrainConfig(
            steps=30,
            batch_size=4,
            accumulation_steps=1,
            learning_rate=0.01,
            min_learning_rate=0.001,
            warmup_steps=1,
            eval_every=30,
            save_every=30,
            eval_batches=1,
            log_every=30,
        )
        device = torch.device("cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.root / "mixed", self.root / "prior", model, config, device)
            train(
                self.root / "rain",
                self.root / "conditional",
                model,
                config,
                device,
                initialize_from=self.root / "prior/last.pt",
            )
        return tokenizer, self.root / "prior/last.pt", self.root / "conditional/last.pt"

    def write_recipe(self, tokenizer, prior, conditional):
        (self.root / "sources.toml").write_text(
            (ROOT / "configs/sources.toml").read_text()
        )
        (self.root / "previous-shards.toml").write_text(
            (ROOT / "configs/previous-shards.toml").read_text()
        )
        shutil.copytree(
            ROOT / "configs/continue-v4-shards", self.root / "continue-v4-shards"
        )
        (self.root / "training.toml").write_text(
            '[model]\narchitecture="llama"\nwidth=24\nheads=4\nkv_heads=2\nlayers=1\nintermediate_size=64\ncontext=64\n'
            "[training]\nsteps=32\nbatch_size=2\naccumulation_steps=1\nwarmup_steps=1\neval_batches=1\n"
        )
        recipe = self.root / "recipe.toml"
        recipe.write_text(
            f"""training_config = "training.toml"
sources = "sources.toml"
seed = 42

[tokenizer]
vocabulary_size = 258
reuse_from = "{tokenizer}"
reuse_sha256 = "{file_digest(tokenizer)}"

[corpus]
validation_tokens_per_language = 200
validation_fraction = 0.1
minimum_characters = 20
maximum_characters = 100000

[evaluation]
framework = "lm-eval"
version = "0.4.13"
base_tasks = ["hellaswag"]
instruction_tasks = ["ifeval"]

[selection]
prior = {{ checkpoint = "{prior}", sha256 = "{file_digest(prior)}" }}
conditional = {{ checkpoint = "{conditional}", sha256 = "{file_digest(conditional)}" }}
scored_share = 40
label_share = 25

[[corpora]]
name = "pool"
purpose = "pool"
training_tokens = 150000
mix = {{ ja-wiki = 50, en-wiki = 50 }}

[[corpora]]
name = "random"
purpose = "alternative"
training_tokens = 4000
select = {{ pool = "pool", method = "random" }}

[[corpora]]
name = "learnable"
purpose = "alternative"
training_tokens = 4000
select = {{ pool = "pool", method = "learnable" }}
"""
        )
        return recipe

    def build_pool(self, recipe_path):
        benchmarks = self.root / "benchmarks"
        benchmarks.mkdir()
        questions = benchmarks / "ifeval.jsonl"
        questions.write_text(json.dumps({"id": "1", "context": "Describe snow."}) + "\n")
        (benchmarks / "sources.json").write_text(
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
        rows = {}
        for source in read_recipe(recipe_path)["sources"]:
            language = source["language"]
            rows[source["id"]] = [
                {
                    "text": TEXTS[language][kind] * 4 + f" {source['id']} {index}",
                    "id": f"{kind}:{index}",
                    "url": f"https://example.org/{source['id']}/{kind}/{index}",
                    "in_language": language,
                    "source_file": source["prefix"] + "fixture.parquet",
                }
                for index in range(1000)
                for kind in ("rain", "stone")
            ]
        with (
            patch(
                "pretrain.recipe.iter_source",
                side_effect=lambda source, seed, part=None: iter(rows[source["id"]]),
            ),
            patch(
                "pretrain.recipe.source_sizes",
                side_effect=lambda source: {source["prefix"] + "fixture.parquet": 1},
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            prepare_tokenizer(recipe_path, self.root / "tokenizer", benchmarks)
            prepare_corpus(recipe_path, self.root / "tokenizer", self.root, benchmarks)

    def rain_share(self, corpus):
        rows = [
            json.loads(line)
            for line in (corpus / "documents.jsonl").read_text().splitlines()
        ]
        trained = [row for row in rows if row["split"] == "train"]
        return sum(row["id"].startswith("rain") for row in trained) / len(trained)

    def test_調整で予測しやすくなる文書を候補の残りから選んでコーパスを作る(self):
        recipe = self.write_recipe(*self.train_models())
        self.build_pool(recipe)
        result = select_corpora(recipe, self.root, torch.device("cpu"))

        validate_corpus(recipe, self.root)
        scored = {
            json.loads(line)["digest"]
            for line in (self.root / "selection-pool/scores.jsonl")
            .read_text()
            .splitlines()
        }
        pool = Corpus(self.root / "corpus-pool")
        for name in ("random", "learnable"):
            corpus = Corpus(self.root / f"corpus-{name}")
            self.assertEqual(corpus.manifest["selection"]["method"], name)
            for language in ("ja", "en"):
                self.assertGreaterEqual(len(corpus.streams["train"][language]), 2000)
                np.testing.assert_array_equal(
                    corpus.streams["validation"][language],
                    pool.streams["validation"][language],
                )
            chosen = {
                json.loads(line)["digest"]
                for line in (self.root / f"corpus-{name}/documents.jsonl")
                .read_text()
                .splitlines()
                if json.loads(line)["split"] == "train"
            }
            self.assertFalse(chosen & scored)
        self.assertGreaterEqual(self.rain_share(self.root / "corpus-learnable"), 0.9)
        self.assertLess(self.rain_share(self.root / "corpus-random"), 0.8)
        ranking = result["pool"]["ranking"]["languages"]
        for language in ("ja", "en"):
            self.assertGreater(ranking[language]["held_out"]["rank_correlation"], 0.5)


if __name__ == "__main__":
    unittest.main()
