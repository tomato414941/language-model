import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None
    for name in ("torch", "tokenizers", "pyarrow", "huggingface_hub")
):
    raise unittest.SkipTest("Install training and corpus dependencies")

import pyarrow as arrow
from pyarrow import parquet

from pretrain.data import Corpus, file_digest, prepare_data
from pretrain.wikipedia import fetch_wikipedia, sample_shard


class WikipediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {}
        self.rows = {}
        for language in ("ja", "en"):
            for shard in range(2):
                rows = [
                    {
                        "text": (
                            "百科事典の文章です。"
                            if language == "ja"
                            else "An encyclopedia article. "
                        )
                        * 40
                        + f" {shard}:{index}",
                        "id": f"{language}wiki/{shard}:{index}",
                        "url": f"https://{language}.wikipedia.org/wiki/Example_{shard}_{index}",
                        "title": f"Example {shard}:{index}",
                        "version": 100 + index,
                        "in_language": language,
                    }
                    for index in range(8)
                ]
                self.rows.update({row["id"]: row for row in rows})
                name = f"data/{language}wiki/000_0000{shard}.parquet"
                path = self.root / f"{language}-{shard}.parquet"
                parquet.write_table(
                    arrow.Table.from_pylist(rows), path, row_group_size=3
                )
                self.paths[name] = path

    def export(self, output):
        info = SimpleNamespace(
            sha="fixture-revision",
            siblings=[SimpleNamespace(rfilename=name) for name in self.paths],
        )

        def open_source(path, mode, **kwargs):
            name = path.split("fixture-revision/", 1)[1]
            return self.paths[name].open(mode)

        with (
            patch("huggingface_hub.HfApi") as api,
            patch("huggingface_hub.HfFileSystem") as filesystem,
        ):
            api.return_value.dataset_info.return_value = info
            filesystem.return_value.open.side_effect = open_source
            return fetch_wikipedia(output, documents_per_language=6, seed=42)

    def test_exports_both_languages_with_article_attribution_and_checksums(self):
        """両言語を各シャードから採取し、記事の出典とチェックサムを保存する。"""
        output = self.root / "source"
        manifest = self.export(output)
        for language in ("ja", "en"):
            path = output / f"{language}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["in_language"] for row in rows}, {language})
            self.assertEqual(
                {row["id"].split("/")[1].split(":")[0] for row in rows}, {"0", "1"}
            )
            for row in rows:
                self.assertEqual(row, self.rows[row["id"]])
            self.assertEqual(manifest["files"][path.name]["sha256"], file_digest(path))
        self.assertEqual(manifest["revision"], "fixture-revision")
        self.assertEqual(manifest["license"], "CC-BY-SA-4.0")

    def test_seed_reproduces_the_same_article_selection(self):
        """同じシードで同じ記事を同じ順序で取得する。"""
        first, second = self.root / "first", self.root / "second"
        self.export(first)
        self.export(second)
        for language in ("ja", "en"):
            self.assertEqual(
                (first / f"{language}.jsonl").read_bytes(),
                (second / f"{language}.jsonl").read_bytes(),
            )

    def test_prepared_tokens_retain_verified_source_provenance(self):
        """入力本文を検証し、学習データに出典とライセンスを引き継ぐ。"""
        source = self.root / "source"
        self.export(source)
        manifest = prepare_data(
            source / "ja.jsonl",
            source / "en.jsonl",
            self.root / "tokens",
            vocabulary_size=258,
            provenance=source / "sources.json",
        )
        self.assertEqual(manifest["provenance"]["revision"], "fixture-revision")
        self.assertEqual(manifest["provenance"]["license"], "CC-BY-SA-4.0")
        for language in ("ja", "en"):
            self.assertEqual(
                manifest["inputs"][language]["sha256"],
                file_digest(source / f"{language}.jsonl"),
            )

    def test_reports_source_checksum_mismatch(self):
        """取得時と内容が異なる入力本文をチェックサム不一致として説明する。"""
        source = self.root / "source"
        self.export(source)
        with (source / "ja.jsonl").open("a") as destination:
            destination.write(json.dumps({"text": "追加した本文"}) + "\n")
        with self.assertRaisesRegex(ValueError, "source checksum"):
            prepare_data(
                source / "ja.jsonl",
                source / "en.jsonl",
                self.root / "tokens",
                vocabulary_size=258,
                provenance=source / "sources.json",
            )

    def test_tokenizer_sample_budget_keeps_full_training_documents(self):
        """指定量の本文で語彙を学習し、モデル学習用には全文を保存する。"""
        source = self.root / "source"
        self.export(source)
        output = self.root / "tokens"
        manifest = prepare_data(
            source / "ja.jsonl",
            source / "en.jsonl",
            output,
            vocabulary_size=258,
            tokenizer_bytes_per_language=64,
        )
        corpus = Corpus(output)
        for language in ("ja", "en"):
            self.assertLessEqual(
                manifest["tokenizer_training"]["samples"][language]["bytes"], 64
            )
            full_text = "\n".join(
                corpus.tokenizer.decode(corpus.streams[split][language].tolist())
                for split in ("train", "validation")
            )
            for line in (source / f"{language}.jsonl").read_text().splitlines():
                self.assertIn(json.loads(line)["text"], full_text)

    def test_samples_eligible_articles_in_the_requested_language(self):
        """指定言語で十分な長さの本文を持つ記事を採取する。"""
        rows = [row for row in self.rows.values() if row["in_language"] == "ja"]
        rows += [
            dict(rows[0], text="短い本文", id="short"),
            dict(rows[0], in_language="en", id="wrong-language"),
        ]
        path = self.root / "mixed.parquet"
        parquet.write_table(arrow.Table.from_pylist(rows), path, row_group_size=3)
        selected = self.root / "selected.jsonl"
        with parquet.ParquetFile(path) as source, selected.open("w") as destination:
            count = sample_shard(source, "ja", 6, random.Random(42), destination)
        self.assertEqual(count, 6)
        for line in selected.read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(row["in_language"], "ja")
            self.assertGreaterEqual(len(row["text"].strip()), 300)


if __name__ == "__main__":
    unittest.main()
