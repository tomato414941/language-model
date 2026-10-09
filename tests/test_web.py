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
from pretrain.web import SOURCES, eligible, fetch_web, sample_web_shard


class WebCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {}
        self.documents = {}
        for language, source in SOURCES.items():
            texts = []
            for shard in range(2):
                rows = [
                    {
                        "text": (
                            "自然科学の解説です。"
                            if language == "ja"
                            else "An educational explanation. "
                        )
                        * 40
                        + f" {shard}:{index}",
                        "id": f"{language}-{shard}-{index}",
                        "url": f"https://example.org/{language}/{shard}/{index}",
                        "language": source["language"],
                        "language_score": 0.99,
                        source["quality_field"]: 0.8 if language == "ja" else 4.0,
                    }
                    for index in range(8)
                ]
                name = source["prefix"] + f"{shard}.parquet"
                path = self.root / f"{language}-{shard}.parquet"
                parquet.write_table(
                    arrow.Table.from_pylist(rows), path, row_group_size=3
                )
                self.paths[(source["repository"], name)] = path
                texts.extend(row["text"] for row in rows)
            path = self.root / f"{language}.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    for text in texts
                )
            )
            self.documents[language] = path
        self.data = self.root / "tokens"
        prepare_data(
            self.documents["ja"], self.documents["en"], self.data, vocabulary_size=258
        )
        self.tokenizer = Corpus(self.data).tokenizer

    def export(self, output, workers=2):
        def dataset_info(repo, revision):
            return SimpleNamespace(
                sha=revision,
                siblings=[
                    SimpleNamespace(rfilename=name)
                    for owner, name in self.paths
                    if owner == repo
                ],
            )

        def open_source(remote, mode, **kwargs):
            repository, path = remote.removeprefix("datasets/").split("@", 1)
            name = path.split("/", 1)[1]
            return self.paths[(repository, name)].open(mode)

        with (
            patch("huggingface_hub.HfApi") as api,
            patch("huggingface_hub.HfFileSystem") as filesystem,
        ):
            api.return_value.dataset_info.side_effect = dataset_info
            filesystem.return_value.open.side_effect = open_source
            return fetch_web(
                output,
                self.data / "tokenizer.json",
                2000,
                shards_per_language=2,
                workers=workers,
            )

    def test_downloads_both_languages_to_the_token_budget_with_provenance(self):
        output = self.root / "web"
        manifest = self.export(output)
        for language, source in SOURCES.items():
            path = output / f"{language}.jsonl"
            metadata = manifest["files"][path.name]
            self.assertGreaterEqual(metadata["web_tokens"], 2000)
            self.assertEqual(metadata["sha256"], file_digest(path))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), metadata["documents"])
            self.assertEqual(
                {row["source_repository"] for row in rows}, {source["repository"]}
            )
            self.assertEqual({row["language"] for row in rows}, {source["language"]})
            self.assertEqual(len(metadata["shards"]), 2)

    def test_parallel_download_reproduces_the_same_seeded_data(self):
        first, second = self.root / "first", self.root / "second"
        self.export(first, workers=1)
        self.export(second, workers=2)
        for language in ("ja", "en"):
            self.assertEqual(
                (first / f"{language}.jsonl").read_bytes(),
                (second / f"{language}.jsonl").read_bytes(),
            )

    def test_sampling_keeps_confident_educational_text_and_filters_repeated_lines(self):
        source = SOURCES["en"]
        row = {
            "text": "A useful explanation. " * 30,
            "language": "en",
            "language_score": 0.99,
            "score": 4.0,
        }
        self.assertTrue(eligible(row, source))
        self.assertFalse(
            eligible({**row, "text": ("A repeated navigation line\n" * 30)}, source)
        )
        self.assertFalse(eligible({**row, "score": 2.0}, source))
        self.assertFalse(eligible({**row, "language_score": 0.5}, source))
        path = next(
            path
            for (owner, _), path in self.paths.items()
            if owner == source["repository"]
        )
        selected = self.root / "selected.jsonl"
        with parquet.ParquetFile(path) as table, selected.open("w") as destination:
            result = sample_web_shard(
                table, self.tokenizer, source, 1000, random.Random(42), destination
            )
        rows = [json.loads(line) for line in selected.read_text().splitlines()]
        self.assertGreaterEqual(result["tokens"], 1000)
        self.assertTrue(all(eligible(item, source) for item in rows))


if __name__ == "__main__":
    unittest.main()
