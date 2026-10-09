import json
import os
import random
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .data import LANGUAGES, file_digest

REPOSITORY = "HuggingFaceFW/finewiki"
REVISION = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"
COLUMNS = ("text", "id", "url", "title", "version", "in_language")


def sample_shard(source, language, limit, generator, destination):
    count = 0
    groups = list(range(source.num_row_groups))
    generator.shuffle(groups)
    for group in groups:
        rows = source.read_row_group(group, columns=list(COLUMNS)).to_pylist()
        generator.shuffle(rows)
        for row in rows:
            text = row["text"]
            if (
                row["in_language"] != language
                or not isinstance(text, str)
                or not 300 <= len(text.strip()) <= 100000
            ):
                continue
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count == limit:
                return count
    return count


def fetch_wikipedia(output, documents_per_language=20000, seed=42, revision=REVISION):
    try:
        from huggingface_hub import HfApi, HfFileSystem
        from pyarrow import parquet
    except ImportError as error:
        raise ValueError("Install corpus dependencies with --extra corpus") from error

    if type(documents_per_language) is not int or documents_per_language < 2:
        raise ValueError("documents_per_language must be an integer of at least two")
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new source directory")
    info = HfApi(token=False).dataset_info(REPOSITORY, revision=revision)
    filesystem = HfFileSystem(token=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest = {
            "repository": REPOSITORY,
            "revision": info.sha,
            "dataset_url": f"https://huggingface.co/datasets/{REPOSITORY}",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "attribution": "Wikipedia contributors; FineWiki processing by Hugging Face",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "documents_per_language": documents_per_language,
            "selection": "Seeded shard and row-group sampling; 300-100000 characters",
            "files": {},
        }
        for language in LANGUAGES:
            shards = sorted(
                item.rfilename
                for item in info.siblings
                if item.rfilename.startswith(f"data/{language}wiki/")
                and item.rfilename.endswith(".parquet")
            )
            if not shards:
                raise ValueError(f"dataset has no {language} Parquet shards")
            generator = random.Random(f"{seed}:{language}")
            generator.shuffle(shards)
            count = 0
            used_shards = []
            path = staging / f"{language}.jsonl"
            with path.open("w", encoding="utf-8") as destination:
                for index, shard in enumerate(shards):
                    quota = max(
                        1,
                        (documents_per_language - count) // (len(shards) - index),
                    )
                    remote = f"datasets/{REPOSITORY}@{info.sha}/{shard}"
                    with (
                        filesystem.open(remote, "rb", block_size=1024 * 1024) as handle,
                        parquet.ParquetFile(handle) as source,
                    ):
                        selected = sample_shard(
                            source, language, quota, generator, destination
                        )
                    count += selected
                    used_shards.append({"path": shard, "documents": selected})
                    print(
                        f"{language}: {count}/{documents_per_language} documents",
                        flush=True,
                    )
                    if count == documents_per_language:
                        break
            if count != documents_per_language:
                raise ValueError(f"{language} has only {count} eligible documents")
            manifest["files"][path.name] = {
                "documents": count,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
                "shards": used_shards,
            }
        (staging / "sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
