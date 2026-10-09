import json
import math
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tokenizers import Tokenizer

from .data import LANGUAGES, file_digest

SOURCES = {
    "ja": {
        "repository": "epfml/FineWeb2-HQ",
        "revision": "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        "prefix": "jpn_Jpan/",
        "language": "jpn",
        "quality_field": "quality_score",
        "selection": "Top 10% quality subset of FineWeb2",
    },
    "en": {
        "repository": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "prefix": "sample/10BT/",
        "language": "en",
        "quality_field": "score",
        "selection": "Seeded sample of FineWeb-Edu, educational score >= 3",
    },
}


def eligible(row, source):
    text = row.get("text")
    if not isinstance(text, str) or not 300 <= len(text.strip()) <= 100000:
        return False
    if row.get("language") != source["language"]:
        return False
    score = row.get("language_score")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0.8:
        return False
    quality = row.get(source["quality_field"])
    if not isinstance(quality, (int, float)) or not math.isfinite(quality):
        return False
    if source["language"] == "en" and quality < 3:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) < 5 or len(set(lines)) / len(lines) >= 0.7


def sample_web_shard(parquet, tokenizer, source, limit, generator, destination):
    columns = [
        "text",
        "id",
        "url",
        "language",
        "language_score",
        source["quality_field"],
    ]
    groups = list(range(parquet.num_row_groups))
    generator.shuffle(groups)
    count, tokens, rejected = 0, 0, 0
    for group in groups:
        rows = parquet.read_row_group(group, columns=columns).to_pylist()
        generator.shuffle(rows)
        accepted = []
        for row in rows:
            if eligible(row, source):
                accepted.append(row)
            else:
                rejected += 1
        for start in range(0, len(accepted), 64):
            batch = accepted[start : start + 64]
            encodings = tokenizer.encode_batch(
                [row["text"] for row in batch], add_special_tokens=False
            )
            for row, encoding in zip(batch, encodings):
                row["source_repository"] = source["repository"]
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                tokens += len(encoding.ids) + 2
                if tokens >= limit:
                    return {"documents": count, "tokens": tokens, "rejected": rejected}
    return {"documents": count, "tokens": tokens, "rejected": rejected}


def fetch_web(
    output,
    tokenizer_path,
    tokens_per_language=500000000,
    wiki_source=None,
    shards_per_language=64,
    workers=4,
    seed=42,
):
    try:
        from huggingface_hub import HfApi, HfFileSystem
        from pyarrow import parquet
    except ImportError as error:
        raise ValueError("Install corpus dependencies with --extra corpus") from error
    for name, value in (
        ("tokens_per_language", tokens_per_language),
        ("shards_per_language", shards_per_language),
        ("workers", workers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new source directory")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    wiki_manifest = None
    if wiki_source is not None:
        wiki_source = Path(wiki_source)
        wiki_manifest = json.loads((wiki_source / "sources.json").read_text())
        for language in LANGUAGES:
            name = f"{language}.jsonl"
            if (
                file_digest(wiki_source / name)
                != wiki_manifest["files"][name]["sha256"]
            ):
                raise ValueError(
                    "Wikipedia source checksum does not match its provenance"
                )
    jobs, resolved, unused = [], {}, {}
    api = HfApi(token=False)
    for language, source in SOURCES.items():
        info = api.dataset_info(source["repository"], revision=source["revision"])
        shards = sorted(
            item.rfilename
            for item in info.siblings
            if item.rfilename.startswith(source["prefix"])
            and item.rfilename.endswith(".parquet")
        )
        random.Random(f"{seed}:{language}").shuffle(shards)
        selected_count = min(shards_per_language, tokens_per_language, len(shards))
        unused[language] = shards[selected_count:]
        shards = shards[:selected_count]
        if not shards:
            raise ValueError(f"dataset has no {language} Parquet shards")
        resolved[language] = {
            **source,
            "revision": info.sha,
            "license": "ODC-By-1.0",
            "license_url": "https://opendatacommons.org/licenses/by/1-0/",
            "terms_url": "https://commoncrawl.org/terms-of-use",
            "dataset_url": f"https://huggingface.co/datasets/{source['repository']}",
        }
        quota, remainder = divmod(tokens_per_language, len(shards))
        for index, shard in enumerate(shards):
            jobs.append((language, shard, quota + (index < remainder)))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    chunks = staging / "chunks"
    chunks.mkdir()
    filesystem = HfFileSystem(token=False)

    def download(index, job):
        language, shard, limit = job
        source = resolved[language]
        remote = f"datasets/{source['repository']}@{source['revision']}/{shard}"
        path = chunks / f"{index}.jsonl"
        for attempt in range(3):
            try:
                with (
                    filesystem.open(remote, "rb", block_size=1024 * 1024) as handle,
                    parquet.ParquetFile(handle) as table,
                    path.open("w", encoding="utf-8") as destination,
                ):
                    result = sample_web_shard(
                        table,
                        tokenizer,
                        source,
                        limit,
                        random.Random(f"{seed}:{language}:{shard}"),
                        destination,
                    )
                print(
                    json.dumps({"language": language, "shard": shard, **result}),
                    flush=True,
                )
                return {"path": shard, **result}
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    try:
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download, index, job): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        for language in LANGUAGES:
            remaining = tokens_per_language - sum(
                results[index]["tokens"]
                for index, job in enumerate(jobs)
                if job[0] == language
            )
            for shard in unused[language]:
                if remaining <= 0:
                    break
                index = len(jobs)
                job = (language, shard, remaining)
                jobs.append(job)
                results[index] = download(index, job)
                remaining -= results[index]["tokens"]
            if remaining > 0:
                raise ValueError(
                    f"{language} sources are short of the requested budget by {remaining} tokens"
                )
        manifest = {
            "version": 1,
            "sources": resolved,
            "wikipedia": wiki_manifest,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "tokenizer_sha256": file_digest(tokenizer_path),
            "web_tokens_per_language": tokens_per_language,
            "selection": "Seeded shard/row-group sampling; language confidence >= 0.8; 300-100000 characters; repeated-line filtering",
            "files": {},
        }
        for language in LANGUAGES:
            path = staging / f"{language}.jsonl"
            selected = [
                results[index] for index, job in enumerate(jobs) if job[0] == language
            ]
            count = sum(item["documents"] for item in selected)
            with path.open("wb") as destination:
                if wiki_source is not None:
                    with (wiki_source / f"{language}.jsonl").open("rb") as source:
                        shutil.copyfileobj(source, destination)
                    count += wiki_manifest["files"][path.name]["documents"]
                for index, job in enumerate(jobs):
                    if job[0] == language:
                        with (chunks / f"{index}.jsonl").open("rb") as source:
                            shutil.copyfileobj(source, destination)
                        (chunks / f"{index}.jsonl").unlink()
            manifest["files"][path.name] = {
                "documents": count,
                "web_tokens": sum(item["tokens"] for item in selected),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
                "shards": selected,
            }
        chunks.rmdir()
        (staging / "sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
