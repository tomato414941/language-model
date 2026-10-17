import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .data import file_digest

SOURCE_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"
# Collect all splits to exclude benchmark questions from instruction training.
# Evaluation uses the harness's held-out splits in pretrain.evaluation.
TASKS = {
    "hellaswag": {
        "language": "en",
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "license": "MIT",
        "files": {
            "data/validation-00000-of-00001.parquet": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        },
    },
    "arc_easy": {
        "language": "en",
        "repository": "allenai/ai2_arc",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "CC-BY-SA-4.0",
        "files": {
            "ARC-Easy/train-00000-of-00001.parquet": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
            "ARC-Easy/validation-00000-of-00001.parquet": "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",
            "ARC-Easy/test-00000-of-00001.parquet": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",
        },
    },
    "jcommonsenseqa": {
        "language": "ja",
        "repository": "sbintuitions/JCommonsenseQA",
        "revision": "b72d3115609bed00c386701fcb40234a1fd08c7a",
        "license": "CC-BY-SA-4.0",
        "files": {
            "data/train-00000-of-00001.parquet": "8e1851eb5d7f9579432f2af2ff474f654977b077ef2a83da197d81518c6024ba",
            "data/validation-00000-of-00001.parquet": "b72e6c05535a7ccde7b66f8a721c3b0707e67097a7f33bab626195354180994f",
        },
    },
    "ifeval": {
        "language": "en",
        "repository": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "license": "Apache-2.0",
        "files": {
            "ifeval_input_data.jsonl": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        },
    },
}


def clean_hellaswag(text):
    # Remove WikiHow markup such as [header] in the usual HellaSwag way.
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ").strip()


def hellaswag_item(row):
    context = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
    return {
        "id": f"{row['source_id']}/{row['ind']}",
        "context": clean_hellaswag(context),
        "choices": [" " + clean_hellaswag(ending) for ending in row["endings"]],
        "label": int(row["label"]),
    }


def arc_item(row):
    return {
        "id": row["id"],
        "context": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + text for text in row["choices"]["text"]],
        "label": row["choices"]["label"].index(row["answerKey"]),
    }


def jcommonsenseqa_item(row):
    return {
        "id": str(row["q_id"]),
        "context": f"質問：{row['question']}\n回答：",
        "choices": [row[f"choice{index}"] for index in range(5)],
        "label": row["label"],
    }


def ifeval_item(row):
    return {"id": str(row["key"]), "context": row["prompt"]}


CONVERTERS = {
    "hellaswag": hellaswag_item,
    "arc_easy": arc_item,
    "jcommonsenseqa": jcommonsenseqa_item,
    "ifeval": ifeval_item,
}


def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def fetch_benchmarks(output):
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise ValueError("Install corpus dependencies with --extra corpus") from error
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new benchmark directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest = {
            "version": 1,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "tasks": {},
        }
        for name, task in TASKS.items():
            items = []
            for path, expected in task["files"].items():
                data = download(SOURCE_URL.format(path=path, **task))
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"{name} source checksum does not match: {path}")
                rows = (
                    [
                        json.loads(line)
                        for line in data.decode("utf-8").splitlines()
                        if line.strip()
                    ]
                    if path.endswith(".jsonl")
                    else parquet.read_table(io.BytesIO(data)).to_pylist()
                )
                items.extend(CONVERTERS[name](row) for row in rows)
            if len({item["id"] for item in items}) != len(items):
                raise ValueError(f"{name} items need unique identifiers")
            for item in items:
                if not isinstance(item["context"], str) or not item["context"].strip():
                    raise ValueError(f"{name} item {item['id']} is malformed")
                if "choices" in item and (
                    not 0 <= item["label"] < len(item["choices"])
                    or not all(choice.strip() for choice in item["choices"])
                ):
                    raise ValueError(f"{name} item {item['id']} is malformed")
            filename = f"{name}.jsonl"
            with (staging / filename).open("w", encoding="utf-8") as destination:
                for item in items:
                    destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest["tasks"][name] = {
                **task,
                "dataset_url": f"https://huggingface.co/datasets/{task['repository']}",
                "filename": filename,
                "items": len(items),
                "sha256": file_digest(staging / filename),
            }
        (staging / "sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_benchmarks(path, tasks=None, limit=None):
    path = Path(path)
    manifest = json.loads((path / "sources.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported benchmark format")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    benchmarks = {}
    for name in tasks or manifest["tasks"]:
        if name not in manifest["tasks"]:
            raise ValueError(f"unknown benchmark: {name}")
        metadata = manifest["tasks"][name]
        filename = path / metadata["filename"]
        if file_digest(filename) != metadata["sha256"]:
            raise ValueError(f"benchmark checksum does not match: {name}")
        with filename.open(encoding="utf-8") as source:
            items = [json.loads(line) for line in source]
        if limit is not None and limit < len(items):
            # A seeded sample keeps limited runs comparable across checkpoints.
            selected = random.Random(name).sample(range(len(items)), limit)
            items = [items[index] for index in sorted(selected)]
        benchmarks[name] = items
    return benchmarks
