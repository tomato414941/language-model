import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

from .data import BOS, file_digest
from .training import autocast

SOURCE_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"
# Zero-shot evaluation never trains on these items, so every labeled split counts.
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


CONVERTERS = {
    "hellaswag": hellaswag_item,
    "arc_easy": arc_item,
    "jcommonsenseqa": jcommonsenseqa_item,
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
                rows = parquet.read_table(io.BytesIO(data)).to_pylist()
                items.extend(CONVERTERS[name](row) for row in rows)
            if len({item["id"] for item in items}) != len(items):
                raise ValueError(f"{name} items need unique identifiers")
            for item in items:
                if not 0 <= item["label"] < len(item["choices"]) or not all(
                    choice.strip() for choice in item["choices"]
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


@torch.inference_mode()
def choice_log_likelihoods(
    model, tokenizer, items, device, precision="float32", batch_tokens=4096
):
    """Return the log-likelihood of every choice following its context."""
    if type(batch_tokens) is not int or batch_tokens <= 0:
        raise ValueError("batch_tokens must be a positive integer")
    bos = tokenizer.token_to_id(BOS)
    requests = []
    for item_index, item in enumerate(items):
        context = [
            bos,
            *tokenizer.encode(item["context"], add_special_tokens=False).ids,
        ]
        for choice_index, choice in enumerate(item["choices"]):
            continuation = tokenizer.encode(choice, add_special_tokens=False).ids
            if not 0 < len(continuation) <= model.config.context:
                raise ValueError("each choice must fit within the context length")
            tokens = (context + continuation)[-(model.config.context + 1) :]
            requests.append((item_index, choice_index, tokens, len(continuation)))
    scores = [[0.0] * len(item["choices"]) for item in items]
    # Similar lengths share a batch; right padding never reaches earlier tokens.
    order = sorted(range(len(requests)), key=lambda index: -len(requests[index][2]))
    was_training = model.training
    model.eval()
    try:
        start = 0
        while start < len(order):
            width = len(requests[order[start]][2]) - 1
            batch = [
                requests[index]
                for index in order[start : start + max(1, batch_tokens // width)]
            ]
            start += len(batch)
            inputs = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (_, _, tokens, _) in enumerate(batch):
                inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
            with autocast(device, precision):
                logits = model(inputs.to(device))
            log_probabilities = logits.float().log_softmax(-1)
            for row, (item_index, choice_index, tokens, length) in enumerate(batch):
                end = len(tokens) - 1
                targets = torch.tensor(tokens[-length:], device=device)
                scores[item_index][choice_index] = (
                    log_probabilities[row, end - length : end]
                    .gather(-1, targets[:, None])
                    .sum()
                    .item()
                )
    finally:
        model.train(was_training)
    return scores


def evaluate_benchmarks(
    model, tokenizer, benchmarks, device, precision="float32", batch_tokens=4096
):
    results = {}
    for name, items in benchmarks.items():
        if not items:
            raise ValueError(f"{name} has no items")
        scores = choice_log_likelihoods(
            model, tokenizer, items, device, precision, batch_tokens
        )
        correct = normalized = 0
        for item, values in zip(items, scores):
            # Per-character scores keep longer answers from losing by length alone.
            per_character = [
                value / len(choice) for value, choice in zip(values, item["choices"])
            ]
            correct += max(range(len(values)), key=values.__getitem__) == item["label"]
            normalized += (
                max(range(len(values)), key=per_character.__getitem__) == item["label"]
            )
        count = len(items)
        accuracy, normalized_accuracy = correct / count, normalized / count
        results[name] = {
            "items": count,
            "chance": sum(1 / len(item["choices"]) for item in items) / count,
            "normalized_accuracy": normalized_accuracy,
            "normalized_accuracy_stderr": math.sqrt(
                normalized_accuracy * (1 - normalized_accuracy) / count
            ),
            "accuracy": accuracy,
            "accuracy_stderr": math.sqrt(accuracy * (1 - accuracy) / count),
        }
    return results
