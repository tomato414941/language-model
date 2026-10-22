"""Measure local checkpoints on every validation window, on one Modal GPU.

Name the checkpoints and pretraining corpora as comma-separated paths under
the project in VALIDATION_CHECKPOINTS and VALIDATION_DATASETS. Only each
corpus's manifest, tokenizer and validation streams are sent.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import modal

ROOT = Path(__file__).parent
CHECKPOINTS = [p for p in os.environ.get("VALIDATION_CHECKPOINTS", "").split(",") if p]
DATASETS = [p for p in os.environ.get("VALIDATION_DATASETS", "").split(",") if p]
REMOTE = Path("/inputs")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.13.4"
    )
    .add_local_python_source("pretrain")
)
for path in CHECKPOINTS:
    image = image.add_local_file(ROOT / path, str(REMOTE / path))
for directory in DATASETS:
    for name in (
        "manifest.json",
        "tokenizer.json",
        "validation_ja.bin",
        "validation_en.bin",
    ):
        image = image.add_local_file(
            ROOT / directory / name, str(REMOTE / directory / name)
        )
app = modal.App("language-model-validation")


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=16384,
    max_containers=1,
    scaledown_window=2,
    timeout=3600,
    retries=0,
)
def measure(checkpoints, datasets, limit):
    import torch

    from pretrain.data import Corpus, file_digest
    from pretrain.training import evaluate_windows, load_model, select_device

    started = perf_counter()
    device = select_device("cuda")
    corpora = {
        directory: Corpus(REMOTE / directory, ("validation",)) for directory in datasets
    }
    results = []
    for checkpoint in checkpoints:
        model, tokenizer, payload = load_model(REMOTE / checkpoint, device)
        sha256 = file_digest(REMOTE / checkpoint)
        for directory, corpus in corpora.items():
            if corpus.tokenizer.to_str() != tokenizer.to_str():
                raise ValueError(
                    f"{directory} does not use the tokenizer of {checkpoint}"
                )
            results.append(
                {
                    "checkpoint": checkpoint,
                    "sha256": sha256,
                    "step": payload["step"],
                    "tokens_seen": payload["tokens_seen"],
                    "data": directory,
                    "fingerprint": corpus.fingerprint,
                    **evaluate_windows(model, corpus, device, "bfloat16", 16, limit),
                }
            )
        del model
        torch.cuda.empty_cache()
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "precision": "bfloat16",
        "windows_per_language_limit": limit,
        "results": results,
        "elapsed_seconds": perf_counter() - started,
        "completed_at": datetime.now(UTC).isoformat(),
    }


@app.local_entrypoint()
def main(output: str, max_windows: int = 1000):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Select a new result file so previous measurements are kept")
    if not CHECKPOINTS or not DATASETS:
        raise ValueError(
            "Name the checkpoints and corpora in VALIDATION_CHECKPOINTS and VALIDATION_DATASETS"
        )
    result = measure.remote(CHECKPOINTS, DATASETS, max_windows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in result["results"]:
        print(
            json.dumps(
                {
                    "checkpoint": row["checkpoint"],
                    "data": row["data"],
                    "bits_per_byte": round(row["bits_per_byte"], 4),
                    "ja_bits_per_byte": round(row["ja"]["bits_per_byte"], 4),
                    "en_bits_per_byte": round(row["en"]["bits_per_byte"], 4),
                    "loss": round(row["loss"], 4),
                    "ja": round(row["ja"]["loss"], 4),
                    "en": round(row["en"]["loss"], 4),
                    "tokens": row["ja"]["evaluated_tokens"]
                    + row["en"]["evaluated_tokens"],
                }
            )
        )
