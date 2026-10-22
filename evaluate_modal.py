"""Run a pinned checkpoint's standard benchmarks on one temporary Modal GPU."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import modal

ROOT = Path(__file__).parent
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "tomato414941/language-model")
MODEL_RUN = os.environ.get("HF_MODEL_RUN_ID", "instructions-v2")
MODEL_REVISION = os.environ.get(
    "HF_MODEL_REVISION", "1b6a794d17085b380d51ff51380d24689937b645"
)
MODEL_DIRECTORY = "/opt/models"


def download_model(repo_id, run_id, revision, directory):
    from pretrain.hub import download_checkpoint

    restored = download_checkpoint(repo_id, run_id, directory, revision=revision)
    if restored["revision"] != revision:
        raise ValueError("The downloaded model does not match the pinned revision")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT),
        extras=["train", "hub", "eval"],
        extra_options="--no-dev",
        uv_version="0.13.4",
    )
    .add_local_file(
        ROOT / "pretrain/__init__.py", "/root/pretrain/__init__.py", copy=True
    )
    .add_local_file(ROOT / "pretrain/hub.py", "/root/pretrain/hub.py", copy=True)
    .run_function(
        download_model,
        args=(MODEL_REPO, MODEL_RUN, MODEL_REVISION, MODEL_DIRECTORY),
        secrets=[
            modal.Secret.from_name(
                "language-model-hugging-face", required_keys=["HF_TOKEN"]
            )
        ],
        cpu=1,
        memory=2048,
        timeout=600,
    )
    .add_local_python_source("pretrain")
)

app = modal.App("language-model-evaluation")


@app.function(
    image=image,
    gpu="L4",
    cpu=(1, 1),
    memory=(8192, 8192),
    max_containers=1,
    scaledown_window=2,
    timeout=3600,
    retries=0,
    env={"TOKENIZERS_PARALLELISM": "false"},
)
def evaluate():
    import torch

    from pretrain.data import file_digest
    from pretrain.evaluation import evaluate_benchmarks
    from pretrain.training import load_model, select_device

    started = perf_counter()
    torch.set_num_threads(2)
    device = select_device("cuda")
    checkpoint = Path(MODEL_DIRECTORY) / "runs" / MODEL_RUN / "model.pt"
    model, tokenizer, saved = load_model(checkpoint, device)
    print(
        json.dumps(
            {
                "event": "evaluation_started",
                "run": MODEL_RUN,
                "step": saved["step"],
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    result = evaluate_benchmarks(
        model, tokenizer, device, precision="bfloat16", batch_tokens=16384
    )
    result["checkpoint"] = {
        "repository": MODEL_REPO,
        "run": MODEL_RUN,
        "revision": MODEL_REVISION,
        "sha256": file_digest(checkpoint),
        "step": saved["step"],
        "tokens_seen": saved["tokens_seen"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    result["execution"] = {
        "provider": "Modal",
        "gpu": torch.cuda.get_device_name(),
        "precision": "bfloat16",
        "torch_version": str(torch.__version__),
        "evaluation_source_sha256": file_digest(ROOT / "pretrain/evaluation.py"),
        "elapsed_seconds": perf_counter() - started,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    return result


@app.local_entrypoint()
def main(output: str = "runs/standard-evaluation-instructions-v2.json"):
    destination = Path(output)
    if destination.exists():
        raise ValueError(
            "Select a new result file so previous evaluations are preserved"
        )
    result = evaluate.remote()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": str(destination),
                "checkpoint": result["checkpoint"],
                "scores": result["results"],
                "elapsed_seconds": result["execution"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
