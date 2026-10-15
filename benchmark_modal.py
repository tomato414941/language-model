"""Compare native training throughput on short-lived Modal GPUs."""

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import modal

ROOT = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.10.12"
    )
    .add_local_python_source("pretrain")
)
app = modal.App("language-model-training-benchmark")


@app.function(
    image=image,
    gpu="H100",
    cpu=2,
    memory=16384,
    max_containers=1,
    scaledown_window=2,
    timeout=600,
    retries=0,
)
def measure(config, vocabulary_size):
    from pretrain.throughput import measure_training

    started = perf_counter()
    result = measure_training(config, vocabulary_size)
    result["execution_seconds"] = perf_counter() - started
    result["completed_at"] = datetime.now(UTC).isoformat()
    return result


@app.local_entrypoint()
def main(
    config: str = "configs/bilingual-1b.toml",
    gpus: str = "H100,L40S",
    vocabulary_size: int = 32768,
    output: str = "runs/training-throughput-1b.json",
):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Select a new result file to preserve previous measurements")
    selected = gpus.split(",")
    if not selected or any(gpu not in {"H100", "L40S"} for gpu in selected):
        raise ValueError("Select H100, L40S, or H100,L40S")
    with (ROOT / config).open("rb") as source:
        settings = tomllib.load(source)
    results = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    for gpu in selected:
        result = measure.with_options(gpu=gpu).remote(settings, vocabulary_size)
        result["modal_gpu"] = gpu
        results.append(result)
        report = {"config": settings, "measurements": results}
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"result": str(destination)}, ensure_ascii=False))
