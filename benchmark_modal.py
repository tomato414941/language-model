"""Compare native training throughput on short-lived Modal GPUs."""

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import modal

ROOT = Path(__file__).parent
GPUS = ("H100", "A100-80GB", "A100-40GB", "L40S", "L4")
MODES = ("eager", "compiled")
image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch.compile builds Triton's kernel launcher with the system C compiler.
    .apt_install("gcc", "libc6-dev")
    .uv_sync(
        str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.10.12"
    )
    .add_local_python_source("pretrain")
)
app = modal.App("language-model-training-benchmark")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    max_containers=1,
    scaledown_window=2,
    timeout=1800,
    retries=0,
)
def measure(config, vocabulary_size, batches, modes):
    from pretrain.throughput import measure_training

    results = []
    for mode in modes:
        started = perf_counter()
        result = measure_training(config, vocabulary_size, batches, mode == "compiled")
        result["execution_seconds"] = perf_counter() - started
        result["completed_at"] = datetime.now(UTC).isoformat()
        results.append(result)
    return results


@app.local_entrypoint()
def main(
    config: str = "configs/bilingual-v4.toml",
    gpus: str = "H100,A100-80GB,A100-40GB,L40S",
    vocabulary_size: int = 16384,
    batches: str = "4,8,16",
    modes: str = "eager,compiled",
    output: str = "runs/training-throughput-v4.json",
):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Select a new result file to preserve previous measurements")
    selected, chosen = gpus.split(","), modes.split(",")
    if any(gpu not in GPUS for gpu in selected):
        raise ValueError(f"Select GPUs from {','.join(GPUS)}")
    if any(mode not in MODES for mode in chosen):
        raise ValueError(f"Select modes from {','.join(MODES)}")
    sizes = [int(size) for size in batches.split(",")]
    with (ROOT / config).open("rb") as source:
        settings = tomllib.load(source)
    # Each GPU runs in its own container at the same time.
    calls = [
        (
            gpu,
            measure.with_options(gpu=gpu).spawn(
                settings, vocabulary_size, sizes, chosen
            ),
        )
        for gpu in selected
    ]
    results = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    for gpu, call in calls:
        for result in call.get():
            result["modal_gpu"] = gpu
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        report = {"config": settings, "measurements": results}
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": str(destination)}, ensure_ascii=False))
