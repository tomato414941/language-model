"""Train short runs from scratch on Modal GPUs and measure each on validation corpora.

TRAINING_PLAN names a TOML plan: the corpus to train on, the corpora every final
checkpoint is measured on, the model and training settings, and the runs, each
a name, changes to the training settings and the minutes it may train. Only
the manifests, tokenizers and validation streams go into the image. The
training streams are downloaded inside each container from the entries of the
JSON file named by TRAINING_DOWNLOADS, which holds signed URLs. Its contents
reach the containers only as a secret under another name, so no container
mistakes them for a path, and nothing prints them.
"""

import json
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter

import modal

ROOT = Path(__file__).parent
REMOTE = Path("/inputs")
SMALL_FILES = (
    "manifest.json",
    "tokenizer.json",
    "validation_ja.bin",
    "validation_en.bin",
)
PLAN_PATH = os.environ.get("TRAINING_PLAN") if modal.is_local() else None
PLAN = tomllib.loads((ROOT / PLAN_PATH).read_text()) if PLAN_PATH else None
DOWNLOADS = os.environ.get("TRAINING_DOWNLOADS") if modal.is_local() else None
SECRET_NAME = "TRAINING_DOWNLOAD_ENTRIES"
# Containers never print their own time limit; the plan sets it for every run.
TIMEOUT = 60 * (max(run["minutes"] for run in PLAN["runs"]) + 20) if PLAN else 3600

image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch.compile builds Triton's kernel launcher with the system C compiler.
    .apt_install("gcc", "libc6-dev")
    .uv_sync(
        str(ROOT), extras=["train"], extra_options="--no-dev", uv_version="0.13.4"
    )
    .add_local_python_source("pretrain")
    .add_local_file(ROOT / "scripts/cloud_files.py", "/root/scripts/cloud_files.py")
)
if PLAN:
    for directory in dict.fromkeys([PLAN["data"], *PLAN["validation"]]):
        for name in SMALL_FILES:
            image = image.add_local_file(
                ROOT / directory / name, str(REMOTE / directory / name)
            )
# Containers define the same objects as the local run; they receive the
# secret's value from Modal, not from this definition.
secret = modal.Secret.from_dict(
    {SECRET_NAME: Path(DOWNLOADS).read_text() if DOWNLOADS else ""}
)
app = modal.App("language-model-training-runs")


def without_urls(text):
    # A path normalizes the scheme's "//" to "/", so match either.
    text = re.sub(r"https?:/+\S+", "<url>", text)
    return re.sub(r"X-Amz-[A-Za-z-]+=[^&\s\"']+", "X-Amz-<redacted>", text)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    secrets=[secret],
    max_containers=8,
    scaledown_window=2,
    timeout=TIMEOUT,
    retries=0,
)
def train_run(spec):
    import shutil

    import torch

    from pretrain.data import Corpus
    from pretrain.model import ModelConfig
    from pretrain.training import (
        TrainConfig,
        evaluate_windows,
        load_model,
        select_device,
        train,
    )
    from scripts.cloud_files import transfer_file

    started = perf_counter()
    torch.set_num_threads(4)
    work = Path("/root/work")
    data = work / spec["data"]
    data.mkdir(parents=True)
    for name in SMALL_FILES:
        shutil.copyfile(REMOTE / spec["data"] / name, data / name)
    for entry in json.loads(os.environ[SECRET_NAME])["downloads"]:
        if Path(entry["path"]).parent == Path(spec["data"]):
            transfer_file("download", entry, work)
    downloaded = perf_counter() - started
    model_config = ModelConfig(**spec["model"])
    config = TrainConfig(**spec["training"])
    deadline = monotonic() + 60 * spec["minutes"]
    output = work / "run"
    device = select_device("cuda")
    result = train(
        data,
        output,
        model_config,
        config,
        device,
        compiled=True,
        stop_requested=lambda: monotonic() > deadline,
    )
    model, tokenizer, payload = load_model(output / "last.pt", device)
    measured = []
    for directory in spec["validation"]:
        corpus = Corpus(REMOTE / directory, ("validation",))
        if corpus.tokenizer.to_str() != tokenizer.to_str():
            raise ValueError(f"{directory} does not use the run's tokenizer")
        measured.append(
            {
                "data": directory,
                "fingerprint": corpus.fingerprint,
                **evaluate_windows(
                    model, corpus, device, "bfloat16", 16, spec["windows_per_language"]
                ),
            }
        )
    values = [
        row[language]["bits_per_byte"] for row in measured for language in ("ja", "en")
    ]
    return {
        "name": spec["name"],
        "training": spec["training"],
        "step": result["step"],
        "tokens_seen": result["tokens_seen"],
        "finished": result["step"] == config.steps,
        "bits_per_byte": sum(values) / len(values),
        "validation": measured,
        "metrics": [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ],
        "download_seconds": downloaded,
        "elapsed_seconds": perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "completed_at": datetime.now(UTC).isoformat(),
    }


@app.local_entrypoint()
def main(output: str, runs: str = ""):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Select a new result file so earlier results are kept")
    if not PLAN or not DOWNLOADS:
        raise ValueError(
            "Name the plan in TRAINING_PLAN and the downloads in TRAINING_DOWNLOADS"
        )
    chosen = [name for name in runs.split(",") if name]
    known = [run["name"] for run in PLAN["runs"]]
    if any(name not in known for name in chosen):
        raise ValueError(f"Select runs from {','.join(known)}")
    specs = []
    for run in PLAN["runs"]:
        if chosen and run["name"] not in chosen:
            continue
        changes = {
            key: value for key, value in run.items() if key not in ("name", "minutes")
        }
        specs.append(
            {
                "name": run["name"],
                "minutes": run["minutes"],
                "data": PLAN["data"],
                "validation": PLAN["validation"],
                "windows_per_language": PLAN["windows_per_language"],
                "model": PLAN["model"],
                "training": {**PLAN["training"], **changes},
            }
        )
    results = []
    # Reading every output before pairing them lets the map finish cleanly.
    outputs = list(train_run.map(specs, return_exceptions=True))
    for spec, result in zip(specs, outputs, strict=True):
        if isinstance(result, Exception):
            result = {
                "name": spec["name"],
                "error": type(result).__name__,
                "detail": without_urls(str(result))[:500],
            }
        results.append(result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "name",
                        "error",
                        "finished",
                        "step",
                        "bits_per_byte",
                        "elapsed_seconds",
                    )
                }
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"plan": PLAN_PATH, "results": results}, ensure_ascii=False, indent=2
        )
        + "\n"
    )
