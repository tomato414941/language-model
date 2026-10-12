"""Deploy the native Responses API on demand using Modal CPUs."""

import os
import re
from pathlib import Path

import modal

ROOT = Path(__file__).parent
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "tomato414941/language-model")
MODEL_RUN = os.environ.get("HF_MODEL_RUN_ID", "bilingual-v2-1b")
MODEL_REVISION = os.environ.get(
    "HF_MODEL_REVISION", "d7aa43a7e0467493ebc6d519d4608a1dcbb0e04a"
)
MODEL_ID = os.environ.get("MODEL_ID", MODEL_RUN)
MODEL_DIRECTORY = "/opt/models"

if not re.fullmatch(r"[a-f0-9]{40}", MODEL_REVISION):
    raise ValueError("Pin HF_MODEL_REVISION to the model's full commit SHA")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", MODEL_RUN):
    raise ValueError("Use a single training run name for HF_MODEL_RUN_ID")


def download_model(repo_id, run_id, revision, directory):
    from pretrain.hub import download_checkpoint

    restored = download_checkpoint(repo_id, run_id, directory, revision=revision)
    if restored["revision"] != revision:
        raise ValueError("The downloaded model does not match the pinned revision")


# Keep the authenticated download in a cached build layer, before application code.
# Running containers only need the API key and the verified model files.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT),
        extras=["train-cpu", "hub"],
        extra_options="--no-dev",
        uv_version="0.10.12",
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
    .add_local_python_source("server", "responses_api", "model", "pretrain")
    .add_local_file(ROOT / "checkpoints/names.json", "/root/checkpoints/names.json")
)

app = modal.App("language-model-api")


@app.function(
    image=image,
    env={
        "MODEL_ID": MODEL_ID,
        "HF_MODEL_REPO": MODEL_REPO,
        "HF_MODEL_RUN_ID": MODEL_RUN,
        "HF_MODEL_REVISION": MODEL_REVISION,
    },
    secrets=[
        modal.Secret.from_name(
            "language-model-api-key", required_keys=["MODEL_API_KEY"]
        )
    ],
    cpu=(1, 1),
    memory=(2048, 3072),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    timeout=1800,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    from responses_api import ModelSettings
    from server import create_app

    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise ValueError("Set MODEL_API_KEY in the language-model-api-key secret")
    return create_app(
        response_settings=ModelSettings(
            model_id=MODEL_ID,
            checkpoint=Path(MODEL_DIRECTORY) / "runs" / MODEL_RUN / "model.pt",
            device="cpu",
            threads=1,
            api_key=api_key,
        )
    )
