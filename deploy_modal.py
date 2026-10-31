"""Deploy the native Responses API and its chat page on Modal.

A small CPU container serves the page and the API and checks the API key.
Only then does it ask the GPU container, which has no address of its own,
to sample the reply, so requests without the key never start the GPU. Neither
do requests with the key, once the GPU has used up its time for the day.
"""

import os
import re
from pathlib import Path

import modal

ROOT = Path(__file__).parent
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "tomato414941/language-model")
MODEL_RUN = os.environ.get("HF_MODEL_RUN_ID", "v5")
MODEL_REVISION = os.environ.get(
    "HF_MODEL_REVISION", "d7bb35ca44648345cbe1eae607a1c1508b063f46"
)
MODEL_ID = os.environ.get("MODEL_ID", MODEL_RUN)
MODEL_DIRECTORY = "/opt/models"
GPU_IDLE_SECONDS = 60
GPU_SECONDS_PER_DAY = 3600
# The containers import this file again, and need it to name the same model.
ENVIRONMENT = {
    "MODEL_ID": MODEL_ID,
    "HF_MODEL_REPO": MODEL_REPO,
    "HF_MODEL_RUN_ID": MODEL_RUN,
    "HF_MODEL_REVISION": MODEL_REVISION,
}

if not re.fullmatch(r"[a-f0-9]{40}", MODEL_REVISION):
    raise ValueError("Pin HF_MODEL_REVISION to the model's full commit SHA")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", MODEL_RUN):
    raise ValueError("Use a single training run name for HF_MODEL_RUN_ID")


def download_model(repo_id, run_id, revision, directory):
    from pretrain.hub import download_checkpoint

    restored = download_checkpoint(repo_id, run_id, directory, revision=revision)
    if restored["revision"] != revision:
        raise ValueError("The downloaded model does not match the pinned revision")


def check_model(run_id, directory):
    from pretrain.conversation import chat_prompt
    from pretrain.serving import TextModel

    model = TextModel(Path(directory) / "runs" / run_id / "model.pt", "cpu", 1)
    list(model.start(chat_prompt([{"role": "user", "content": "Hello"}]), 4, 1.0, 1.0))


# Keep the authenticated download in a cached build layer, before application code.
# Running containers only need the verified model files.
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT),
        extras=["train", "hub"],
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
    .add_local_python_source("pretrain", copy=True)
    # A model this code cannot use stops the deploy here, and the running version stays up.
    .run_function(
        check_model,
        args=(MODEL_RUN, MODEL_DIRECTORY),
        cpu=1,
        memory=6144,
        timeout=600,
    )
)
# The entrance needs only the API's own dependencies.
entrance_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(ROOT), extra_options="--no-dev", uv_version="0.13.4")
    .add_local_python_source("server", "responses_api", "pretrain")
    .add_local_file(ROOT / "chat.html", "/root/chat.html")
)

app = modal.App("language-model-api")


@app.cls(
    image=model_image,
    env=ENVIRONMENT,
    gpu="T4",
    cpu=(1, 1),
    # Loading reads the whole checkpoint, optimizer state included, beside the model.
    memory=(2048, 6144),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=GPU_IDLE_SECONDS,
    timeout=1800,
)
class Sampler:
    @modal.enter()
    def load(self):
        from pretrain.serving import TextModel

        self.model = TextModel(
            Path(MODEL_DIRECTORY) / "runs" / MODEL_RUN / "model.pt", "cuda", 1
        )

    @modal.method()
    def sample(self, *arguments):
        from pretrain.remote import relay

        yield from relay(self.model, *arguments)


@app.function(
    image=entrance_image,
    env=ENVIRONMENT,
    secrets=[
        modal.Secret.from_name(
            "language-model-api-key", required_keys=["LANGUAGE_MODEL_API_KEY"]
        )
    ],
    cpu=0.125,
    memory=512,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    timeout=1800,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    from pretrain.remote import GPUTime, RemoteTextModel
    from responses_api import ModelSettings
    from server import create_app

    api_key = os.environ.get("LANGUAGE_MODEL_API_KEY")
    if not api_key:
        raise ValueError(
            "Set LANGUAGE_MODEL_API_KEY in the language-model-api-key secret"
        )
    gpu_time = GPUTime(
        modal.Dict.from_name("language-model-api-gpu-time", create_if_missing=True),
        GPU_SECONDS_PER_DAY,
        GPU_IDLE_SECONDS,
    )
    return create_app(
        response_settings=ModelSettings(model_id=MODEL_ID, api_key=api_key),
        backend=RemoteTextModel(Sampler().sample.remote_gen, MODEL_REVISION, gpu_time),
    )
