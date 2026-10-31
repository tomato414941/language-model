"""Deploy a model in Hugging Face's format behind the native Responses API and its chat page on Modal.

A small CPU container serves the page and the API and checks the API key.
Only then does it ask the GPU container, which has no address of its own,
to sample the reply, so requests without the key never start the GPU. Neither
do requests with the key, once the GPU has used up its time for the day.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent
MODEL_REPO = "google/gemma-4-31B-it"
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MODEL_ID = "gemma-4-31b-it"
WEIGHTS_VOLUME = "language-model-weights"
WEIGHTS_ROOT = "/weights"
MODEL_DIRECTORY = f"{WEIGHTS_ROOT}/{MODEL_REPO}/{MODEL_REVISION}"
# The cheapest GPU that holds the weights as they are released, 62.5 GB at 16 bits.
GPU = "A100-80GB"
# What the weights leave of the GPU's memory holds about 1 MB for each token a reply caches.
CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 2048
# The top_p that Google recommends for this model.
TOP_P = 0.95
GPU_IDLE_SECONDS = 60
# Billing shows the GPU up about 45 seconds past its idle window.
GPU_SECONDS_AFTER_REPLY = 105
GPU_SECONDS_PER_DAY = 3600

weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True, version=2)


def download_model(repo_id, revision, directory, volume):
    import modal
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, revision=revision, local_dir=directory)
    modal.Volume.from_name(volume).commit()


def check_model(directory, context_tokens):
    """Fail unless this code can load the model on this GPU, reply, and hold a full context."""
    import time

    from pretrain.hf_serving import HuggingFaceModel

    started = time.monotonic()
    model = HuggingFaceModel(directory, "cuda", context_tokens)
    print(f"Loaded the model in {time.monotonic() - started:.0f} seconds")
    words = model.tokenizer("word " * context_tokens, add_special_tokens=False)["input_ids"]
    longest = model.tokenizer.decode(words[: context_tokens - 128])
    for text, tokens in (("こんにちは。自己紹介をしてください。", 64), (longest, 4)):
        started = time.monotonic()
        generation = model.start([{"role": "user", "content": text}], tokens, 1.0, TOP_P)
        written = list(generation)
        generation.close()
        print(
            f"Read {generation.input_tokens} tokens and wrote {generation.output_tokens} "
            f"in {time.monotonic() - started:.1f} seconds"
        )
        if not written:
            raise ValueError("The model wrote no reply")


# Downloading comes before application code, so changing the code does not repeat it.
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(
        str(ROOT),
        extras=["train", "hub", "serve-huggingface"],
        extra_options="--no-dev",
        uv_version="0.13.4",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        download_model,
        args=(MODEL_REPO, MODEL_REVISION, MODEL_DIRECTORY, WEIGHTS_VOLUME),
        volumes={WEIGHTS_ROOT: weights},
        cpu=4,
        memory=8192,
        timeout=3600,
    )
    .add_local_python_source("pretrain", copy=True)
    # A model this code cannot use stops the deploy here, and the running version stays up.
    .run_function(
        check_model,
        args=(MODEL_DIRECTORY, CONTEXT_TOKENS),
        gpu=GPU,
        volumes={WEIGHTS_ROOT: weights.with_mount_options(read_only=True)},
        cpu=4,
        memory=16384,
        timeout=1200,
    )
)
# The entrance needs only the API's own dependencies.
entrance_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(ROOT), extra_options="--no-dev", uv_version="0.13.4")
    .add_local_python_source("server", "responses_api", "pretrain")
    .add_local_file(ROOT / "chat.html", "/root/chat.html")
)

app = modal.App(f"{MODEL_ID}-api")


@app.cls(
    image=model_image,
    gpu=GPU,
    cpu=(4, 4),
    volumes={WEIGHTS_ROOT: weights.with_mount_options(read_only=True)},
    # Loading reads all the weights through memory on their way to the GPU.
    memory=(16384, 102400),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=GPU_IDLE_SECONDS,
    startup_timeout=900,
    timeout=600,
)
class Sampler:
    @modal.enter()
    def load(self):
        from pretrain.hf_serving import HuggingFaceModel

        self.model = HuggingFaceModel(MODEL_DIRECTORY, "cuda", CONTEXT_TOKENS)

    @modal.method()
    def sample(self, *arguments):
        from pretrain.remote import relay

        yield from relay(self.model, *arguments)


@app.function(
    image=entrance_image,
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
    import os

    from pretrain.remote import GPUTime, RemoteTextModel
    from responses_api import ModelSettings
    from server import create_app

    api_key = os.environ.get("LANGUAGE_MODEL_API_KEY")
    if not api_key:
        raise ValueError(
            "Set LANGUAGE_MODEL_API_KEY in the language-model-api-key secret"
        )
    gpu_time = GPUTime(
        modal.Dict.from_name(f"{MODEL_ID}-api-gpu-time", create_if_missing=True),
        GPU_SECONDS_PER_DAY,
        GPU_SECONDS_AFTER_REPLY,
    )
    return create_app(
        response_settings=ModelSettings(
            model_id=MODEL_ID, api_key=api_key, max_output_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P
        ),
        backend=RemoteTextModel(Sampler().sample.remote_gen, MODEL_REVISION, gpu_time),
    )
