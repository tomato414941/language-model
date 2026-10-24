import json
import math
import os
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .data import BOS, LANGUAGES, file_digest, open_corpus, to_device, token_bytes
from .model import LanguageModel, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10000
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    eval_every: int = 250
    save_every: int = 250
    eval_batches: int = 16
    log_every: int = 10
    seed: int = 42
    # "cosine" decays over the whole run; "wsd" warms up, holds the peak rate,
    # and decays linearly over the last decay_steps so a run can be extended
    # from any checkpoint taken before the decay.
    schedule: str = "cosine"
    decay_steps: int = 0
    # The shape of the wsd decay: "linear", or "1-sqrt", which falls faster at
    # first and spends more of the decay near the final rate.
    decay_shape: str = "linear"
    # "random" draws windows with replacement; "sequential" reads every window
    # once per epoch in a seeded order.
    sampling: str = "random"
    # "adamw" updates every weight with AdamW; "muon" updates the matrices in
    # the blocks with Muon and the embedding and norms with AdamW.
    optimizer: str = "adamw"

    def __post_init__(self):
        for field in (
            "steps",
            "batch_size",
            "accumulation_steps",
            "eval_every",
            "save_every",
            "eval_batches",
            "log_every",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            type(self.warmup_steps) is not int
            or not 0 <= self.warmup_steps < self.steps
        ):
            raise ValueError(
                "warmup_steps must be a nonnegative integer smaller than steps"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        for field in (
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "gradient_clip",
        ):
            value = getattr(self, field)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.schedule not in ("cosine", "wsd"):
            raise ValueError("schedule must be cosine or wsd")
        if self.sampling not in ("random", "sequential"):
            raise ValueError("sampling must be random or sequential")
        if type(self.decay_steps) is not int or not (
            0 <= self.decay_steps <= self.steps - self.warmup_steps
        ):
            raise ValueError("decay_steps must fit after the warmup")
        if self.schedule == "wsd" and self.decay_steps == 0:
            raise ValueError("a wsd schedule needs decay_steps")
        if self.decay_shape not in ("linear", "1-sqrt"):
            raise ValueError("decay_shape must be linear or 1-sqrt")
        if self.schedule != "wsd" and self.decay_shape != "linear":
            raise ValueError("decay_shape applies to a wsd schedule")
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError("optimizer must be adamw or muon")


def read_config(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"model", "training"}:
        raise ValueError("configuration requires model and training sections")
    return ModelConfig(**config["model"]), TrainConfig(**config["training"])


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA GPU is unavailable; check the GPU driver and PyTorch build, or use --device cpu"
        )
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or auto")
    return torch.device(name)


def select_precision(device, name="auto"):
    if name == "auto":
        name = (
            "float32"
            if device.type == "cpu"
            else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        )
    if name not in ("float32", "bfloat16", "float16"):
        raise ValueError("precision must be auto, float32, bfloat16, or float16")
    if device.type == "cpu" and name != "float32":
        raise ValueError("CPU verification uses float32; select --precision float32")
    if (
        device.type == "cuda"
        and name == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError(
            "this CUDA GPU does not support bfloat16; select float16 or float32"
        )
    return name


def autocast(device, precision):
    if precision == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, precision))


class Optimizers:
    """Several optimizers stepped as one, each over its own parameters."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        for optimizer, saved in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)


def make_optimizer(model, config, device):
    """Build the configured optimizer; weight decay applies to matrices only.

    Muon takes the matrices inside the blocks and AdamW the embedding and the
    norms. Muon scales each matrix's update to the size AdamW's would have, so
    both follow the same learning rate.
    """

    def adamw(parameters):
        return torch.optim.AdamW(
            [
                {
                    "params": [p for p in parameters if p.ndim >= 2],
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": [p for p in parameters if p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=device.type == "cuda",
        )

    if config.optimizer == "adamw":
        return adamw(list(model.parameters()))
    named = list(model.named_parameters())
    matrices = [p for name, p in named if p.ndim == 2 and name.startswith("blocks.")]
    chosen = {id(p) for p in matrices}
    return Optimizers(
        torch.optim.Muon(
            matrices,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        ),
        adamw([p for _, p in named if id(p) not in chosen]),
    )


@torch.inference_mode()
def evaluate(model, corpus, device, precision="float32", batches=16, batch_size=8):
    if batches <= 0 or batch_size <= 0:
        raise ValueError("evaluation batches and batch_size must be positive")
    sizes = torch.from_numpy(token_bytes(corpus.tokenizer)).to(device)
    was_training = model.training
    model.eval()
    result = {}
    try:
        for language in LANGUAGES:
            generator = torch.Generator().manual_seed(1729)
            total = bits = 0.0
            size = 0
            for _ in range(batches):
                x, y = corpus.batch(
                    "validation",
                    batch_size,
                    model.config.context,
                    generator,
                    device,
                    language,
                )
                with autocast(device, precision):
                    loss = model(x, y)
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError("validation loss is not finite")
                total += value
                # Instruction prompts and padding are -100 and add no loss.
                learned = y[y >= 0]
                bits += value * learned.numel() / math.log(2)
                size += sizes[learned].sum().item()
            loss = total / batches
            result[language] = {
                "loss": loss,
                "perplexity": math.exp(min(loss, 700)),
                "bits_per_byte": bits / size,
                "evaluated_tokens": batches * batch_size * model.config.context,
                "evaluated_bytes": size,
            }
    finally:
        model.train(was_training)
    for measure in ("loss", "bits_per_byte"):
        result[measure] = sum(
            result[language][measure] for language in LANGUAGES
        ) / len(LANGUAGES)
    return result


@torch.inference_mode()
def evaluate_windows(
    model, corpus, device, precision="float32", batch_size=8, limit=None
):
    """Measure the loss over every context-sized window of each validation stream.

    The windows follow one another from the start of the stream, so every
    model measured on the same data predicts exactly the same tokens. limit
    keeps the first windows of a long stream; its documents are stored in
    digest order, so the first windows sample its documents evenly. Bits per
    byte divide the bits of every prediction by the bytes of text the
    predicted tokens stand for, so models whose tokenizers differ compare on
    the same text.
    """
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("evaluation batch_size and window limit must be positive")
    context = model.config.context
    sizes = torch.from_numpy(token_bytes(corpus.tokenizer)).to(device)
    was_training = model.training
    model.eval()
    result = {}
    try:
        for language in LANGUAGES:
            stream = corpus.streams["validation"][language]
            count = (len(stream) - 1) // context
            if limit is not None:
                count = min(count, limit)
            total = 0.0
            size = 0
            for first in range(0, count, batch_size):
                x, y = to_device(
                    [
                        np.asarray(
                            stream[index * context : (index + 1) * context + 1],
                            dtype=np.int64,
                        )
                        for index in range(first, min(first + batch_size, count))
                    ],
                    device,
                )
                with autocast(device, precision):
                    loss = model(x, y)
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError("validation loss is not finite")
                total += value * y.numel()
                size += sizes[y].sum().item()
            loss = total / (count * context)
            result[language] = {
                "loss": loss,
                "perplexity": math.exp(min(loss, 700)),
                "bits_per_byte": total / math.log(2) / size,
                "evaluated_tokens": count * context,
                "evaluated_bytes": size,
            }
    finally:
        model.train(was_training)
    for measure in ("loss", "bits_per_byte"):
        result[measure] = sum(
            result[language][measure] for language in LANGUAGES
        ) / len(LANGUAGES)
    return result


def learning_rate(config, step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if config.schedule == "wsd":
        start = config.steps - config.decay_steps
        if step < start:
            return config.learning_rate
        progress = min(1, (step - start + 1) / config.decay_steps)
        if config.decay_shape == "1-sqrt":
            progress = math.sqrt(progress)
        return config.learning_rate - progress * (
            config.learning_rate - config.min_learning_rate
        )
    progress = (step - config.warmup_steps) / max(
        1, config.steps - config.warmup_steps - 1
    )
    return config.min_learning_rate + 0.5 * (
        config.learning_rate - config.min_learning_rate
    ) * (1 + math.cos(math.pi * progress))


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported pretraining checkpoint")
    return payload


def load_model(path, device):
    payload = load_checkpoint(path)
    tokenizer = Tokenizer.from_str(payload["tokenizer"])
    model = LanguageModel(
        tokenizer.get_vocab_size(), ModelConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model"])
    return model.to(device), tokenizer, payload


def same_weights(saved, config):
    """Whether a saved model carries over to this configuration.

    A rotary model may change only its context and the base of its rotation,
    as a phase that lengthens the context does, raising the base as Llama 3
    and SmolLM3 do; its weights do not depend on either.
    """
    saved = ModelConfig(**saved)
    if config.architecture == "llama":
        saved = replace(saved, context=config.context, rope_theta=config.rope_theta)
    return saved == config


def train(
    data,
    output,
    model_config,
    config,
    device,
    precision="auto",
    resume=None,
    max_steps=None,
    stop_requested=None,
    initialize_from=None,
    continue_from=None,
    compiled=False,
):
    """Train from scratch, or start from an earlier checkpoint.

    resume repeats an interrupted run exactly. initialize_from copies only the
    weights into a new run. continue_from carries the weights, optimizer and
    step into a new phase on other data, as long as the learning rate schedule
    continues without a jump. compiled runs the model through torch.compile,
    which fuses its small operations; checkpoints are the same either way.
    """
    if sum(item is not None for item in (resume, initialize_from, continue_from)) > 1:
        raise ValueError(
            "choose one of resume, initialize_from or continue_from for a training phase"
        )
    corpus = open_corpus(data)
    corpus.validate_context(model_config.context)
    precision = select_precision(device, precision)
    output = Path(output)
    if resume is None and (output / "last.pt").exists():
        raise ValueError(
            "a checkpoint already exists; use --resume or select a new output directory"
        )
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    if model_config.document_start not in (None, corpus.tokenizer.token_to_id(BOS)):
        raise ValueError("document_start must be the token that begins each document")
    model = LanguageModel(corpus.tokenizer.get_vocab_size(), model_config).to(device)
    optimizer = make_optimizer(model, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    generator = torch.Generator().manual_seed(config.seed)
    step, tokens_seen = 0, 0
    skipped_updates = 0
    # The step this phase began at and the row of the data it began reading.
    phase_start, phase_row = 0, 0
    initialization = None
    if initialize_from is not None:
        saved = load_checkpoint(initialize_from)
        if not same_weights(saved["model_config"], model_config):
            raise ValueError("initialization requires the same model configuration")
        if Tokenizer.from_str(saved["tokenizer"]).to_str() != corpus.tokenizer.to_str():
            raise ValueError("initialization requires the same tokenizer")
        model.load_state_dict(saved["model"])
        initialization = {
            "checkpoint_sha256": file_digest(initialize_from),
            "source_step": saved["step"],
            "source_tokens_seen": saved["tokens_seen"],
            "source_dataset_fingerprint": saved["dataset_fingerprint"],
        }
    if continue_from is not None:
        saved = load_checkpoint(continue_from)
        if not same_weights(saved["model_config"], model_config):
            raise ValueError("continuation requires the same model configuration")
        if Tokenizer.from_str(saved["tokenizer"]).to_str() != corpus.tokenizer.to_str():
            raise ValueError("continuation requires the same tokenizer")
        # The new schedule must agree with the rate of the last saved update, so
        # a decay phase or a longer stable phase picks up without a jump.
        previous = TrainConfig(**saved["train_config"])
        if previous.optimizer != config.optimizer:
            raise ValueError(
                "continuation keeps the optimizer and its state; "
                "use initialize_from to change the optimizer"
            )
        last = max(saved["step"] - 1, 0)
        if not math.isclose(
            learning_rate(previous, last), learning_rate(config, last), rel_tol=1e-9
        ):
            raise ValueError(
                "continuation must keep the learning rate used up to the saved step; "
                "use initialize_from to restart the schedule"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            # Only AdamW's groups have a fused kernel to select.
            if "fused" in group:
                group["fused"] = device.type == "cuda"
        if saved["precision"] == precision:
            scaler.load_state_dict(saved["scaler"])
        step, tokens_seen = saved["step"], saved["tokens_seen"]
        skipped_updates = saved["skipped_updates"]
        phase_start = step
        if saved["dataset_fingerprint"] == corpus.fingerprint:
            # The same data is read on from where the saved run stopped.
            if (previous.seed, previous.sampling) != (config.seed, config.sampling):
                raise ValueError(
                    "continuing on the same data keeps the seed and sampling that order it"
                )
            phase_row = saved.get("phase_start_row", 0) + (
                saved["step"] - saved.get("phase_start_step", 0)
            ) * previous.accumulation_steps * previous.batch_size
            generator.set_state(saved["sampling_rng"])
        initialization = {
            "continued_from": {
                "checkpoint_sha256": file_digest(continue_from),
                "step": saved["step"],
                "tokens_seen": saved["tokens_seen"],
                "dataset_fingerprint": saved["dataset_fingerprint"],
            },
            "earlier": saved.get("initialization"),
        }
    if resume:
        saved = load_checkpoint(resume)
        if saved["dataset_fingerprint"] != corpus.fingerprint:
            raise ValueError("resume requires the same prepared dataset")
        if (
            ModelConfig(**saved["model_config"]) != model_config
            or TrainConfig(**saved["train_config"]) != config
        ):
            raise ValueError(
                "resume requires the same model and training configuration"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            # Only AdamW's groups have a fused kernel to select.
            if "fused" in group:
                group["fused"] = device.type == "cuda"
        if saved["precision"] == precision:
            scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        step, tokens_seen = saved["step"], saved["tokens_seen"]
        skipped_updates = saved["skipped_updates"]
        phase_start = saved.get("phase_start_step", 0)
        phase_row = saved.get("phase_start_row", 0)
        initialization = saved.get("initialization")
    if compiled:
        # Training and validation batches keep one shape, so each compiles
        # once, and again for each power of two that their documents' bounds
        # are padded to; the limit leaves room for every such count in both.
        torch._dynamo.config.recompile_limit = 2 * (model_config.context * config.batch_size).bit_length()
        model.compile(dynamic=False)
    target = min(config.steps, max_steps) if max_steps is not None else config.steps
    if target <= step:
        raise ValueError(
            "the requested stopping step must be greater than the checkpoint step"
        )
    output.mkdir(parents=True, exist_ok=True)
    validation = evaluate(
        model, corpus, device, precision, config.eval_batches, config.batch_size
    )
    validation_step = step

    def checkpoint():
        return {
            "version": 1,
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "tokenizer": corpus.tokenizer.to_str(),
            "dataset_fingerprint": corpus.fingerprint,
            "step": step,
            "tokens_seen": tokens_seen,
            "validation": validation,
            "validation_step": validation_step,
            "sampling_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
            "precision": precision,
            "skipped_updates": skipped_updates,
            "torch_version": str(torch.__version__),
            "initialization": initialization,
            "phase_start_step": phase_start,
            "phase_start_row": phase_row,
        }

    atomic_save(checkpoint(), output / "last.pt")
    saved_step = step
    tokens_per_step = (
        config.batch_size * config.accumulation_steps * model_config.context
    )
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "precision": precision,
                "compiled": compiled,
                "parameters": sum(p.numel() for p in model.parameters()),
                "step": step,
                "target_step": target,
                "tokens_per_step": tokens_per_step,
                "target_tokens": config.steps * tokens_per_step,
                "validation": validation,
            }
        ),
        flush=True,
    )
    model.train()
    started, initial_tokens = time.perf_counter(), tokens_seen
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        while step < target:
            if stop_requested and stop_requested():
                break
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            total_loss = 0.0
            for micro in range(config.accumulation_steps):
                if config.sampling == "sequential":
                    x, y = corpus.sequential_batch(
                        phase_row
                        + ((step - phase_start) * config.accumulation_steps + micro)
                        * config.batch_size,
                        config.batch_size,
                        model_config.context,
                        config.seed,
                        device,
                    )
                else:
                    x, y = corpus.batch(
                        "train",
                        config.batch_size,
                        model_config.context,
                        generator,
                        device,
                    )
                with autocast(device, precision):
                    loss = model(x, y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        "training loss is not finite; inspect the data and saved checkpoint"
                    )
                total_loss += loss.detach().item() / config.accumulation_steps
                scaler.scale(loss / config.accumulation_steps).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
                error_if_nonfinite=precision != "float16",
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_updates += scaler.get_scale() < previous_scale
            step += 1
            tokens_seen += tokens_per_step
            stopping = bool(stop_requested and stop_requested())
            final = step == target or stopping
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "training_loss": total_loss,
                "learning_rate": rate,
                "skipped_updates": skipped_updates,
                "tokens_per_second": (tokens_seen - initial_tokens)
                / (time.perf_counter() - started),
            }
            if step % config.eval_every == 0 or final:
                validation = evaluate(
                    model,
                    corpus,
                    device,
                    precision,
                    config.eval_batches,
                    config.batch_size,
                )
                validation_step = step
                record["validation"] = validation
            if step % config.save_every == 0 or final:
                atomic_save(checkpoint(), output / "last.pt")
                saved_step = step
            if step % config.log_every == 0 or "validation" in record:
                line = json.dumps(record, allow_nan=False)
                metrics.write(line + "\n")
                metrics.flush()
                print(line, flush=True)
            if stopping:
                break
    # A signal can arrive after the last iteration's stop check. Persist every
    # completed update even when the next iteration observes that signal first.
    if saved_step != step:
        if validation_step != step:
            validation = evaluate(
                model, corpus, device, precision, config.eval_batches, config.batch_size
            )
            validation_step = step
        atomic_save(checkpoint(), output / "last.pt")
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "validation": validation,
    }
