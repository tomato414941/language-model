import json
import math
import os
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .data import LANGUAGES, Corpus
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


@torch.inference_mode()
def evaluate(model, corpus, device, precision="float32", batches=16, batch_size=8):
    if batches <= 0 or batch_size <= 0:
        raise ValueError("evaluation batches and batch_size must be positive")
    was_training = model.training
    model.eval()
    result = {}
    try:
        for language in LANGUAGES:
            generator = torch.Generator().manual_seed(1729)
            total = 0.0
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
            loss = total / batches
            result[language] = {
                "loss": loss,
                "perplexity": math.exp(min(loss, 700)),
                "evaluated_tokens": batches * batch_size * model.config.context,
            }
    finally:
        model.train(was_training)
    result["loss"] = sum(result[language]["loss"] for language in LANGUAGES) / len(
        LANGUAGES
    )
    return result


def learning_rate(config, step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
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
):
    corpus = Corpus(data)
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
    model = LanguageModel(corpus.tokenizer.get_vocab_size(), model_config).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.ndim >= 2],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for p in model.parameters() if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    generator = torch.Generator().manual_seed(config.seed)
    step, tokens_seen, best_loss = 0, 0, math.inf
    skipped_updates = 0
    if resume:
        saved = load_checkpoint(resume)
        if saved["dataset_fingerprint"] != corpus.fingerprint:
            raise ValueError("resume requires the same prepared dataset")
        if saved["model_config"] != asdict(model_config) or saved[
            "train_config"
        ] != asdict(config):
            raise ValueError(
                "resume requires the same model and training configuration"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["fused"] = device.type == "cuda"
        if saved["precision"] == precision:
            scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        step, tokens_seen, best_loss = (
            saved["step"],
            saved["tokens_seen"],
            saved["best_loss"],
        )
        skipped_updates = saved["skipped_updates"]
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
            "best_loss": best_loss,
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
        }

    if validation["loss"] < best_loss:
        best_loss = validation["loss"]
        atomic_save(checkpoint(), output / "best.pt")
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
            for _ in range(config.accumulation_steps):
                x, y = corpus.batch(
                    "train", config.batch_size, model_config.context, generator, device
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
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    atomic_save(checkpoint(), output / "best.pt")
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
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            atomic_save(checkpoint(), output / "best.pt")
        atomic_save(checkpoint(), output / "last.pt")
    return {
        "step": step,
        "tokens_seen": tokens_seen,
        "validation": validation,
        "best_loss": best_loss,
    }
