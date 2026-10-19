"""Measure native GPU training throughput with synthetic token batches."""

import gc
import math
from time import perf_counter

import torch

from .model import LanguageModel, ModelConfig
from .training import TrainConfig, autocast


def measure_training(
    config, vocabulary_size=32768, batches=(1, 2, 4, 8), compiled=False
):
    """Time optimizer steps of the configured size at each micro-batch size.

    Every step covers the configuration's tokens per step and accumulates
    gradients over as many micro-batches as that takes, as training does.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Run this measurement on a CUDA GPU with bfloat16 support")
    model_config = ModelConfig(**config["model"])
    training = TrainConfig(**config["training"])
    rows = training.batch_size * training.accumulation_steps
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    model = LanguageModel(vocabulary_size, model_config).to(device).train()
    if compiled:
        # Training shapes never change, so each micro-batch size compiles once.
        model.compile(dynamic=False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.ndim >= 2],
                "weight_decay": training.weight_decay,
            },
            {
                "params": [p for p in model.parameters() if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=training.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    def update(batch_size, accumulation):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            tokens = torch.randint(
                2,
                vocabulary_size,
                (batch_size, model_config.context + 1),
                device=device,
            )
            inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            with autocast(device, "bfloat16"):
                loss = model(inputs, targets)
            # Training reads each micro-batch's loss, waiting for its forward pass.
            if not math.isfinite(loss.detach().item()):
                raise RuntimeError("Synthetic training loss is not finite")
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()

    profiles = []
    for batch_size in batches:
        accumulation, remainder = divmod(rows, batch_size)
        if remainder:
            profiles.append(
                {"batch_size": batch_size, "status": "does_not_divide_the_step"}
            )
            continue
        try:
            started = perf_counter()
            for _ in range(2):
                update(batch_size, accumulation)
            torch.cuda.synchronize()
            warmup = perf_counter() - started
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            steps = 0
            while steps < 32:
                update(batch_size, accumulation)
                steps += 1
                torch.cuda.synchronize()
                if perf_counter() - started >= 10:
                    break
            elapsed = perf_counter() - started
            profiles.append(
                {
                    "batch_size": batch_size,
                    "accumulation_steps": accumulation,
                    "warmup_seconds": warmup,
                    "measured_steps": steps,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": steps * rows * model_config.context / elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )
        except torch.OutOfMemoryError:
            profiles.append({"batch_size": batch_size, "status": "out_of_memory"})
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "torch_version": str(torch.__version__),
        "precision": "bfloat16",
        "compiled": compiled,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": vocabulary_size,
        "context": model_config.context,
        "tokens_per_step": rows * model_config.context,
        "input": "synthetic_tokens",
        "includes": [
            "forward",
            "backward",
            "gradient_accumulation",
            "gradient_clipping",
            "fused_adamw",
        ],
        "excludes": ["dataset_io", "checkpoints", "validation", "preemption"],
        "profiles": profiles,
    }
