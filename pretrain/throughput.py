"""Measure native GPU training throughput with synthetic token batches."""

import gc
import math
from time import perf_counter

import torch
from torch._dynamo.utils import counters

from .model import LanguageModel, ModelConfig
from .training import TrainConfig, autocast, make_optimizer


def measure_training(
    config,
    vocabulary_size=32768,
    batches=(1, 2, 4, 8),
    compiled=False,
    document_length=None,
    warmup_steps=2,
):
    """Time optimizer steps of the configured size at each micro-batch size.

    Every step covers the configuration's tokens per step and accumulates
    gradients over as many micro-batches as that takes, as training does.
    With document_length, a document starts at each position with one chance
    in that many, so attention within documents meets the varying bounds of
    real text; compiled code then compiles once per shape it meets, during
    the warmup_steps if they are enough.
    """
    model_config = ModelConfig(**config["model"])
    if document_length is not None and model_config.document_start is None:
        raise ValueError("documents need the model's document_start token")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Run this measurement on a CUDA GPU with bfloat16 support")
    training = TrainConfig(**config["training"])
    rows = training.batch_size * training.accumulation_steps
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    model = LanguageModel(vocabulary_size, model_config).to(device).train()
    if compiled:
        # Each micro-batch size compiles once, and again for each padded count
        # of documents it meets. Code compiled for an earlier model would
        # count against this one's limit on recompilations.
        torch._dynamo.reset()
        counters.clear()
        torch._dynamo.config.recompile_limit = 2 * (model_config.context * max(batches)).bit_length()
        model.compile(dynamic=False)
    optimizer = make_optimizer(model, training, device)

    def update(batch_size, accumulation):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            tokens = torch.randint(
                2,
                vocabulary_size,
                (batch_size, model_config.context + 1),
                device=device,
            )
            if document_length is not None:
                tokens[torch.rand(tokens.shape, device=device) < 1 / document_length] = (
                    model_config.document_start
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
            for _ in range(warmup_steps):
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
        "compiled_graphs": counters["stats"]["unique_graphs"] if compiled else None,
        "graph_breaks": sum(counters["graph_break"].values()) if compiled else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": vocabulary_size,
        "context": model_config.context,
        "tokens_per_step": rows * model_config.context,
        "input": "synthetic_tokens",
        "document_length": document_length,
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
