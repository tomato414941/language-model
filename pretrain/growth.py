"""Grow a trained model into a deeper one that predicts the same.

Each layer is followed by copies of itself whose attention and feed-forward
results start at zero, as LLaMA Pro's block expansion (arXiv:2401.02415)
adds layers. The grown model begins where the trained one left off, and the
copies learn from there.
"""

from dataclasses import asdict, replace

from .data import file_digest
from .model import ModelConfig
from .training import atomic_save, load_checkpoint

# The projections that add a layer's attention and feed-forward results to the
# residual stream.
OUTPUTS = ("attention_output.weight", "down.weight")
# States that only continue the run that saved them.
RUN_STATES = ("optimizer", "scaler", "sampling_rng", "torch_rng", "cuda_rng")


def grow_checkpoint(checkpoint, output, factor=2):
    """Save a model with `factor` times the layers, for a new run to start from."""
    if type(factor) is not int or factor < 2:
        raise ValueError("factor must be an integer of at least 2")
    saved = load_checkpoint(checkpoint)
    config = ModelConfig(**saved["model_config"])
    if config.architecture != "llama":
        raise ValueError("growth needs the llama architecture")
    weights = {}
    for name, value in saved["model"].items():
        if not name.startswith("blocks."):
            weights[name] = value
            continue
        _, layer, rest = name.split(".", 2)
        for copy in range(factor):
            added = copy > 0 and rest in OUTPUTS
            weights[f"blocks.{int(layer) * factor + copy}.{rest}"] = (
                value.new_zeros(value.shape) if added else value.clone()
            )
    grown = {key: value for key, value in saved.items() if key not in RUN_STATES}
    grown["model_config"] = asdict(replace(config, layers=config.layers * factor))
    grown["model"] = weights
    grown["growth"] = {
        "checkpoint_sha256": file_digest(checkpoint),
        "source_layers": config.layers,
        "factor": factor,
    }
    atomic_save(grown, output)
    return {
        "output": str(output),
        "layers": config.layers * factor,
        "parameters": sum(value.numel() for value in weights.values()),
        "sha256": file_digest(output),
        "source": grown["growth"],
    }
