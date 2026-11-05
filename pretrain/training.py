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
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from .count import CountTable
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
    # once per epoch in a seeded order; "ordered" reads them once per epoch
    # from the lowest quality to the highest, from a corpus written in that
    # order, shuffling each hundredth of every source together.
    sampling: str = "random"
    # "adamw" updates every weight with AdamW; "muon" updates the matrices in
    # the blocks with Muon and the embedding and norms with AdamW; "normuon"
    # gives the matrices NorMuon, as nanochat does.
    optimizer: str = "adamw"
    # "constant" keeps weight_decay; "cosine" takes the decay of the matrices
    # NorMuon updates from weight_decay down to zero over the run, as nanochat
    # does, so the weights settle unpulled by the end.
    weight_decay_schedule: str = "constant"
    # Above zero, NorMuon holds the matrices of the attention layers and
    # feed-forward networks at the Frobenius norms they start with (Hyperball,
    # arxiv.org/abs/2606.16899), as Puro-2B trained them: each step moves a
    # matrix along its normalized update by this multiple of the learning rate
    # times its norm, then scales it back to that norm, without weight decay.
    # The attention's sigmoid gates, whose scale sets how open they are, go to
    # AdamW with the embedding. Puro-2B used 10.
    hyperball_rate: float = 0.0
    # Every this many steps of a wsd decay, the model's weights are kept apart
    # from last.pt, for averaging the last of them after training; 0 keeps none.
    snapshot_every: int = 0
    # Multiply the large matrices in FP8; the weights stay as they are.
    fp8: bool = False
    # Read this many tokens as one position; context then counts patches.
    patch_size: int = 1
    # The share of their rates that the residual scalars take.
    scalar_rate: float = 1.0
    # How far each update moves the biases of experts balanced by bias toward
    # even use, DeepSeek-V3's 0.001.
    expert_bias_rate: float = 0.001

    def __post_init__(self):
        for field in (
            "patch_size",
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
            "scalar_rate",
            "hyperball_rate",
            "expert_bias_rate",
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
        if self.sampling not in ("random", "sequential", "ordered"):
            raise ValueError("sampling must be random, sequential or ordered")
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
        if self.optimizer not in ("adamw", "muon", "normuon"):
            raise ValueError("optimizer must be adamw, muon or normuon")
        if self.weight_decay_schedule not in ("constant", "cosine"):
            raise ValueError("weight_decay_schedule must be constant or cosine")
        if self.weight_decay_schedule == "cosine" and self.optimizer != "normuon":
            raise ValueError("a cosine weight decay schedule applies to normuon")
        if self.hyperball_rate and self.optimizer != "normuon":
            raise ValueError("hyperball_rate applies to normuon")
        if self.hyperball_rate and self.weight_decay_schedule != "constant":
            raise ValueError("the matrices hyperball holds take no weight decay to schedule")
        if type(self.snapshot_every) is not int or self.snapshot_every < 0:
            raise ValueError("snapshot_every must be a nonnegative integer")
        if self.snapshot_every and self.schedule != "wsd":
            raise ValueError("snapshots are kept through a wsd decay")
        if type(self.fp8) is not bool:
            raise ValueError("fp8 must be true or false")


# Ada, the oldest generation whose matrix products take FP8.
FP8_CAPABILITY = (8, 9)


def check_fp8(device, precision, compiled):
    """Refuse FP8 where it cannot run, or where it slows training down."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device) < FP8_CAPABILITY:
        raise ValueError("fp8 needs an Ada or newer GPU")
    if precision != "bfloat16":
        raise ValueError("fp8 runs alongside bfloat16")
    if not compiled:
        # nanochat measured FP8 four times slower than bfloat16 uncompiled.
        raise ValueError("fp8 needs torch.compile, which fuses its scaling")


def fp8_matrix(module, _name):
    """Whether a module is a matrix FP8 can multiply and gains anything from.

    The hardware needs sides divisible by 16, and scaling costs more than FP8
    saves on small matrices, as nanochat found.
    """
    return (
        isinstance(module, torch.nn.Linear)
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
        and min(module.in_features, module.out_features) >= 128
    )


def convert_to_fp8(model):
    """Multiply the model's large matrices in FP8, with one scale per tensor.

    The modules keep their weights under the same names, so the optimizer and
    the checkpoints see no difference.
    """
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    convert_to_float8_training(
        model,
        config=Float8LinearConfig.from_recipe_name("tensorwise"),
        module_filter_fn=fp8_matrix,
    )


def read_length(context, patch_size):
    """The context a data batch needs so that patches and their next patches fit."""
    return context * patch_size + patch_size - 1


def patch_batch(inputs, targets, patch_size):
    """Whole patches of a read_length batch, and for each, the next patch."""
    return inputs[:, : inputs.shape[1] - patch_size + 1], targets[:, patch_size - 1 :]


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


def orthogonalize(matrix, coefficients, steps, eps):
    """Muon's Newton-Schulz orthogonalization of a matrix, or of each matrix in a stack.

    A stack runs PyTorch's iteration with batched products, so each of its
    matrices comes out as PyTorch makes a matrix on its own.
    """
    if matrix.ndim == 2:
        return _zeropower_via_newtonschulz(matrix, coefficients, steps, eps)
    a, b, c = coefficients
    tall = matrix.size(-2) > matrix.size(-1)
    result = matrix.bfloat16()
    if tall:
        result = result.mT
    result = result / torch.linalg.vector_norm(result, dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = result @ result.mT
        result = torch.baddbmm(result, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), result, beta=a)
    return result.mT if tall else result


class NorMuon(torch.optim.Muon):
    """Muon whose orthogonalized updates are evened out across neurons, with cautious decay.

    NorMuon (arxiv.org/abs/2510.05491) keeps a running mean of each output
    neuron's squared update, of each row of a tall or square matrix and each
    column of a wide one, and divides by its root so that no neuron's update
    outgrows the others', then rescales the update to its orthogonalized
    norm. Cautious weight decay (arxiv.org/abs/2510.12402) pulls only the
    weights the update already moves toward zero. Both follow nanochat's
    implementation; the update keeps Muon's size, so match_rms_adamw scales it
    as it scales Muon's.

    A group with hyperball set instead keeps each matrix at the Frobenius norm
    it had when this optimizer first moved it (arxiv.org/abs/2606.16899): the
    update, scaled to the learning rate times that norm, moves the matrix,
    which is then scaled back to that norm, without weight decay.

    A stack of matrices, as the experts of a mixture hold theirs, inputs
    first, is updated as each of its matrices on its own, read as the
    outputs-by-inputs matrix a Linear layer holds, so that NorMuon evens out
    its output neurons.
    """

    def __init__(self, params, beta2=0.9, hyperball=False, **options):
        # Muon's constructor checks the options but refuses stacks of matrices,
        # so the options pass its checks on a stand-in and the parameters,
        # matrices or stacks of them, go straight to the optimizer.
        defaults = torch.optim.Muon([torch.nn.Parameter(torch.zeros(1, 1))], **options).defaults
        torch.optim.Optimizer.__init__(self, params, defaults)
        for group in self.param_groups:
            if any(parameter.ndim not in (2, 3) for parameter in group["params"]):
                raise ValueError("NorMuon takes matrices and stacks of matrices")
            group.setdefault("beta2", beta2)
            group.setdefault("hyperball", hyperball)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                stacked = grad.ndim == 3
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    *stack, rows, columns = (grad.mT if stacked else grad).shape
                    state["second_moment"] = grad.new_zeros(
                        (*stack, rows, 1) if rows >= columns else (*stack, 1, columns), dtype=torch.float32
                    )
                    if group.get("hyperball"):
                        state["radius"] = (
                            torch.linalg.vector_norm(parameter, dim=(-2, -1), keepdim=True)
                            if stacked else torch.linalg.vector_norm(parameter)
                        )
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = orthogonalize(
                    update.mT if stacked else update, group["ns_coefficients"], group["ns_steps"], group["eps"]
                ).float()
                axis = -1 if update.size(-2) >= update.size(-1) else -2
                mean = update.square().mean(dim=axis, keepdim=True)
                state["second_moment"].lerp_(mean, 1 - group["beta2"])
                evened = update * state["second_moment"].clamp_min(1e-10).rsqrt()
                update = evened * (
                    torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
                    / torch.linalg.vector_norm(evened, dim=(-2, -1), keepdim=True).clamp_min(1e-10)
                )
                update = (update.mT if stacked else update).to(parameter.dtype)
                if group.get("hyperball"):
                    radius = state["radius"]
                    parameter.sub_(update * (
                        group["lr"] * radius
                        / torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True).clamp_min(1e-10)
                    ))
                    parameter.mul_(
                        radius / torch.linalg.vector_norm(parameter, dim=(-2, -1), keepdim=True).clamp_min(1e-10)
                    )
                    continue
                if stacked:
                    raise ValueError("NorMuon takes stacks of matrices under hyperball only")
                decay = group["lr"] * group["weight_decay"]
                parameter.sub_(decay * parameter * (update * parameter >= 0))
                parameter.add_(
                    update, alpha=-_adjust_lr(group["lr"], group["adjust_lr_fn"], parameter.shape)
                )
        return loss


def weight_decay(config, step):
    """The decay of the matrices NorMuon updates at a step."""
    if config.weight_decay_schedule == "cosine":
        return config.weight_decay * 0.5 * (1 + math.cos(math.pi * step / config.steps))
    return config.weight_decay


# nanochat's AdamW settings for the residual scalars at its reference batch
# of 2**19 tokens: the scales of the stream multiply through the layers and
# take a hundredth of the rate of the mixing scales.
SCALAR_BATCH = 2**19
SCALAR_SETTINGS = {
    "residual_scales": {"lr": 0.005, "betas": (0.8, 0.95)},
    "input_scales": {"lr": 0.5, "betas": (0.96, 0.95)},
}

# The matrices of the attention layers and the feed-forward networks, which
# Puro-2B held with Hyperball as weights nearly free of scale; the experts of
# a mixture, and its shared expert, hold their networks' matrices under the
# same names.
HYPERBALL_MATRICES = ("query", "key_value", "attention_output", "gate", "up", "down")


def make_optimizer(model, config, device):
    """Build the configured optimizer; weight decay applies to matrices only.

    Muon, or NorMuon, takes the matrices inside the blocks and AdamW the
    embedding and the norms. Muon scales each matrix's update to the size
    AdamW's would have, so both follow the same learning rate. The residual
    scalars take nanochat's rates instead, grown with the square root of the
    batch as its rates are, times the config's scalar_rate, and follow the
    schedule in proportion through each group's lr_scale. With hyperball_rate,
    NorMuon takes only HYPERBALL_MATRICES, the experts' stacks of them
    included, at that multiple of the rate, and AdamW the rest, the experts'
    router among them.
    """
    scalars = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if name in SCALAR_SETTINGS
    }
    batch = config.batch_size * config.accumulation_steps * model.config.context * config.patch_size
    scalar_groups = [
        {
            "params": [parameter],
            "weight_decay": 0.0,
            "betas": SCALAR_SETTINGS[name]["betas"],
            "eps": 1e-10,
            "lr_scale": SCALAR_SETTINGS[name]["lr"]
            * config.scalar_rate
            * math.sqrt(batch / SCALAR_BATCH)
            / config.learning_rate,
        }
        for name, parameter in model.named_parameters()
        if name in SCALAR_SETTINGS
    ]

    def adamw(parameters):
        parameters = [p for p in parameters if id(p) not in scalars]
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
                *scalar_groups,
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=device.type == "cuda",
        )

    if config.optimizer == "adamw":
        return adamw(list(model.parameters()))
    named = list(model.named_parameters())
    if config.hyperball_rate:
        shape = model.config
        if shape.architecture != "llama" or shape.full_attention_every is not None:
            raise ValueError("hyperball holds the matrices of attention layers and feed-forward networks")
        if shape.zero_output_init:
            raise ValueError("hyperball keeps each matrix at the norm it starts with, so none may start at zero")
        matrices = [
            p for name, p in named if name.startswith("blocks.") and (
                name.split(".")[-2] in HYPERBALL_MATRICES
                or name.split(".")[-2] == "experts" and name.split(".")[-1] in HYPERBALL_MATRICES
            )
        ]
        first = NorMuon(
            [{"params": matrices, "lr_scale": config.hyperball_rate}],
            lr=config.learning_rate,
            weight_decay=0.0,
            hyperball=True,
        )
    else:
        matrices = [p for name, p in named if p.ndim == 2 and name.startswith("blocks.")]
        first = (NorMuon if config.optimizer == "normuon" else torch.optim.Muon)(
            matrices,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
    chosen = {id(p) for p in matrices}
    return Optimizers(first, adamw([p for _, p in named if id(p) not in chosen]))


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
                    read_length(model.config.context, model.patch_size),
                    generator,
                    device,
                    language,
                )
                x, y = patch_batch(x, y, model.patch_size)
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
                "evaluated_tokens": batches * batch_size * model.config.context * model.patch_size,
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


def saved_count_table(payload):
    """The count table a checkpoint carries, or None."""
    saved = payload.get("count_table")
    return None if saved is None else CountTable.from_checkpoint(saved)


def load_model(path, device):
    payload = load_checkpoint(path)
    tokenizer = Tokenizer.from_str(payload["tokenizer"])
    model = LanguageModel(
        tokenizer.get_vocab_size(),
        ModelConfig(**payload["model_config"]),
        count_table=saved_count_table(payload),
    )
    model.load_state_dict(payload["model"])
    return model.to(device), tokenizer, payload


def average_snapshots(run, count, output):
    """Write the run's last checkpoint with its weights averaged over its last snapshots.

    LAWA (arXiv 2306.03241) and IMU-1 (arXiv 2602.02522) found the average of
    a run's last checkpoints better than the last alone. The average is plain,
    as LAWA's is; the rest of the checkpoint is last.pt's.
    """
    run = Path(run)
    if type(count) is not int or count < 2:
        raise ValueError("average at least two snapshots")
    snapshots = sorted((run / "snapshots").glob("*.pt"))[-count:]
    if len(snapshots) < count:
        raise ValueError(f"the run keeps {len(snapshots)} snapshots, fewer than {count}")
    payload = load_checkpoint(run / "last.pt")
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in snapshots]
    if saved[-1]["step"] != payload["step"]:
        raise ValueError("the last snapshot must be the last checkpoint's step")
    payload["model"] = {
        name: (sum(item["model"][name].float() for item in saved) / count).to(value.dtype)
        for name, value in payload["model"].items()
    }
    payload["averaged_steps"] = [item["step"] for item in saved]
    atomic_save(payload, output)
    return {"output": str(output), "averaged_steps": payload["averaged_steps"]}


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
    count_table=None,
):
    """Train from scratch, or start from an earlier checkpoint.

    resume repeats an interrupted run exactly. initialize_from copies only the
    weights into a new run. continue_from carries the weights, optimizer and
    step into a new phase on other data, as long as the learning rate schedule
    continues without a jump. compiled runs the model through torch.compile,
    which fuses its small operations; checkpoints are the same either way.
    A model that reads a count table takes it from the checkpoint it starts
    from, or else from count_table, and keeps it in every checkpoint.
    """
    if sum(item is not None for item in (resume, initialize_from, continue_from)) > 1:
        raise ValueError(
            "choose one of resume, initialize_from or continue_from for a training phase"
        )
    corpus = open_corpus(data)
    corpus.validate_context(read_length(model_config.context, config.patch_size))
    if config.sampling == "ordered" and not corpus.manifest.get("quality"):
        raise ValueError("ordered sampling reads a corpus written in quality order")
    precision = select_precision(device, precision)
    if config.fp8:
        check_fp8(device, precision, compiled)
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
    source = next((path for path in (resume, initialize_from, continue_from) if path), None)
    saved = None if source is None else load_checkpoint(source)
    table = count_table
    if model_config.count_table:
        kept = None if saved is None else saved_count_table(saved)
        if kept is not None and table is not None and kept.digest != table.digest:
            raise ValueError("the count table differs from the one the checkpoint carries")
        table = kept or table
        if table is None:
            raise ValueError("the model reads a count table; build one with count-table and give it")
        table.check_tokenizer(corpus.tokenizer)
    elif table is not None:
        raise ValueError("the model's configuration reads no count table")
    model = LanguageModel(
        corpus.tokenizer.get_vocab_size(),
        model_config,
        patch_size=config.patch_size,
        count_table=table,
    ).to(device)
    if config.fp8:
        convert_to_fp8(model)
    optimizer = make_optimizer(model, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    generator = torch.Generator().manual_seed(config.seed)
    step, tokens_seen = 0, 0
    skipped_updates = 0
    # The step this phase began at and the row of the data it began reading.
    phase_start, phase_row = 0, 0
    initialization = None
    if initialize_from is not None:
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
        if not same_weights(saved["model_config"], model_config):
            raise ValueError("continuation requires the same model configuration")
        if Tokenizer.from_str(saved["tokenizer"]).to_str() != corpus.tokenizer.to_str():
            raise ValueError("continuation requires the same tokenizer")
        # The new schedule must agree with the rate of the last saved update, so
        # a decay phase or a longer stable phase picks up without a jump.
        previous = TrainConfig(**saved["train_config"])
        if (previous.optimizer, previous.hyperball_rate) != (config.optimizer, config.hyperball_rate):
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
        # The scalars' rates follow this phase's batch, not the saved one's.
        scales = [group.get("lr_scale") for group in optimizer.param_groups]
        optimizer.load_state_dict(saved["optimizer"])
        for group, scale in zip(optimizer.param_groups, scales, strict=True):
            if scale is not None:
                group["lr_scale"] = scale
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
            "count_table": None if table is None else table.checkpoint(),
        }

    atomic_save(checkpoint(), output / "last.pt")
    saved_step = step
    tokens_per_step = (
        config.batch_size * config.accumulation_steps * model_config.context * config.patch_size
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
                group["lr"] = rate * group.get("lr_scale", 1.0)
            if config.optimizer == "normuon" and not config.hyperball_rate:
                for group in optimizer.optimizers[0].param_groups:
                    group["weight_decay"] = weight_decay(config, step)
            # The update's loss stays on the GPU until its last window, so that
            # no window waits for the GPU to finish the one before it.
            losses = torch.zeros((), device=device)
            for micro in range(config.accumulation_steps):
                if config.sampling != "random":
                    read = corpus.ordered_batch if config.sampling == "ordered" else corpus.sequential_batch
                    x, y = read(
                        phase_row
                        + ((step - phase_start) * config.accumulation_steps + micro)
                        * config.batch_size,
                        config.batch_size,
                        read_length(model_config.context, config.patch_size),
                        config.seed,
                        device,
                    )
                else:
                    x, y = corpus.batch(
                        "train",
                        config.batch_size,
                        read_length(model_config.context, config.patch_size),
                        generator,
                        device,
                    )
                x, y = patch_batch(x, y, config.patch_size)
                with autocast(device, precision):
                    loss = model(x, y)
                losses += loss.detach().float() / config.accumulation_steps
                scaler.scale(loss / config.accumulation_steps).backward()
            total_loss = losses.item()
            if not math.isfinite(total_loss):
                raise RuntimeError(
                    "training loss is not finite; inspect the data and saved checkpoint"
                )
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
            # Experts balanced by bias move toward even use by this update's tokens.
            expert_share = model.update_expert_bias(config.expert_bias_rate)
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
            if expert_share is not None:
                # The most any expert took of an even share of this update's tokens.
                record["expert_share"] = expert_share
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
            decayed = step - (config.steps - config.decay_steps)
            if config.snapshot_every and decayed > 0 and decayed % config.snapshot_every == 0:
                atomic_save(
                    {"step": step, "tokens_seen": tokens_seen, "model": model.state_dict()},
                    output / "snapshots" / f"{step:07d}.pt",
                )
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
