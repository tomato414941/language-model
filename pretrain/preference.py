"""Direct preference optimization on pairs of replies to the same conversation.

The policy learns to give the preferred reply more probability than the
rejected one, relative to the model it starts from, which stays frozen as the
reference and keeps the policy near it (Rafailov et al., 2023). Each reply's
log-probability may be divided by its length, as Tülu 3 found best.

Pairs arrive as JSON lines, one file per language, each with the conversation
before the reply ("prompt", a list of messages) and the preferred and
rejected replies ("chosen", "rejected"). They are written in the
conversation format of instruction tuning, and only the final reply and its
end token count.
"""

import copy
import hashlib
import json
import math
import random
import shutil
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from .conversation import conversation_text, require_markers
from .data import BOS, EOS, LANGUAGES, TOKEN_DTYPE, file_digest
from .instructions import MASK_DTYPE, BenchmarkFilter, contaminated, normalize_messages
from .training import atomic_save, autocast, load_checkpoint, load_model, select_precision

SIDES = ("chosen", "rejected")


@dataclass(frozen=True)
class PreferenceConfig:
    steps: int = 1000
    batch_size: int = 4
    accumulation_steps: int = 8
    learning_rate: float = 1e-6
    warmup_steps: int = 100
    # How strongly the policy may move from the reference; length-normalized
    # scores are smaller, so they take a larger beta, as Tülu 3 used 5.
    beta: float = 0.1
    length_normalized: bool = False
    gradient_clip: float = 1.0
    eval_every: int = 100
    log_every: int = 10
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "batch_size", "accumulation_steps", "eval_every", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be a nonnegative integer smaller than steps")
        for name in ("learning_rate", "beta", "gradient_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.length_normalized) is not bool:
            raise ValueError("length_normalized must be true or false")


def read_config(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"preference"}:
        raise ValueError("the configuration holds one preference section")
    return PreferenceConfig(**config["preference"])


def encode_reply(tokenizer, turns):
    """Tokens of a conversation and the mask of its final reply and end token."""
    require_markers(tokenizer)
    bos, eos = tokenizer.token_to_id(BOS), tokenizer.token_to_id(EOS)
    text, replies = conversation_text(turns)
    start, end = replies[-1]
    encoding = tokenizer.encode(text, add_special_tokens=False)
    starts = np.asarray([offset for offset, _ in encoding.offsets], dtype=np.int64)
    learn = (starts >= start) & (starts < end)
    return (
        np.asarray([bos, *encoding.ids, eos], dtype=TOKEN_DTYPE),
        np.concatenate(([0], learn, [1])).astype(MASK_DTYPE),
    )


def pair_turns(record):
    """The conversation with each reply, or None if the record is not a usable pair."""
    prompt, chosen, rejected = (record.get(key) for key in ("prompt", *SIDES))
    if not isinstance(prompt, list) or not prompt or not all(
        isinstance(reply, str) and reply.strip() for reply in (chosen, rejected)
    ):
        return None
    sides = []
    for reply in (chosen, rejected):
        turns = normalize_messages([*prompt, {"role": "assistant", "content": reply}])
        # The final turn must stay the reply, after a prompt that ends with the user.
        if turns is None or len(turns) != len(prompt) + 1:
            return None
        sides.append(turns)
    if sides[0][-1]["content"] == sides[1][-1]["content"]:
        return None
    return sides


def prepare_preferences(
    source, output, tokenizer_path, context=2048, validation_pairs=500, benchmarks=None, seed=42
):
    """Encode each language's pairs into fixed-width rows, holding out validation pairs.

    Pairs that repeat an earlier one, contain a benchmark question or do not
    fit the context are left out.
    """
    for name, value in (("context", context), ("validation_pairs", validation_pairs)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new dataset directory")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id(BOS) is None or tokenizer.token_to_id(EOS) is None:
        raise ValueError("the tokenizer needs BOS and EOS tokens")
    eos = tokenizer.token_to_id(EOS)
    width = context + 1
    questions = BenchmarkFilter(benchmarks) if benchmarks is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
        manifest = {
            "version": 1,
            "kind": "preferences",
            "dtype": TOKEN_DTYPE.str,
            "context": context,
            "seed": seed,
            "vocabulary_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "sources": {
                language: file_digest(source / f"{language}.jsonl") for language in LANGUAGES
            },
            "benchmarks_sha256": file_digest(Path(benchmarks) / "sources.json")
            if benchmarks is not None
            else None,
            "pairs": {},
            "files": {},
        }
        for language in LANGUAGES:
            stats = dict.fromkeys(("malformed", "duplicates", "contaminated", "too_long"), 0)
            seen, encoded = set(), []
            with (source / f"{language}.jsonl").open(encoding="utf-8") as lines:
                for line in lines:
                    sides = pair_turns(json.loads(line))
                    if sides is None:
                        stats["malformed"] += 1
                        continue
                    digest = hashlib.sha256(
                        json.dumps(sides, ensure_ascii=False, sort_keys=True).encode()
                    ).digest()
                    if digest in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(digest)
                    if contaminated(sides[0][:-1], questions, language):
                        stats["contaminated"] += 1
                        continue
                    pair = [encode_reply(tokenizer, turns) for turns in sides]
                    if any(len(tokens) > width for tokens, _ in pair):
                        stats["too_long"] += 1
                        continue
                    encoded.append(pair)
            random.Random(f"{seed}:{language}").shuffle(encoded)
            splits = {"validation": encoded[:validation_pairs], "train": encoded[validation_pairs:]}
            if not splits["train"]:
                raise ValueError(f"{language} has no training pairs after holding out validation")
            for split, pairs in splits.items():
                for index, side in enumerate(SIDES):
                    tokens = np.full((len(pairs), width), eos, dtype=TOKEN_DTYPE)
                    mask = np.zeros((len(pairs), width), dtype=MASK_DTYPE)
                    for row, pair in enumerate(pairs):
                        pair_tokens, pair_mask = pair[index]
                        tokens[row, : len(pair_tokens)] = pair_tokens
                        mask[row, : len(pair_mask)] = pair_mask
                    for suffix, array in (("", tokens), ("_mask", mask)):
                        filename = f"{split}_{language}_{side}{suffix}.bin"
                        array.tofile(staging / filename)
                        manifest["files"][filename] = {
                            "rows": len(pairs),
                            "sha256": file_digest(staging / filename),
                        }
            manifest["pairs"][language] = {
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                **stats,
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class PreferenceCorpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != 1 or self.manifest.get("kind") != "preferences":
            raise ValueError("unsupported preference dataset format")
        tokenizer_path = self.path / "tokenizer.json"
        if file_digest(tokenizer_path) != self.manifest["tokenizer_sha256"]:
            raise ValueError("dataset tokenizer checksum does not match")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.context = self.manifest["context"]
        width = self.context + 1
        self.rows = {}
        for split in ("train", "validation"):
            for language in LANGUAGES:
                arrays = []
                for side in SIDES:
                    for suffix, dtype in (("", TOKEN_DTYPE), ("_mask", MASK_DTYPE)):
                        filename = f"{split}_{language}_{side}{suffix}.bin"
                        metadata = self.manifest["files"][filename]
                        path = self.path / filename
                        if file_digest(path) != metadata["sha256"]:
                            raise ValueError(f"dataset checksum does not match: {filename}")
                        arrays.append(
                            np.memmap(path, mode="r", dtype=dtype, shape=(metadata["rows"], width))
                        )
                self.rows[split, language] = arrays
        self.fingerprint = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()

    def size(self, split):
        return sum(len(self.rows[split, language][0]) for language in LANGUAGES)

    def pick(self, split, start, count, seed):
        """Pairs start.. of an endless pass over a split, epoch by epoch in a seeded order."""
        sizes = [len(self.rows[split, language][0]) for language in LANGUAGES]
        picked = []
        for row in range(start, start + count):
            epoch, position = divmod(row, sum(sizes))
            cached = getattr(self, "_orders", {}).get(split)
            if cached is None or cached[0] != epoch:
                order = np.random.default_rng([seed, epoch]).permutation(sum(sizes))
                self._orders = {**getattr(self, "_orders", {}), split: (epoch, order)}
            index = int(self._orders[split][1][position])
            for language, size in zip(LANGUAGES, sizes):
                if index < size:
                    picked.append((language, index))
                    break
                index -= size
        return picked

    def load(self, split, picked, device):
        """Inputs and targets of the chosen replies, then of the rejected ones.

        Targets outside the final replies are -100, which cross-entropy ignores.
        """
        rows = []
        for side in range(len(SIDES)):
            for language, index in picked:
                arrays = self.rows[split, language]
                rows.append((arrays[2 * side][index], arrays[2 * side + 1][index]))
        tokens = torch.from_numpy(np.stack([np.asarray(tokens, dtype=np.int64) for tokens, _ in rows]))
        learn = torch.from_numpy(np.stack([np.asarray(mask, dtype=bool) for _, mask in rows]))
        targets = tokens[:, 1:].masked_fill(~learn[:, 1:], -100)
        return tokens[:, :-1].to(device), targets.to(device)


def reply_scores(model, inputs, targets, length_normalized):
    """Each row's log-probability of its reply, summed or averaged over its tokens."""
    logits = model(inputs)
    likelihoods = -F.cross_entropy(
        logits.flatten(0, 1).float(), targets.flatten(), reduction="none"
    ).view(targets.shape)
    scores = likelihoods.sum(-1)
    if length_normalized:
        scores = scores / (targets != -100).sum(-1).clamp(min=1)
    return scores


def preference_loss(policy, reference, inputs, targets, config):
    """The DPO loss of a batch of pairs, how often the policy prefers the chosen reply, and the margin."""
    scores = reply_scores(policy, inputs, targets, config.length_normalized)
    with torch.no_grad():
        anchors = reply_scores(reference, inputs, targets, config.length_normalized)
    pairs = len(scores) // 2
    margins = (scores[:pairs] - scores[pairs:]) - (anchors[:pairs] - anchors[pairs:])
    loss = -F.logsigmoid(config.beta * margins).mean()
    return loss, (margins > 0).float().mean(), margins.mean()


def rate(config, step):
    """Linear warmup, then a linear decay to zero, as Tülu 3 trains DPO."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    remaining = (config.steps - step) / (config.steps - config.warmup_steps)
    return config.learning_rate * remaining


@torch.no_grad()
def validate(policy, reference, corpus, config, device, precision):
    was_training = policy.training
    policy.eval()
    totals, pairs = [0.0, 0.0, 0.0], corpus.size("validation")
    try:
        for start in range(0, pairs, config.batch_size):
            picked = corpus.pick("validation", start, min(config.batch_size, pairs - start), config.seed)
            inputs, targets = corpus.load("validation", picked, device)
            with autocast(device, precision):
                values = preference_loss(policy, reference, inputs, targets, config)
            for index, value in enumerate(values):
                totals[index] += value.item() * len(picked)
    finally:
        policy.train(was_training)
    return dict(zip(("loss", "accuracy", "margin"), (total / pairs for total in totals)))


def train_preferences(data, output, initialize_from, config, device, precision="auto"):
    """Train a policy from a checkpoint on preference pairs; save it as training saves models."""
    corpus = PreferenceCorpus(data)
    precision = select_precision(device, precision)
    output = Path(output)
    if (output / "last.pt").exists():
        raise ValueError("a checkpoint already exists; select a new output directory")
    source = load_checkpoint(initialize_from)
    if Tokenizer.from_str(source["tokenizer"]).to_str() != corpus.tokenizer.to_str():
        raise ValueError("the preference data needs the checkpoint's tokenizer")
    torch.manual_seed(config.seed)
    policy, _, _ = load_model(initialize_from, device)
    if policy.config.context < corpus.context:
        raise ValueError("the pairs do not fit the model's context")
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=0.0, fused=device.type == "cuda"
    )
    log = []
    validation = validate(policy, reference, corpus, config, device, precision)
    log.append({"step": 0, "validation": validation})
    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = rate(config, step)
        optimizer.zero_grad(set_to_none=True)
        totals = [0.0, 0.0, 0.0]
        for micro in range(config.accumulation_steps):
            row = (step * config.accumulation_steps + micro) * config.batch_size
            inputs, targets = corpus.load("train", corpus.pick("train", row, config.batch_size, config.seed), device)
            with autocast(device, precision):
                loss, accuracy, margin = preference_loss(policy, reference, inputs, targets, config)
            (loss / config.accumulation_steps).backward()
            for index, value in enumerate((loss, accuracy, margin)):
                totals[index] += value.item() / config.accumulation_steps
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        optimizer.step()
        done = step + 1
        if done % config.log_every == 0 or done == config.steps:
            log.append({"step": done, **dict(zip(("loss", "accuracy", "margin"), totals)), "learning_rate": rate(config, step)})
        if done % config.eval_every == 0 or done == config.steps:
            validation = validate(policy, reference, corpus, config, device, precision)
            log.append({"step": done, "validation": validation})
    payload = {
        "version": 1,
        "model_config": source["model_config"],
        "model": policy.state_dict(),
        "tokenizer": source["tokenizer"],
        "preference_config": asdict(config),
        "dataset_fingerprint": corpus.fingerprint,
        "step": config.steps,
        "validation": validation,
        "precision": precision,
        "torch_version": str(torch.__version__),
        "initialization": {
            "checkpoint_sha256": file_digest(initialize_from),
            "source_step": source.get("step"),
            "method": "dpo",
        },
    }
    atomic_save(payload, output / "last.pt")
    (output / "log.jsonl").write_text("".join(json.dumps(line) + "\n" for line in log), encoding="utf-8")
    return {"step": config.steps, "validation": validation, "initial_validation": log[0]["validation"]}
