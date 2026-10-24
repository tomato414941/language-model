"""Reinforcement learning from verifiable rewards on prompts whose answers can be checked.

For each prompt the policy samples a group of replies, reasoning first, and a
verifier scores each: a math answer against the reference, or code against
its tests. A reply that runs past the token limit or never closes its
reasoning scores nothing. Each reply's advantage is its reward minus its
group's mean, without dividing by the group's spread, and the loss sums the
advantage-weighted log-probabilities of the reply tokens over a constant,
so neither the reply's length nor its group's spread biases the update
(Dr. GRPO, Liu et al., 2025). Groups whose replies all score alike teach
nothing and are left out, as DAPO samples (Yu et al., 2025). The policy
being trained samples each batch, so the update needs no importance ratio,
clipping or reference model.

Prompts arrive as JSON lines with the conversation before the reply
("messages") and how to check it ("verifier"): {"type": "math", "answer":
"42"} or {"type": "code", "tests": "assert f(1) == 2"}.
"""

import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import torch
from torch.nn import functional as F

from .conversation import THINK_END, TURN_END, chat_prompt, require_markers
from .data import BOS, file_digest
from .instructions import normalize_turn
from .training import atomic_save, autocast, load_checkpoint, load_model, select_precision

# Seconds a reply's program may run against its tests, and the memory and
# file size it may use.
CODE_SECONDS = 10
CODE_BYTES = 2 * 1024**3
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ReinforcementConfig:
    steps: int = 200
    prompts_per_step: int = 32
    group_size: int = 8
    max_reply_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    learning_rate: float = 1e-6
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    # Replies per forward pass of the loss, which bounds its memory.
    micro_batch: int = 16
    reasoning: bool = True
    log_every: int = 1
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "prompts_per_step", "max_reply_tokens", "micro_batch", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be an integer of at least 2, to compare replies")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be a nonnegative integer smaller than steps")
        for name in ("temperature", "learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.top_p) not in (int, float) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be within (0, 1]")
        if type(self.reasoning) is not bool:
            raise ValueError("reasoning must be true or false")


def read_config(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"reinforcement"}:
        raise ValueError("the configuration holds one reinforcement section")
    return ReinforcementConfig(**config["reinforcement"])


def read_prompts(path):
    """Return the prompts of a JSON lines file, each its turns before the reply and its verifier."""
    prompts = []
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            turns = [normalize_turn(message) for message in item.get("messages") or []]
            verifier = item.get("verifier")
            if not turns or not all(turns) or turns[-1]["role"] != "user":
                raise ValueError(f"line {number} needs turns that end with the user's")
            if not isinstance(verifier, dict) or not (
                (verifier.get("type") == "math" and isinstance(verifier.get("answer"), str))
                or (verifier.get("type") == "code" and isinstance(verifier.get("tests"), str))
            ):
                raise ValueError(f"line {number} needs a math answer or code tests to check against")
            prompts.append({"messages": turns, "verifier": verifier})
    if not prompts:
        raise ValueError("the prompt file holds no prompts")
    return prompts


def number(text):
    """The value a number's text spells, or None."""
    try:
        return Fraction(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def final_answer(text):
    """The reply's answer: what its last \\boxed{} holds, or else its last number."""
    boxed = text.rfind("\\boxed{")
    if boxed >= 0:
        depth, start = 1, boxed + len("\\boxed{")
        for index in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            if depth == 0:
                return text[start:index].strip()
    found = NUMBER.findall(text)
    return found[-1] if found else None


def math_reward(answer, reference):
    """1 if the answer equals the reference, as numbers when both are numbers."""
    found = final_answer(answer)
    if found is None:
        return 0.0
    value, expected = number(found), number(reference.strip())
    if value is not None and expected is not None:
        return float(value == expected)
    return float(found.replace(" ", "") == reference.replace(" ", ""))


def confine():
    """Limit a reply's program before it runs: its time, memory and file sizes,
    and, when started by root, the rights of the nobody user."""
    resource.setrlimit(resource.RLIMIT_CPU, (CODE_SECONDS, CODE_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (CODE_BYTES, CODE_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)


def code_reward(answer, tests):
    """1 if the reply's program, its last Python block or else the whole answer, passes the tests.

    The program runs in a separate interpreter in an empty directory, confined
    in time, memory and file sizes; a GPU instance that runs it is disposable.
    """
    blocks = CODE_BLOCK.findall(answer)
    program = (blocks[-1] if blocks else answer) + "\n\n" + tests + "\n"
    with tempfile.TemporaryDirectory() as directory:
        os.chmod(directory, 0o777)
        script = Path(directory) / "check.py"
        script.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=directory, capture_output=True,
                timeout=2 * CODE_SECONDS, preexec_fn=confine, env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return float(completed.returncode == 0)


def reward(text, verifier, reasoning, ended):
    """Score a reply's text, which reasons first if asked; one that never ends scores 0."""
    if not ended:
        return 0.0
    if reasoning:
        if THINK_END not in text:
            return 0.0
        text = text.split(THINK_END, 1)[1]
    text = text.split(TURN_END, 1)[0]
    if verifier["type"] == "math":
        return math_reward(text, verifier["answer"])
    return code_reward(text, verifier["tests"])


def advantages(rewards, group_size):
    """Each reward minus its group's mean, and which groups' replies did not all score alike."""
    grouped = torch.tensor(rewards, dtype=torch.float32).view(-1, group_size)
    centered = grouped - grouped.mean(-1, keepdim=True)
    return centered.flatten(), (grouped.max(-1).values > grouped.min(-1).values).repeat_interleave(group_size)


def policy_loss(policy, sequences, prompt_lengths, weights, config, device, precision):
    """Backpropagate the advantage-weighted log-probabilities of the replies, in micro-batches.

    sequences hold each prompt and its reply, with the reply's end token when
    it ended; the sum is divided by a constant, the rows times the token limit.
    """
    scale = len(sequences) * config.max_reply_tokens
    total = 0.0
    for start in range(0, len(sequences), config.micro_batch):
        rows = sequences[start : start + config.micro_batch]
        width = max(map(len, rows))
        tokens = torch.zeros(len(rows), width, dtype=torch.long)
        targets = torch.full((len(rows), width - 1), -100, dtype=torch.long)
        for row, (sequence, prompt) in enumerate(zip(rows, prompt_lengths[start : start + config.micro_batch])):
            tokens[row, : len(sequence)] = torch.tensor(sequence)
            targets[row, prompt - 1 : len(sequence) - 1] = tokens[row, prompt : len(sequence)]
        tokens, targets = tokens.to(device), targets.to(device)
        with autocast(device, precision):
            logits = policy(tokens[:, :-1])
        likelihoods = -F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="none"
        ).view(targets.shape)
        loss = -(weights[start : start + len(rows), None].to(device) * likelihoods).sum() / scale
        loss.backward()
        total += loss.item()
    return total


def rate(config, step):
    """Linear warmup, then a constant rate."""
    return config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_steps))


def train_reinforcement(prompts_path, output, initialize_from, config, device, precision="auto"):
    """Train a policy from a checkpoint on verifiable prompts; save it as training saves models."""
    prompts = read_prompts(prompts_path)
    precision = select_precision(device, precision)
    output = Path(output)
    if (output / "last.pt").exists():
        raise ValueError("a checkpoint already exists; select a new output directory")
    source = load_checkpoint(initialize_from)
    torch.manual_seed(config.seed)
    policy, tokenizer, _ = load_model(initialize_from, device)
    require_markers(tokenizer)
    # A reply ends only with its turn's end, which training then learns.
    bos, turn_end = tokenizer.token_to_id(BOS), tokenizer.token_to_id(TURN_END)
    encoded = [
        [bos, *tokenizer.encode(chat_prompt(item["messages"], config.reasoning), add_special_tokens=False).ids]
        for item in prompts
    ]
    if max(map(len, encoded)) + config.max_reply_tokens > policy.config.context:
        raise ValueError("a prompt and its longest reply must fit the model's context")
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=0.0, fused=device.type == "cuda"
    )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    order, log = [], []
    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = rate(config, step)
        picked = []
        while len(picked) < config.prompts_per_step:
            if not order:
                order = list(range(len(prompts)))
                random.Random(config.seed + step).shuffle(order)
            picked.append(order.pop())
        starts = [encoded[index] for index in picked for _ in range(config.group_size)]
        policy.eval()
        with autocast(device, precision):
            replies = policy.continue_sequences(
                starts, config.max_reply_tokens, turn_end, banned=[bos],
                temperature=config.temperature, top_p=config.top_p, generator=generator,
            )
        ended = [len(reply) < config.max_reply_tokens for reply in replies]
        rewards = [
            reward(
                tokenizer.decode(reply, skip_special_tokens=False), prompts[index]["verifier"], config.reasoning, done
            )
            for reply, done, index in zip(replies, ended, (i for i in picked for _ in range(config.group_size)))
        ]
        centered, informative = advantages(rewards, config.group_size)
        kept = [row for row in range(len(replies)) if informative[row]]
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        if kept:
            sequences = [starts[row] + replies[row] + ([turn_end] if ended[row] else []) for row in kept]
            loss = policy_loss(
                policy, sequences, [len(starts[row]) for row in kept], centered[kept], config, device, precision
            )
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
            optimizer.step()
        done = step + 1
        if done % config.log_every == 0 or done == config.steps:
            log.append({
                "step": done,
                "reward": sum(rewards) / len(rewards),
                "informative_groups": len(kept) // config.group_size,
                "ended": sum(ended) / len(ended),
                "reply_tokens": sum(map(len, replies)) / len(replies),
                "loss": loss,
                "learning_rate": rate(config, step),
            })
    payload = {
        "version": 1,
        "model_config": source["model_config"],
        "model": policy.state_dict(),
        "tokenizer": source["tokenizer"],
        "reinforcement_config": asdict(config),
        "prompts_sha256": file_digest(prompts_path),
        "step": config.steps,
        "precision": precision,
        "torch_version": str(torch.__version__),
        "initialization": {
            "checkpoint_sha256": file_digest(initialize_from),
            "source_step": source.get("step"),
            "method": "grpo",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, output / "last.pt")
    (output / "log.jsonl").write_text("".join(json.dumps(line) + "\n" for line in log), encoding="utf-8")
    return {"step": config.steps, "final": log[-1]}
