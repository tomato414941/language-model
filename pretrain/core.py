"""DCLM's CORE score (arXiv:2406.11794), measured the way nanochat measures it.

nanochat (MIT; nanochat/core_eval.py and scripts/base_eval.py at 92d63d4) asks the 22 tasks of its eval bundle, which
holds the questions of DCLM's core tasks with their few-shot counts, continuation delimiters and random baselines.
Each question is plain text after BOS, after examples drawn from the task's other questions. A choice is taken by the
lowest mean loss over its own tokens, and a language-modeling answer is right when greedy prediction gives every one of
its tokens. A task's accuracy is centered on its random baseline, and CORE is the mean of the 22 centered accuracies.
The bundle carries the scores of the GPT-2 models measured this way, and nanochat reports its models by this number.
"""

import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from jinja2 import Template

from .data import BOS
from .training import autocast

BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BUNDLE_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
# nanochat's seeds: the order a task's questions are asked in, and each question's examples.
ORDER_SEED = 1337
EXAMPLES_SEED = 1234
QUESTIONS_AT_ONCE = 1024
# nanochat's templates, as written, since they decide the exact text of each question.
CHOICES = Template("""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip())
SCHEMA = Template("""
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip())
LANGUAGE_MODELING = Template("""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip())


def fetch_bundle(directory):
    """Put nanochat's eval bundle in `directory`, unless it is there, and return the directory."""
    directory = Path(directory)
    if (directory / "core.yaml").exists():
        return directory
    with urllib.request.urlopen(BUNDLE_URL, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != BUNDLE_SHA256:
        raise ValueError("the eval bundle is not the one nanochat's CORE scores were measured on")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as staging:
        archive_path = Path(staging) / "eval_bundle.zip"
        archive_path.write_bytes(data)
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"the eval bundle holds an unsafe path: {name}")
            archive.extractall(staging)
        shutil.move(str(Path(staging) / "eval_bundle"), directory)
    return directory


def read_tasks(bundle):
    """The 22 tasks of core.yaml, each with its random baseline in percent from eval_meta_data.csv."""
    bundle = Path(bundle)
    with (bundle / "eval_meta_data.csv").open(encoding="utf-8") as table:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) for row in csv.DictReader(table)}
    tasks = yaml.safe_load((bundle / "core.yaml").read_text(encoding="utf-8"))["icl_tasks"]
    return [
        {
            "label": task["label"],
            "type": task["icl_task_type"],
            "path": bundle / "eval_data" / task["dataset_uri"],
            "shots": task["num_fewshot"][0],
            "delimiter": task.get("continuation_delimiter", " "),
            "random_baseline": baselines[task["label"]],
        }
        for task in tasks
    ]


def read_questions(path, limit=None):
    """A task's questions in nanochat's order, the first `limit` of them if given."""
    with Path(path).open(encoding="utf-8") as lines:
        questions = [json.loads(line.strip()) for line in lines]
    random.Random(ORDER_SEED).shuffle(questions)
    return questions if limit is None else questions[:limit]


def example_indices(index, count, shots):
    """The examples nanochat asks question `index` after: a sample of the other questions, from its own seed.

    It samples from the list of the other indices, and random.sample picks positions by the list's length alone,
    so the same positions in the range without `index` give the same questions without building the list.
    """
    if shots == 0:
        return []
    positions = random.Random(EXAMPLES_SEED + index).sample(range(count - 1), shots)
    return [position if position < index else position + 1 for position in positions]


def prompts(kind, item, delimiter, examples):
    """The texts nanochat asks a question in: one for each choice or context, or the context without and with
    the answer."""
    values = {"fewshot_examples": examples, "continuation_delimiter": delimiter, "item": item}
    if kind == "multiple_choice":
        return [CHOICES.render(choice=choice, **values) for choice in item["choices"]]
    if kind == "schema":
        return [SCHEMA.render(context=option, **values) for option in item["context_options"]]
    if kind == "language_modeling":
        without = LANGUAGE_MODELING.render(include_continuation=False, **values).strip()
        return [without, LANGUAGE_MODELING.render(include_continuation=True, **values)]
    raise ValueError(f"CORE has no task of type {kind}")


def common_length(sequences, from_end=False):
    shortest = min(len(sequence) for sequence in sequences)
    for offset in range(shortest):
        position = -1 - offset if from_end else offset
        if any(sequence[position] != sequences[0][position] for sequence in sequences):
            return offset
    return shortest


def answer_spans(kind, sequences):
    """The sequences to read and where each one's answer lies, [start, end), as nanochat finds them: after the
    tokens all choices share, before the tokens all contexts share, or after the context alone."""
    if kind == "multiple_choice":
        start = common_length(sequences)
        return [(sequence, start, len(sequence)) for sequence in sequences]
    if kind == "schema":
        shared = common_length(sequences, from_end=True)
        return [(sequence, len(sequence) - shared, len(sequence)) for sequence in sequences]
    without, whole = sequences
    if not (len(without) < len(whole) and whole[: len(without)] == without):
        raise ValueError("a context's tokens must begin the tokens of the context with its answer")
    return [(whole, len(without), len(whole))]


def fit_context(sequence, start, end, context):
    """Keep the last `context` tokens, as nanochat does for a model with a shorter context."""
    cut = max(0, len(sequence) - context)
    if start - cut < 1:
        raise ValueError("an answer longer than the model's context")
    return sequence[cut:], start - cut, end - cut


@torch.inference_mode()
def read_answers(model, requests, device, precision, batch_tokens, pad):
    """For each (sequence, start, end), the mean loss over the answer's tokens and whether greedy prediction gives
    all of them. Sequences are read longest first, padded at the end with `pad`, as many as fit `batch_tokens`."""
    order = sorted(range(len(requests)), key=lambda index: -len(requests[index][0]))
    results = [None] * len(requests)
    start = 0
    while start < len(order):
        width = len(requests[order[start]][0])
        batch = order[start : start + max(1, batch_tokens // width)]
        start += len(batch)
        inputs = torch.full((len(batch), width), pad, dtype=torch.long)
        for row, index in enumerate(batch):
            sequence = requests[index][0]
            inputs[row, : len(sequence)] = torch.tensor(sequence)
        inputs = inputs.to(device)
        with autocast(device, precision):
            logits = model(inputs)
        for row, index in enumerate(batch):
            _, first, end = requests[index]
            predicted = logits[row, first - 1 : end - 1].float()
            targets = inputs[row, first:end]
            loss = F.cross_entropy(predicted, targets, reduction="none").mean().item()
            results[index] = (loss, bool((predicted.argmax(-1) == targets).all().item()))
    return results


def ask_task(model, tokenizer, device, precision, task, batch_tokens, bos, limit=None):
    """Whether the model answers each of the task's questions correctly. The questions go in groups of
    QUESTIONS_AT_ONCE, so that ten examples before each of HellaSwag's ten thousand questions are not all held as
    tokens at once."""
    questions = read_questions(task["path"], limit)
    if len(questions) <= task["shots"]:
        raise ValueError(f"{task['label']} needs more than {task['shots']} questions, for the examples of each")
    correct = []
    for first in range(0, len(questions), QUESTIONS_AT_ONCE):
        asked = []
        for index in range(first, min(first + QUESTIONS_AT_ONCE, len(questions))):
            examples = [questions[other] for other in example_indices(index, len(questions), task["shots"])]
            asked.append(prompts(task["type"], questions[index], task["delimiter"], examples))
        encoded = iter(tokenizer.encode_batch([text for texts in asked for text in texts], add_special_tokens=False))
        requests, owners = [], []
        for offset, texts in enumerate(asked):
            sequences = [[bos, *next(encoded).ids] for _ in texts]
            for request in answer_spans(task["type"], sequences):
                requests.append(fit_context(*request, model.config.context))
                owners.append(offset)
        answers = [[] for _ in asked]
        for owner, result in zip(owners, read_answers(model, requests, device, precision, batch_tokens, bos)):
            answers[owner].append(result)
        for offset, results in enumerate(answers):
            if task["type"] == "language_modeling":
                correct.append(results[0][1])
            else:
                losses = [loss for loss, _ in results]
                correct.append(losses.index(min(losses)) == questions[first + offset]["gold"])
    return correct


def evaluate_core(model, tokenizer, device, precision, bundle, batch_tokens=16384, limit=None, bos=None):
    """Each task's accuracy and centered accuracy, and CORE, the mean of the centered accuracies. `bos` is the token
    that begins each question, the tokenizer's BOS unless given."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    bos = tokenizer.token_to_id(BOS) if bos is None else bos
    was_training = model.training
    model.eval()
    try:
        tasks = {}
        for task in read_tasks(bundle):
            correct = ask_task(model, tokenizer, device, precision, task, batch_tokens, bos, limit)
            accuracy = sum(correct) / len(correct)
            baseline = 0.01 * task["random_baseline"]
            tasks[task["label"]] = {
                "type": task["type"],
                "shots": task["shots"],
                "questions": len(correct),
                "accuracy": accuracy,
                "random_baseline": task["random_baseline"],
                "centered": (accuracy - baseline) / (1.0 - baseline),
                "correct": "".join("1" if answer else "0" for answer in correct),
            }
            print(json.dumps({"task": task["label"], "accuracy": accuracy, "centered": tasks[task["label"]]["centered"]}),
                  file=sys.stderr, flush=True)
    finally:
        model.train(was_training)
    return {
        "procedure": {"bundle": BUNDLE_URL, "sha256": BUNDLE_SHA256, "nanochat": "92d63d4", "limit": limit},
        "tasks": tasks,
        "core": sum(task["centered"] for task in tasks.values()) / len(tasks),
    }


class HubModel(torch.nn.Module):
    """A Hugging Face causal model read as the project's own: logits for tokens, and the context it reads."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(context=model.config.max_position_embeddings)

    def forward(self, tokens):
        return self.model(tokens).logits


def load_hub_model(repository, revision, device):
    """A Hub model, its tokenizer and the token nanochat begins its questions with: the model's BOS, which for
    GPT-2 is its end of text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repository, revision=revision, dtype=torch.float32)
    return HubModel(model).to(device), tokenizer.backend_tokenizer, tokenizer.bos_token_id
