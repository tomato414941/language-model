"""Evaluate native checkpoints with the published LM Evaluation Harness tasks.

Besides the harness's metrics, each task with a correct continuation reports
its bits per byte, averaged over the questions as the accuracy is: the correct
choice of a choice task, or the reference answer of a generation task scored
after the same prompt. It moves with every gain in the answer's probability,
so small models show progress before their answers change; see DataDecide
(arXiv:2504.11393) and Signal and Noise (arXiv:2508.13144).
"""

import json
import math
import os
import statistics
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    import datasets
    from tqdm import tqdm

    from lm_eval import evaluator, utils
    from lm_eval.api.metrics import (
        aggregate_subtask_metrics,
        mean,
        pooled_sample_stderr,
        unweighted_mean_stderr,
    )
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
except ImportError as error:
    raise ImportError("Install evaluation dependencies with --extra eval") from error

from .conversation import chat_prompt, reply_ends, require_markers
from .data import BOS, EOS
from .training import autocast

# The standard tasks, each with the language it asks in and the accuracy read
# from it, written "metric,filter" when the task filters its answers. OLMES
# (arXiv:2406.08446) scores HellaSwag, ARC-Easy, PIQA and MMLU's cloze form by
# each choice's loglikelihood per character, the harness's acc_norm; KUCI and
# JMMLU's cloze form are scored the same way, and so is ARC-Challenge, which
# OLMES scores by PMI that the harness's task lacks. The rest use plain
# accuracy, as OLMo scored SciQ, or the harness's own metric. The easier
# tasks from SciQ to KUCI separate small models; the others stay near chance
# until a model grows enough to learn them. Add tasks or revise these choices
# freely, but not to favor particular models.
STANDARD_TASKS = {
    "hellaswag": ("en", "acc_norm"),
    "arc_easy": ("en", "acc_norm"),
    "ja_leaderboard_jcommonsenseqa": ("ja", "acc"),
    "sciq": ("en", "acc"),
    "piqa": ("en", "acc_norm"),
    "lambada_openai": ("en", "acc"),
    "blimp": ("en", "acc"),
    "kuci": ("ja", "acc_norm"),
    "ja_leaderboard_xwinograd": ("ja", "acc"),
    "ja_leaderboard_jnli": ("ja", "acc"),
    # Knowledge, reasoning, mathematics, programming and reading, which a
    # general model needs in both languages.
    "mmlu_continuation": ("en", "acc_norm"),
    "jmmlu": ("ja", "acc_norm"),
    "arc_challenge": ("en", "acc_norm"),
    "gsm8k": ("en", "exact_match,strict-match"),
    "ja_leaderboard_mgsm": ("ja", "acc"),
    "humaneval": ("en", "pass@1,create_test"),
    "mbpp": ("en", "pass_at_1"),
    "jhumaneval": ("ja", "pass@1,create_test"),
    "ja_leaderboard_jsquad": ("ja", "exact_match,whitespaces"),
}
DEFAULT_TASKS = tuple(STANDARD_TASKS)
# Definitions of published tasks that the harness lacks, in its own format.
TASK_DEFINITIONS = Path(__file__).with_name("harness_tasks")
# The continuation that answers each generation task's question correctly,
# from its reference answer. MGSM publishes no reference solutions for its
# test questions.
REFERENCE_ANSWERS = {
    "gsm8k": lambda doc: " " + doc["answer"],
    "humaneval": lambda doc: doc["canonical_solution"],
    "mbpp": lambda doc: doc["code"],
    "jhumaneval": lambda doc: doc["canonical_solution"],
    "ja_leaderboard_jsquad": lambda doc: "\n" + doc["answers"]["text"][0],
}
# These tasks ask in the conversation format that the API serves.
CONVERSATION_TASKS = ("ifeval", "m_ifeval_ja")
RESULT_SECTIONS = (
    "results",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
)
JGLUE_URL = (
    "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/"
    "datasets/{name}-v1.2/{split}-v1.2.json"
)
JGLUE_TASKS = {
    "ja_leaderboard_jcommonsenseqa": "jcommonsenseqa",
    "ja_leaderboard_jnli": "jnli",
    "ja_leaderboard_jsquad": "jsquad",
}
# JGLUE's loader numbered JNLI's labels in this order, which the task's choices follow.
JNLI_LABELS = ("entailment", "contradiction", "neutral")
ANSWER_BITS = "gold_bits_per_byte"


def number_jnli_labels(dataset):
    def number(doc):
        return {"label": JNLI_LABELS.index(doc["label"])}

    return dataset.map(number)


def squad_questions(dataset):
    """One record per question from SQuAD's articles, as JGLUE's loader gave JSQuAD."""
    return datasets.Dataset.from_list(
        [
            {
                "id": question["id"],
                "title": article["title"],
                "context": paragraph["context"],
                "question": question["question"],
                "answers": {
                    key: [answer[key] for answer in question["answers"]]
                    for key in ("text", "answer_start")
                },
                "is_impossible": question["is_impossible"],
            }
            for article in dataset
            for paragraph in article["paragraphs"]
            for question in paragraph["qas"]
        ]
    )


def jglue_task(manager, name):
    """Read JGLUE's original JSON directly: datasets 4+ cannot run its loader script."""
    path = manager.task_index[name].yaml_path
    config = load_yaml(path, resolve_func=True)
    # Keep the published prompts, validation split, examples and metrics.
    config.update(
        dataset_path="json",
        dataset_name=None,
        dataset_kwargs={
            "data_files": {
                split: JGLUE_URL.format(name=JGLUE_TASKS[name], split=file)
                for split, file in (("train", "train"), ("validation", "valid"))
            }
        },
    )
    if name == "ja_leaderboard_jnli":
        config["process_docs"] = number_jnli_labels
    if name == "ja_leaderboard_jsquad":
        # JSQuAD's files keep SQuAD's articles under "data".
        config["dataset_kwargs"]["field"] = "data"
        config["process_docs"] = squad_questions
    config["metadata"]["config_source"] = str(path)
    return ConfigurableTask(config=config)


def bits_per_byte(context, continuation, loglikelihood):
    # The harness scores the context's trailing spaces with the continuation.
    scored = context[len(context.rstrip()) :] + continuation
    return -loglikelihood / (math.log(2) * len(scored.encode("utf-8")))


def answer_bits(task, sample):
    """Bits per byte of the continuation that the harness counts as correct."""
    if task.OUTPUT_TYPE == "loglikelihood":
        gold = 0
    elif task.OUTPUT_TYPE == "multiple_choice":
        doc = sample["doc"]
        # The harness finds the correct choice this way when it scores the choices.
        gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
        if isinstance(gold, str):
            gold = task.doc_to_choice(doc).index(gold)
        if isinstance(gold, list):
            raise ValueError(f"{task.task_name} needs one correct choice per question")
    else:
        return None
    context, continuation = sample["arguments"][gold]
    return bits_per_byte(context, continuation, sample["filtered_resps"][gold][0])


def score_references(backend, samples):
    """Each generation task's reference answers' bits per byte, in the order of its questions.

    The reference answer continues the very prompt that the task asked.
    """
    scored = {}
    for name, rows in samples.items():
        if name not in REFERENCE_ANSWERS:
            continue
        # A task that filters its answers in several ways records each question once per filter.
        questions = {}
        for sample in rows:
            questions.setdefault(sample["doc_id"], sample)
        pairs = [
            (questions[doc]["arguments"][0][0], REFERENCE_ANSWERS[name](questions[doc]["doc"]))
            for doc in sorted(questions)
        ]
        likelihoods = backend.loglikelihood(
            [SimpleNamespace(args=pair) for pair in pairs], disable_tqdm=True
        )
        scored[name] = [
            bits_per_byte(context, continuation, loglikelihood)
            for (context, continuation), (loglikelihood, _) in zip(pairs, likelihoods)
        ]
    return scored


def task_reading(name):
    """The language a task or group asks in and the accuracy read from it.

    A task outside the standard ones is read by plain accuracy, as "other".
    """
    return STANDARD_TASKS.get(name, ("other", "acc"))


def result_keys(metric):
    """The harness's keys for a metric's value and standard error, which name its filter."""
    name, _, kept = metric.partition(",")
    return f"{name},{kept or 'none'}", f"{name}_stderr,{kept or 'none'}"


def record_questions(result, tasks, groups, references):
    """Record each question's scores and answer bits; report each task's and group's bits.

    `references` holds the generation tasks' reference answers' bits by question.
    """
    results, items = result["results"], {}
    for name, samples in result.pop("samples").items():
        task = tasks[name]
        filtered = {}
        for sample in sorted(samples, key=lambda sample: sample["doc_id"]):
            filtered.setdefault(sample["filter"], []).append(sample)
        # Kept per question so that models can be compared on the same
        # questions, with every score the task averages, so that a revised
        # choice of accuracy still reads this evaluation.
        record = {"doc_id": [sample["doc_id"] for sample in next(iter(filtered.values()))]}
        record["correct"] = {
            metric if kept == "none" else f"{metric},{kept}": [sample[metric] for sample in rows]
            for kept, rows in filtered.items()
            for metric in rows[0]["metrics"]
            if task.aggregation().get(metric) is mean
        }
        bits = references.get(name) or [
            answer_bits(task, sample) for sample in filtered.get("none", [])
        ]
        if bits and bits[0] is not None:
            results[name][f"{ANSWER_BITS},none"] = statistics.fmean(bits)
            results[name][f"{ANSWER_BITS}_stderr,none"] = (
                statistics.stdev(bits) / math.sqrt(len(bits)) if len(bits) > 1 else "N/A"
            )
            result["higher_is_better"][name][ANSWER_BITS] = False
            record[ANSWER_BITS] = [round(value, 6) for value in bits]
        items[name] = record
    for name, group in groups.items():
        members = [task.task_name for task in group.get_all_tasks()]
        weighting = [
            metric.weight_by_size
            for metric in group.aggregate_metric_list or []
            if metric.metric == "acc"
        ]
        if name not in results or not weighting or any(m not in items for m in members):
            continue
        # Combine the tasks' other scores as the group combines their accuracy.
        sizes = [len(items[member]["doc_id"]) for member in members]
        for metric in [*items[members[0]]["correct"], ANSWER_BITS]:
            value, error = result_keys(metric)
            if value in results[name] or any(value not in results[m] for m in members):
                continue
            values = [results[member][value] for member in members]
            errors = [results[member][error] for member in members]
            results[name][value] = aggregate_subtask_metrics(values, sizes, weighting[0])
            results[name][error] = (
                "N/A"
                if "N/A" in errors
                else pooled_sample_stderr(errors, sizes)
                if weighting[0]
                else unweighted_mean_stderr(errors)
            )
    return items


def by_language(result, names):
    """Each task's standard accuracy and correct answer's bits per byte, by language."""
    summary = {}
    for name in names:
        row, (language, metric) = result["results"][name], task_reading(name)
        value, error = result_keys(metric)
        if value not in row:  # Instruction following reports other metrics.
            continue
        summary.setdefault(language, {})[name] = {
            "metric": metric,
            "accuracy": row[value],
            "stderr": row[error],
            ANSWER_BITS: row.get(f"{ANSWER_BITS},none"),
        }
    return summary


class NativeLM(TemplateLM):
    """Expose model inference; task prompts and metrics belong to the harness."""

    def __init__(
        self, model, tokenizer, device, precision="float32", batch_tokens=4096
    ):
        super().__init__()
        if type(batch_tokens) is not int or batch_tokens <= 0:
            raise ValueError("batch_tokens must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.precision = precision
        self.batch_tokens = batch_tokens
        if tokenizer.token_to_id(BOS) is None or tokenizer.token_to_id(EOS) is None:
            raise ValueError("the tokenizer needs BOS and EOS tokens")

    @property
    def eot_token_id(self):
        return self.tokenizer.token_to_id(EOS)

    @property
    def prefix_token_id(self):
        return self.tokenizer.token_to_id(BOS)

    @property
    def tokenizer_name(self):
        return "native-conversation"

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        if not add_generation_prompt:
            raise ValueError(
                "the conversation format does not continue a started reply"
            )
        require_markers(self.tokenizer)
        return chat_prompt(chat_history)

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        tokens = self.tokenizer.encode(string, add_special_tokens=False).ids
        if add_special_tokens is not False and (
            not tokens or tokens[0] != self.prefix_token_id
        ):
            tokens.insert(0, self.prefix_token_id)
        return tokens

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        prepared = []
        for index, (key, context, continuation) in enumerate(requests):
            if not context or not 0 < len(continuation) <= self.model.config.context:
                raise ValueError("each continuation must fit within the context length")
            tokens = (context + continuation)[-(self.model.config.context + 1) :]
            prepared.append((index, key, tokens, len(continuation)))
        prepared.sort(key=lambda request: -len(request[2]))
        results = [None] * len(requests)
        was_training = self.model.training
        self.model.eval()
        # Long evaluations write to log files, so a line every half minute is enough.
        progress = tqdm(
            total=len(requests),
            desc="Scoring requests",
            disable=disable_tqdm,
            mininterval=30,
        )
        try:
            start = 0
            while start < len(prepared):
                width = len(prepared[start][2]) - 1
                batch = prepared[start : start + max(1, self.batch_tokens // width)]
                start += len(batch)
                inputs = torch.full(
                    (len(batch), width), self.eot_token_id, dtype=torch.long
                )
                for row, (_, _, tokens, _) in enumerate(batch):
                    inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
                with autocast(self.device, self.precision):
                    logits = self.model(inputs.to(self.device))
                probabilities = logits.float().log_softmax(-1)
                for row, (index, key, tokens, length) in enumerate(batch):
                    end = len(tokens) - 1
                    targets = torch.tensor(tokens[-length:], device=self.device)
                    scores = probabilities[row, end - length : end]
                    result = (
                        scores.gather(-1, targets[:, None]).sum().item(),
                        bool((scores.argmax(-1) == targets).all().item()),
                    )
                    results[index] = result
                    if key is not None:
                        self.cache_hook.add_partial("loglikelihood", key, result)
                progress.update(len(batch))
        finally:
            progress.close()
            self.model.train(was_training)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            windows = utils.get_rolling_token_windows(
                self.tok_encode(request.args[0], add_special_tokens=False),
                prefix_token=self.prefix_token_id,
                max_seq_len=self.model.config.context,
                context_len=1,
            )
            scores = self._loglikelihood_tokens(
                [(None, *utils.make_disjoint_window(window)) for window in windows],
                disable_tqdm=True,
            )
            results.append(sum(score for score, _ in scores))
        return results

    def stopped(self, tokens, stops):
        """Whether a continuation has just reached one of its stop sequences."""
        if not stops:
            return False
        # A sequence of n characters spans at most 4n bytes, so it begins
        # within the last 4n tokens before the one that completed it.
        window = 4 * max(map(len, stops)) + 1
        text = self.tokenizer.decode(tokens[-window:], skip_special_tokens=True)
        return any(stop in text for stop in stops)

    def generate_until(self, requests, disable_tqdm=False):
        prompts, limits, stops = [], [], []
        for request in requests:
            prompt, options = request.args
            if options.get("do_sample") or options.get("temperature", 0) != 0:
                raise ValueError("evaluation generation uses greedy decoding")
            until = options.get("until", [])
            limit = options.get("max_gen_toks", 256)
            tokens = self.tok_encode(prompt)
            # Keep the end of a prompt that leaves too little room to answer,
            # as the harness's own backends do.
            room = self.model.config.context - limit + 1
            prompts.append(tokens[-room:] if 0 < room < len(tokens) else tokens)
            limits.append(limit)
            stops.append([until] if isinstance(until, str) else until)
        # Decode prompts with the same limit together, longest first, within batch_tokens.
        order = sorted(
            range(len(requests)),
            key=lambda index: (-limits[index], -len(prompts[index])),
        )
        results = [None] * len(requests)
        start = 0
        while start < len(order):
            first = order[start]
            rows = max(1, self.batch_tokens // (len(prompts[first]) + limits[first]))
            batch = [
                index
                for index in order[start : start + rows]
                if limits[index] == limits[first]
            ]
            start += len(batch)
            with autocast(self.device, self.precision):
                continuations = self.model.continue_greedily(
                    [prompts[index] for index in batch],
                    limits[first],
                    reply_ends(self.tokenizer),
                    banned=[self.prefix_token_id],
                    finished=lambda row, tokens: self.stopped(tokens, stops[batch[row]]),
                )
            for index, continuation in zip(batch, continuations):
                text = self.tokenizer.decode(continuation, skip_special_tokens=True)
                endings = [text.index(stop) for stop in stops[index] if stop in text]
                if endings:
                    text = text[: min(endings)]
                results[index] = text
                self.cache_hook.add_partial(
                    "generate_until", requests[index].args, text
                )
        return results


def evaluate_benchmarks(
    model,
    tokenizer,
    device,
    precision="float32",
    batch_tokens=4096,
    tasks=None,
    limit=None,
):
    """Use the harness's task definitions, default few-shot counts and metrics."""
    return run_benchmarks(
        NativeLM(model, tokenizer, device, precision, batch_tokens), tasks, limit
    )


def evaluate_hub_benchmarks(
    repository, revision, device, precision="float32", batch_size=32, tasks=None, limit=None
):
    """Evaluate a model from the Hugging Face Hub on the same tasks and metrics.

    The harness's own Hugging Face backend runs the model, as published
    evaluations of it do.
    """
    from lm_eval.models.huggingface import HFLM

    backend = HFLM(
        pretrained=repository,
        revision=revision,
        device=str(device),
        dtype=precision,
        batch_size=batch_size,
    )
    result = run_benchmarks(backend, tasks, limit)
    result["model"] = {
        "repository": repository,
        "revision": revision,
        "parameters": sum(parameter.numel() for parameter in backend.model.parameters()),
        "precision": precision,
    }
    return result


def run_benchmarks(backend, tasks=None, limit=None):
    """Ask a harness backend the standard tasks and record each question's results."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    manager = TaskManager(include_path=str(TASK_DEFINITIONS))
    selected = list(DEFAULT_TASKS if tasks is None else tasks)
    if not selected:
        raise ValueError("select at least one evaluation task")
    for name in selected:
        if name not in manager.all_subtasks and name not in manager.all_groups:
            raise ValueError(f"unknown harness task: {name}")
    # The programming tasks run the model's programs against their tests,
    # which the code_eval metric makes its caller allow, as the harness does.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # Keep the loaded tasks: their records tell which continuation is correct.
    loaded = manager.load(
        [jglue_task(manager, name) if name in JGLUE_TASKS else name for name in selected]
    )
    chosen = {
        name: loaded["groups"][name] if name in loaded["groups"] else loaded["tasks"][name]
        for name in selected
    }
    result, items = None, {}
    for conversation in (False, True):
        group = [
            task
            for name, task in chosen.items()
            if (name in CONVERSATION_TASKS) == conversation
        ]
        if not group:
            continue
        part = evaluator.simple_evaluate(
            model=backend,
            tasks=group,
            task_manager=manager,
            limit=limit,
            log_samples=True,
            apply_chat_template=conversation,
            confirm_run_unsafe_code=True,
        )
        references = score_references(backend, part["samples"])
        items.update(record_questions(part, loaded["tasks"], loaded["groups"], references))
        if result is None:
            result = part
        else:
            for section in RESULT_SECTIONS:
                result[section].update(part[section])
    result["items"] = items
    result["by_language"] = by_language(result, chosen)
    result["conversation_format_tasks"] = [
        name for name in chosen if name in CONVERSATION_TASKS
    ]
    result["framework"] = {
        "name": "lm-evaluation-harness",
        "version": version("lm_eval"),
    }
    result["evaluation_scope"] = "full" if limit is None else "smoke"
    return json.loads(json.dumps(result, default=utils.handle_non_serializable))
