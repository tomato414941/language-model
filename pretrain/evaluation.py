"""Evaluate native checkpoints with the published LM Evaluation Harness tasks.

Besides the harness's metrics, each choice and continuation task reports the
bits per byte of its correct continuation, averaged over the questions as the
accuracy is. It moves with every gain in the answer's probability, so small
models show progress before their answers change; see DataDecide
(arXiv:2504.11393) and Signal and Noise (arXiv:2508.13144).
"""

import json
import math
import statistics
from importlib.metadata import version

import torch

try:
    from tqdm import tqdm

    from lm_eval import evaluator, utils
    from lm_eval.api.metrics import (
        aggregate_subtask_metrics,
        pooled_sample_stderr,
        unweighted_mean_stderr,
    )
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
except ImportError as error:
    raise ImportError("Install evaluation dependencies with --extra eval") from error

from .conversation import chat_prompt
from .data import BOS, EOS
from .training import autocast

DEFAULT_TASKS = ("hellaswag", "arc_easy", "ja_leaderboard_jcommonsenseqa")
# These tasks ask in the conversation format that the API serves.
CONVERSATION_TASKS = ("ifeval",)
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
}
# JGLUE's loader numbered JNLI's labels in this order, which the task's choices follow.
JNLI_LABELS = ("entailment", "contradiction", "neutral")
ANSWER_BITS = "gold_bits_per_byte"


def number_jnli_labels(dataset):
    def number(doc):
        return {"label": JNLI_LABELS.index(doc["label"])}

    return dataset.map(number)


def jglue_task(manager, name):
    """Read JGLUE's original JSON directly: datasets 4+ cannot run its loader script."""
    path = manager.task_index[name].yaml_path
    config = load_yaml(path, resolve_func=True)
    # Keep the published prompts, validation split, three examples and metrics.
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
    config["metadata"]["config_source"] = str(path)
    return ConfigurableTask(config=config)


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
    # The harness scores the context's trailing spaces with the continuation.
    scored = context[len(context.rstrip()) :] + continuation
    loglikelihood = sample["filtered_resps"][gold][0]
    return -loglikelihood / (math.log(2) * len(scored.encode("utf-8")))


def add_answer_bits(result, tasks, groups):
    """Report each task's and group's answer bits; return the questions' records."""
    results, items = result["results"], {}
    for name, samples in result.pop("samples").items():
        samples = sorted(samples, key=lambda sample: sample["doc_id"])
        bits = [answer_bits(tasks[name], sample) for sample in samples]
        if not bits or bits[0] is None:
            continue
        results[name][f"{ANSWER_BITS},none"] = statistics.fmean(bits)
        results[name][f"{ANSWER_BITS}_stderr,none"] = (
            statistics.stdev(bits) / math.sqrt(len(bits)) if len(bits) > 1 else "N/A"
        )
        result["higher_is_better"][name][ANSWER_BITS] = False
        # Kept per question so that models can be compared on the same questions.
        items[name] = {
            "doc_id": [sample["doc_id"] for sample in samples],
            "correct": [sample["acc"] for sample in samples],
            ANSWER_BITS: [round(value, 6) for value in bits],
        }
    for name, group in groups.items():
        members = [task.task_name for task in group.get_all_tasks()]
        weighting = [
            metric.weight_by_size
            for metric in group.aggregate_metric_list or []
            if metric.metric == "acc"
        ]
        if name not in results or not weighting or any(m not in items for m in members):
            continue
        # Combine the tasks as the group combines their accuracy.
        values = [results[member][f"{ANSWER_BITS},none"] for member in members]
        errors = [results[member][f"{ANSWER_BITS}_stderr,none"] for member in members]
        sizes = [len(items[member]["doc_id"]) for member in members]
        results[name][f"{ANSWER_BITS},none"] = aggregate_subtask_metrics(
            values, sizes, weighting[0]
        )
        results[name][f"{ANSWER_BITS}_stderr,none"] = (
            "N/A"
            if "N/A" in errors
            else pooled_sample_stderr(errors, sizes)
            if weighting[0]
            else unweighted_mean_stderr(errors)
        )
    return items


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

    def generate_until(self, requests, disable_tqdm=False):
        prompts, limits, stops = [], [], []
        for request in requests:
            prompt, options = request.args
            if options.get("do_sample") or options.get("temperature", 0) != 0:
                raise ValueError("evaluation generation uses greedy decoding")
            until = options.get("until", [])
            prompts.append(self.tok_encode(prompt))
            limits.append(options.get("max_gen_toks", 256))
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
                    self.eot_token_id,
                    banned=[self.prefix_token_id],
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
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    manager = TaskManager()
    selected = list(DEFAULT_TASKS if tasks is None else tasks)
    if not selected:
        raise ValueError("select at least one evaluation task")
    for name in selected:
        if name not in manager.all_subtasks and name not in manager.all_groups:
            raise ValueError(f"unknown harness task: {name}")
    # Keep the loaded tasks: their records tell which continuation is correct.
    loaded = manager.load(
        [jglue_task(manager, name) if name in JGLUE_TASKS else name for name in selected]
    )
    chosen = {
        name: loaded["groups"][name] if name in loaded["groups"] else loaded["tasks"][name]
        for name in selected
    }
    backend = NativeLM(model, tokenizer, device, precision, batch_tokens)
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
        )
        items.update(add_answer_bits(part, loaded["tasks"], loaded["groups"]))
        if result is None:
            result = part
        else:
            for section in RESULT_SECTIONS:
                result[section].update(part[section])
    result["items"] = items
    result["conversation_format_tasks"] = [
        name for name in chosen if name in CONVERSATION_TASKS
    ]
    result["framework"] = {
        "name": "lm-evaluation-harness",
        "version": version("lm_eval"),
    }
    result["evaluation_scope"] = "full" if limit is None else "smoke"
    return json.loads(json.dumps(result, default=utils.handle_non_serializable))
