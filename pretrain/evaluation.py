"""Evaluate native checkpoints with the published LM Evaluation Harness tasks."""

import json
from importlib.metadata import version

import torch

try:
    from lm_eval import evaluator, utils
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
JCOMMONSENSEQA_URL = (
    "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/"
    "datasets/jcommonsenseqa-v1.2/{split}-v1.2.json"
)


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
    def _loglikelihood_tokens(self, requests, **kwargs):
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
        finally:
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
                [(None, *utils.make_disjoint_window(window)) for window in windows]
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
    loaded = {}
    for name in selected:
        if name not in manager.all_subtasks:
            raise ValueError(f"unknown harness task: {name}")
        if name == "ja_leaderboard_jcommonsenseqa":
            path = manager.task_index[name].yaml_path
            config = load_yaml(path, resolve_func=True)
            # Read JGLUE's original JSON directly: datasets 4+ cannot run its loader script.
            # Keep the published prompts, validation split, three examples and metrics.
            config.update(
                dataset_path="json",
                dataset_name=None,
                dataset_kwargs={
                    "data_files": {
                        "train": JCOMMONSENSEQA_URL.format(split="train"),
                        "validation": JCOMMONSENSEQA_URL.format(split="valid"),
                    }
                },
            )
            config["metadata"]["config_source"] = str(path)
            loaded[name] = ConfigurableTask(config=config)
        else:
            loaded[name] = name
    backend = NativeLM(model, tokenizer, device, precision, batch_tokens)
    result = None
    for conversation in (False, True):
        group = [
            task
            for name, task in loaded.items()
            if (name in CONVERSATION_TASKS) == conversation
        ]
        if not group:
            continue
        part = evaluator.simple_evaluate(
            model=backend,
            tasks=group,
            task_manager=manager,
            limit=limit,
            log_samples=False,
            apply_chat_template=conversation,
        )
        if result is None:
            result = part
        else:
            for section in RESULT_SECTIONS:
                result[section].update(part[section])
    result["conversation_format_tasks"] = [
        name for name in loaded if name in CONVERSATION_TASKS
    ]
    result["framework"] = {
        "name": "lm-evaluation-harness",
        "version": version("lm_eval"),
    }
    result["evaluation_scope"] = "full" if limit is None else "smoke"
    return json.loads(json.dumps(result, default=utils.handle_non_serializable))
