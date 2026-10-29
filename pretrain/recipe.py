"""A reproducible bilingual corpus and training recipe, independent of its host."""

import gzip
import hashlib
import http.client
import io
import itertools
import json
import math
import multiprocessing
import multiprocessing.connection
import os
import random
import re
import shutil
import sqlite3
import string
import tempfile
import time
import tomllib
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .conversation import conversation_text
from .data import BOS, EOS, LANGUAGES, Corpus, file_digest, read_documents
from .instructions import BenchmarkFilter
from .model import LanguageModel
from .training import TrainConfig, read_config

SOURCE_FORMATS = {
    "parquet": ".parquet",
    "jsonl": ".jsonl",
    "jsonl.zst": ".jsonl.zst",
    "jsonl.gz": ".jsonl.gz",
}
FILTERS = ("equal", "minimum", "maximum", "any_minimum")
SOURCE_FIELDS = {
    "id",
    "language",
    "host",
    "repository",
    "revision",
    "prefix",
    "format",
    "license",
    "terms_url",
    "text_field",
    "template",
    "conversation_field",
    "exclude_files",
    "minimum_characters",
    "maximum_characters",
    *FILTERS,
}
PURPOSES = ("phase", "alternative", "evaluation", "pool")
# A pool is read from the sources but never trained on. A corpus that selects
# from it takes either a random sample or the documents a model learns most from.
SELECTIONS = ("random", "learnable")
TOKEN_PIECE_DTYPE = "<u2"
# GPT-4's pre-tokenization, which OLMo 2 and Llama 3 use: it keeps runs of
# whitespace and line breaks apart from words, so code's indentation and line
# endings become tokens of their own. DIGITS stands for how numbers split.
GPT4_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|DIGITS"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
PRETOKENIZERS = ("gpt2", "gpt4")
# Speakers as datasets name them, and the roles of the conversation format
# that instruction tuning also reads.
SPEAKERS = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "system": "system"}
# Each part reads this much beyond its share of a source, so the merge can drop
# documents another part or source already supplied without falling short.
PART_SLACK = 0.15


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_source(source):
    unknown = set(source) - SOURCE_FIELDS
    if unknown:
        raise ValueError(f"{source.get('id')} has unknown fields: {sorted(unknown)}")
    if source.get("language") not in LANGUAGES:
        raise ValueError("select a supported language for each source")
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValueError("pin each source to a full dataset commit")
    if not source.get("license") or not source.get("terms_url"):
        raise ValueError("each source needs its license and terms")
    if source.get("format", "parquet") not in SOURCE_FORMATS:
        raise ValueError(f"{source['id']} uses an unsupported file format")
    if not isinstance(source.get("prefix"), str) or not source.get("repository"):
        raise ValueError(f"{source['id']} needs a repository and file prefix")
    # Sources come from the Hugging Face Hub unless they name a GitLab server.
    if "host" in source and not re.fullmatch(r"https://[a-z0-9.-]+", source["host"]):
        raise ValueError(f"{source['id']} host must be a GitLab server's https address")
    if sum(name in source for name in ("text_field", "template", "conversation_field")) > 1:
        raise ValueError(f"{source['id']} reads its text one way: a field, a template or a conversation")
    if "template" in source and (
        not isinstance(source["template"], str) or not template_fields(source["template"])
    ):
        raise ValueError(f"{source['id']} template must name fields in braces")
    for name in FILTERS:
        conditions = source.get(name, {})
        if not isinstance(conditions, dict) or not all(
            isinstance(key, str) and key for key in conditions
        ):
            raise ValueError(f"{source['id']} {name} must map field names to values")
        if name != "equal" and not all(map(finite_number, conditions.values())):
            raise ValueError(f"{source['id']} {name} thresholds must be numbers")
    if "any_minimum" in source and not source["any_minimum"]:
        raise ValueError(f"{source['id']} any_minimum needs at least one field")
    for name in ("minimum_characters", "maximum_characters"):
        if name in source and (type(source[name]) is not int or source[name] <= 0):
            raise ValueError(f"{source['id']} {name} must be a positive integer")


def read_sources(recipe, base):
    sources = recipe["sources"]
    if isinstance(sources, str):
        catalog = base / sources
        sources = tomllib.loads(catalog.read_text(encoding="utf-8"))["sources"]
        base = catalog.parent
    resolved = {}
    for source in sources:
        source = dict(source)
        if source.get("id") in resolved:
            raise ValueError("each corpus source needs a unique identifier")
        if "exclude_files_from" in source:
            listed = tomllib.loads(
                (base / source.pop("exclude_files_from")).read_text(encoding="utf-8")
            )["files"]
            source["exclude_files"] = sorted(
                {*source.get("exclude_files", []), *listed}
            )
        validate_source(source)
        resolved[source["id"]] = source
    return resolved


def read_recipe(path):
    path = Path(path).resolve()
    recipe = tomllib.loads(path.read_text(encoding="utf-8"))
    if type(recipe.get("seed")) is not int or recipe["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    tokenizer, corpus = recipe["tokenizer"], recipe["corpus"]
    if (
        type(tokenizer.get("vocabulary_size")) is not int
        or not 258 <= tokenizer["vocabulary_size"] <= 65536
    ):
        raise ValueError("the vocabulary must fit in unsigned 16-bit token IDs")
    if tokenizer.get("pretokenizer", "gpt2") not in PRETOKENIZERS:
        raise ValueError(f"pretokenizer must be one of {PRETOKENIZERS}")
    if type(tokenizer.get("single_digits", False)) is not bool:
        raise ValueError("single_digits must be true or false")
    reserved = tokenizer.get("reserved_tokens", 0)
    if type(reserved) is not int or not 0 <= reserved <= tokenizer["vocabulary_size"] - 258:
        raise ValueError("reserved_tokens must leave room for the special tokens and bytes")
    if "reuse_sha256" in tokenizer:
        if not re.fullmatch(r"[0-9a-f]{64}", tokenizer["reuse_sha256"]) or not (
            tokenizer.get("reuse_from")
        ):
            raise ValueError("a reused tokenizer needs its file and SHA-256 digest")
    elif (
        type(tokenizer.get("bytes_per_language")) is not int
        or tokenizer["bytes_per_language"] <= 0
    ):
        raise ValueError("corpus and tokenizer budgets must be positive integers")
    # Sources are read in parts of about this many tokens, one per worker.
    corpus.setdefault("part_tokens", 250_000_000)
    for value in (
        corpus["validation_tokens_per_language"],
        corpus["minimum_characters"],
        corpus["part_tokens"],
    ):
        if type(value) is not int or value <= 0:
            raise ValueError("corpus and tokenizer budgets must be positive integers")
    if any(
        "validation_fraction" in settings
        and not 0 < settings["validation_fraction"] < 0.5
        for settings in (corpus, *recipe.get("corpora", []))
    ):
        raise ValueError("validation_fraction must be between zero and one half")
    if (
        type(corpus["maximum_characters"]) is not int
        or corpus["maximum_characters"] < corpus["minimum_characters"]
    ):
        raise ValueError("maximum_characters must cover minimum_characters")
    excluded = corpus.get("excluded_corpora", [])
    if not isinstance(excluded, list) or any(
        not isinstance(entry, dict)
        or set(entry) != {"path", "documents_sha256"}
        or not isinstance(entry["path"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(entry["documents_sha256"]))
        for entry in excluded
    ):
        raise ValueError(
            "each excluded corpus needs its path and the SHA-256 digest of its documents"
        )
    sources = read_sources(recipe, path.parent)
    corpora = recipe.get("corpora")
    if not corpora:
        raise ValueError("the recipe needs at least one corpus")
    names = [corpus.get("name", "") for corpus in corpora]
    if len(set(names)) != len(names) or not all(
        re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) for name in names
    ):
        raise ValueError("each corpus needs a unique lowercase name")
    pools = {item["name"]: item for item in corpora if item.get("purpose") == "pool"}
    for item in corpora:
        if item.setdefault("purpose", "phase") not in PURPOSES:
            raise ValueError(f"corpus purpose must be one of {PURPOSES}")
        if "select" in item:
            choice = item["select"]
            if (
                not isinstance(choice, dict)
                or set(choice) != {"pool", "method"}
                or choice["method"] not in SELECTIONS
                or choice["pool"] not in pools
                or item["purpose"] in ("evaluation", "pool")
            ):
                raise ValueError(
                    f"{item['name']} must train on a selection {SELECTIONS} from a pool"
                )
            if "mix" in item or "validation_tokens_per_language" in item:
                raise ValueError(
                    f"{item['name']} takes its sources and held-out split from its pool"
                )
            pool = pools[choice["pool"]]
            item["mix"] = dict(pool.get("mix") or {})
            item["validation_tokens_per_language"] = validation_budget(recipe, pool)
        for name in ("training_tokens", "validation_tokens_per_language"):
            value = item.get(name, corpus["validation_tokens_per_language"])
            if type(value) is not int or value <= 0:
                raise ValueError(f"{item['name']} {name} must be a positive integer")
        if "corpus_tokens" in item and (
            type(item["corpus_tokens"]) is not int
            or item["corpus_tokens"] < item["training_tokens"]
        ):
            raise ValueError(f"{item['name']} corpus_tokens must hold its training_tokens")
        mix = item.get("mix")
        if not mix or any(source not in sources for source in mix):
            raise ValueError(f"{item['name']} mixes sources missing from the catalog")
        if any(type(share) is not int or share <= 0 for share in mix.values()):
            raise ValueError("source shares must be positive integer percentages")
        shares = Counter()
        for source, share in mix.items():
            shares[sources[source]["language"]] += share
        if sum(shares.values()) != 100 or not all(shares[language] for language in LANGUAGES):
            raise ValueError(
                f"{item['name']} must allocate 100 percent, with a share for each language"
            )
    selected = [item for item in corpora if "select" in item]
    if selected:
        settings = recipe.get("selection")
        if not isinstance(settings, dict):
            raise ValueError("a recipe that selects documents needs a selection section")
        for role in ("prior", "conditional"):
            choice = settings.get(role)
            if (
                not isinstance(choice, dict)
                or set(choice) != {"checkpoint", "sha256"}
                or not isinstance(choice["checkpoint"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", str(choice["sha256"]))
            ):
                raise ValueError(
                    f"the selection's {role} model needs a checkpoint and its SHA-256 digest"
                )
        for name in ("scored_share", "label_share"):
            if type(settings.get(name)) is not int or not 0 < settings[name] <= 50:
                raise ValueError(f"the selection's {name} must be a percentage up to 50")
        for item in selected:
            pool = pools[item["select"]["pool"]]
            unscored = corpus_tokens(pool) * (100 - settings["scored_share"]) // 100
            if unscored < corpus_tokens(item):
                raise ValueError(
                    f"{item['name']} needs more tokens than its pool leaves unscored"
                )
    elif "selection" in recipe:
        raise ValueError("the selection section needs a corpus that selects documents")
    # Keep only the sources this recipe uses, so adding a dataset to the shared
    # catalog leaves existing recipes and their prepared corpora unchanged.
    recipe["sources"] = [
        sources[source] for source in sorted({s for c in corpora for s in c["mix"]})
    ]
    model, training = read_config(path.parent / recipe["training_config"])
    if recipe["seed"] != training.seed:
        raise ValueError("use the same seed for the recipe and training")
    tokens_per_step = training.batch_size * training.accumulation_steps * model.context
    phases = [item for item in corpora if item["purpose"] == "phase"]
    alternatives = [item for item in corpora if item["purpose"] == "alternative"]
    if bool(phases) == bool(alternatives):
        raise ValueError("a recipe trains either sequential phases or alternatives")
    start = continued_step(recipe)
    if start and not phases:
        raise ValueError("continue_from continues a recipe's sequential phases")
    budgets = (
        [sum(item["training_tokens"] for item in phases)]
        if phases
        else [item["training_tokens"] for item in alternatives]
    )
    if any(
        training.steps != start + math.ceil(tokens / tokens_per_step)
        for tokens in budgets
    ):
        raise ValueError(
            "training steps must cover the recipe's token budget without shortening it"
        )
    if len(phases) > 1 and (
        training.schedule != "wsd"
        or phase_ends(phases, tokens_per_step, start)[-2]
        != training.steps - training.decay_steps
    ):
        raise ValueError(
            "the last phase must start where the learning rate begins to decay"
        )
    if "initialize_from" in recipe and not isinstance(recipe["initialize_from"], str):
        raise ValueError("initialize_from must name a checkpoint")
    if recipe["evaluation"]["framework"] != "lm-eval":
        raise ValueError("evaluation uses the standard lm-evaluation-harness tasks")
    if "sft" in recipe and (
        type(recipe["sft"]["epochs"]) is not int or recipe["sft"]["epochs"] <= 0
    ):
        raise ValueError("SFT epochs must be positive")
    return recipe


def phase_ends(phases, tokens_per_step, start=0):
    return [
        start + math.ceil(tokens / tokens_per_step)
        for tokens in itertools.accumulate(item["training_tokens"] for item in phases)
    ]


def continued_step(recipe):
    """Return the step of the checkpoint the first phase continues, or 0.

    A recipe that continues a checkpoint carries its weights, optimizer and
    step on, so its schedule and phase ends count from that step.
    """
    source = recipe.get("continue_from")
    if source is None:
        return 0
    if "initialize_from" in recipe:
        raise ValueError("choose initialize_from or continue_from")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("checkpoint"), str)
        or type(source.get("step")) is not int
        or source["step"] <= 0
    ):
        raise ValueError(
            "continue_from needs a checkpoint and the positive step it was saved at"
        )
    return source["step"]


def sources_by_id(recipe):
    return {source["id"]: source for source in recipe["sources"]}


def validation_budget(recipe, item):
    return item.get(
        "validation_tokens_per_language",
        recipe["corpus"]["validation_tokens_per_language"],
    )


def validation_fraction(recipe, item):
    """The share of documents held out, sized so both splits fill together.

    Every language holds out the same tokens, so the sources of the language
    with the smallest share need the most validation tokens for their training
    tokens. A slightly larger fraction than their proportion fills their
    validation split just before training, without reading far past either
    budget.
    """
    for settings in (item, recipe["corpus"]):
        if "validation_fraction" in settings:
            return settings["validation_fraction"]
    smallest = min(language_shares(recipe, item).values())
    ratio = 100 / smallest * validation_budget(recipe, item) / corpus_tokens(item)
    return min(0.25, 1.25 * ratio / (1 + ratio))


def language_shares(recipe, item):
    """Each language's percentage of a corpus, the sum of its sources' shares."""
    sources = sources_by_id(recipe)
    shares = Counter()
    for source_id, share in item["mix"].items():
        shares[sources[source_id]["language"]] += share
    return shares


def source_targets(recipe, item, source_id):
    """A source's training and held-out tokens in a corpus.

    Every language holds out the same tokens, which its sources divide as
    they divide its training tokens.
    """
    share = item["mix"][source_id]
    language = sources_by_id(recipe)[source_id]["language"]
    return {
        "train": corpus_tokens(item) * share // 100,
        "validation": validation_budget(recipe, item)
        * share
        // language_shares(recipe, item)[language],
    }


def corpus_paths(recipe, workdir):
    workdir = Path(workdir)
    names = [item["name"] for item in recipe["corpora"]]
    if len(names) == 1:
        return {names[0]: workdir / "corpus"}
    return {name: workdir / f"corpus-{name}" for name in names}


def streamed(recipe):
    """The corpora read from the sources; a selected corpus comes from its pool."""
    return [item for item in recipe["corpora"] if "select" not in item]


def trained(recipe):
    return [
        item
        for item in recipe["corpora"]
        if item["purpose"] not in ("evaluation", "pool")
    ]


def run_paths(recipe_path, recipe):
    stem = Path(recipe_path).stem
    trained_names = [item["name"] for item in trained(recipe)]
    if len(trained_names) == 1:
        return {trained_names[0]: Path("runs") / stem}
    return {name: Path("runs") / f"{stem}-{name}" for name in trained_names}


# Settings that shape training and evaluation but neither the tokenizer nor
# the corpora, so they may change without rebuilding either.
TRAINING_SETTINGS = ("training_config", "evaluation", "sft", "initialize_from", "continue_from")


def corpus_tokens(item):
    """The training tokens a corpus holds; training may read fewer of them."""
    return item.get("corpus_tokens", item["training_tokens"])


def data_fingerprint(path):
    """What the tokenizer and the corpora are built from.

    The recipe without its training settings, and with each corpus's size in
    place of the tokens training reads, so that another budget, model or
    schedule keeps the prepared data.
    """
    recipe = read_recipe(path)
    data = {key: value for key, value in recipe.items() if key not in TRAINING_SETTINGS}
    data["corpora"] = [
        {
            **{key: value for key, value in item.items() if key != "training_tokens"},
            "corpus_tokens": corpus_tokens(item),
        }
        for item in recipe["corpora"]
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def inspect_recipe(path, workdir):
    recipe = read_recipe(path)
    model, training = read_config(
        Path(path).resolve().parent / recipe["training_config"]
    )
    with torch.device("meta"):
        parameters = sum(
            p.numel()
            for p in LanguageModel(
                recipe["tokenizer"]["vocabulary_size"], model
            ).parameters()
        )
    workdir = Path(workdir)
    corpora = corpus_paths(recipe, workdir)
    runs = run_paths(path, recipe)
    config = str(Path(path).parent / recipe["training_config"])
    tasks = recipe["evaluation"]["base_tasks"]
    japanese = recipe["evaluation"].get("llm_jp_eval_tasks", [])
    commands = [
        [
            "python",
            "-m",
            "pretrain",
            "fetch-benchmarks",
            "--output",
            str(workdir / "benchmarks"),
        ],
        *[
            [
                "python",
                "-m",
                "pretrain",
                "recipe",
                "--config",
                str(path),
                "--workdir",
                str(workdir),
                "--stage",
                stage,
            ]
            for stage in (
                "tokenizer",
                "corpus",
                *(["select"] if len(streamed(recipe)) < len(recipe["corpora"]) else []),
                "validate",
            )
        ],
    ]

    def train_command(name):
        command = [
            "python",
            "-m",
            "pretrain",
            "train",
            "--config",
            config,
            "--data",
            str(corpora[name]),
            "--output",
            str(runs[name]),
            "--device",
            "cuda",
            "--precision",
            "bfloat16",
        ]
        return command

    def benchmark_command(checkpoint, selected, japanese):
        return [
            "python",
            "-m",
            "pretrain",
            "benchmark",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda",
            "--output",
            str(Path(checkpoint).parent / "benchmarks.json"),
            *(["--llm-jp-eval", *japanese] if japanese else []),
            "--tasks",
            *selected,
        ]

    phases = [item for item in recipe["corpora"] if item["purpose"] == "phase"]
    tokens_per_step = training.batch_size * training.accumulation_steps * model.context
    start = continued_step(recipe)
    if phases:
        previous = None
        for item, end in zip(phases, phase_ends(phases, tokens_per_step, start)):
            command = train_command(item["name"])
            if previous is not None:
                command += ["--continue-from", str(previous / "last.pt")]
            elif "continue_from" in recipe:
                command += ["--continue-from", recipe["continue_from"]["checkpoint"]]
            elif "initialize_from" in recipe:
                command += ["--initialize-from", recipe["initialize_from"]]
            if end != training.steps:
                command += ["--max-steps", str(end)]
            commands.append(command)
            previous = runs[item["name"]]
        commands.append(benchmark_command(previous / "last.pt", tasks, japanese))
    else:
        evaluations = [
            corpora[item["name"]]
            for item in recipe["corpora"]
            if item["purpose"] == "evaluation"
        ]
        for item in recipe["corpora"]:
            if item["purpose"] != "alternative":
                continue
            command = train_command(item["name"])
            if "initialize_from" in recipe:
                command += ["--initialize-from", recipe["initialize_from"]]
            commands.append(command)
            checkpoint = runs[item["name"]] / "last.pt"
            for data in evaluations:
                commands.append(
                    [
                        "python",
                        "-m",
                        "pretrain",
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        str(data),
                        "--other-data",
                    ]
                )
            commands.append(benchmark_command(checkpoint, tasks, japanese))
    if phases and "sft" in recipe:
        pretrained = previous / "last.pt"
        stem = Path(path).stem
        instruction_output = Path("runs") / (
            stem.replace("pretrain-", "instructions-", 1)
            if stem.startswith("pretrain-")
            else "instructions-" + stem
        )
        commands += [
            [
                "python",
                "-m",
                "pretrain",
                "fetch-instructions",
                "--output",
                str(workdir / "instruction-source"),
            ],
            [
                "python",
                "-m",
                "pretrain",
                "prepare-instructions",
                "--source",
                str(workdir / "instruction-source"),
                "--tokenizer",
                str(workdir / "tokenizer/tokenizer.json"),
                "--output",
                str(workdir / "instructions"),
                "--context",
                str(model.context),
                "--validation-conversations",
                str(recipe["sft"]["validation_conversations_per_language"]),
                "--max-reply-tokens",
                str(recipe["sft"]["max_reply_tokens"]),
                "--benchmarks",
                str(workdir / "benchmarks"),
            ],
            [
                "python",
                "-m",
                "pretrain",
                "recipe",
                "--config",
                str(path),
                "--workdir",
                str(workdir),
                "--stage",
                "sft-config",
            ],
            [
                "python",
                "-m",
                "pretrain",
                "train",
                "--config",
                str(workdir / "sft.toml"),
                "--data",
                str(workdir / "instructions"),
                "--output",
                str(instruction_output),
                "--initialize-from",
                str(pretrained),
                "--device",
                "cuda",
                "--precision",
                "bfloat16",
            ],
            benchmark_command(
                instruction_output / "last.pt",
                [*tasks, *recipe["evaluation"]["instruction_tasks"]],
                [*japanese, *recipe["evaluation"].get("llm_jp_eval_instruction_tasks", [])],
            ),
        ]
    sources = sources_by_id(recipe)
    return {
        "data_fingerprint": data_fingerprint(path),
        "parameters": parameters,
        "initialization": recipe.get("continue_from")
        or recipe.get("initialize_from", "random_weights"),
        "model": asdict(model),
        "training": asdict(training),
        "nominal_training_tokens": sum(item["training_tokens"] for item in phases)
        if phases
        else trained(recipe)[0]["training_tokens"],
        "scheduled_training_tokens": (training.steps - start) * tokens_per_step,
        "tokenizer": recipe["tokenizer"],
        "corpora": [
            {
                "name": item["name"],
                "purpose": item["purpose"],
                "path": str(corpora[item["name"]]),
                "training_tokens": item["training_tokens"],
                "corpus_tokens": corpus_tokens(item),
                "validation_tokens_per_language": validation_budget(recipe, item),
            }
            for item in recipe["corpora"]
        ],
        "sources": [
            {
                **sources[source],
                "corpus": item["name"],
                "share": share,
                "training_tokens": item["training_tokens"] * share // 100,
            }
            for item in recipe["corpora"]
            for source, share in item["mix"].items()
        ],
        "packed_dtype": "<u2",
        "minimum_packed_bytes": 2
        * sum(
            corpus_tokens(item) + 2 * validation_budget(recipe, item)
            for item in recipe["corpora"]
        ),
        "checkpoint_bytes_approximate": parameters * 12,
        "evaluation": recipe["evaluation"],
        "sft": recipe.get("sft"),
        "commands": commands,
    }


def template_fields(template):
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


def source_columns(source):
    if "template" in source:
        fields = {*template_fields(source["template"]), "id", "url"}
    elif "conversation_field" in source:
        fields = {source["conversation_field"], "id", "url"}
    else:
        fields = {source.get("text_field", "text"), "id", "url"}
    for name in FILTERS:
        fields.update(key.split(".")[0] for key in source.get(name, {}))
    return sorted(fields)


def read_json_lines(handle, form):
    if form == "jsonl.zst":
        import zstandard

        stream = zstandard.ZstdDecompressor().stream_reader(handle)
    elif form == "jsonl.gz":
        stream = gzip.GzipFile(fileobj=handle)
    else:
        stream = handle
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


class RemoteFile(io.RawIOBase):
    """A file read in order over HTTP, continuing where a response stopped early.

    Only reads that bring nothing count against the attempts.
    """

    def __init__(self, url, attempts=5):
        self.url, self.attempts = url, attempts
        self.position, self.size, self.response = 0, None, None

    def readable(self):
        return True

    def connect(self):
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-"}
        )
        self.response = urllib.request.urlopen(request, timeout=120)
        if self.response.status == 206:
            size = int(self.response.headers["Content-Range"].rpartition("/")[2])
        elif self.position == 0:
            size = int(self.response.headers["Content-Length"])
        else:
            raise ValueError("the host ignored a range request")
        if self.size not in (None, size):
            raise ValueError("the file changed while it was read")
        self.size = size

    def readinto(self, buffer):
        stalled = 0
        while self.size is None or self.position < self.size:
            error = None
            try:
                if self.response is None:
                    self.connect()
                count = self.response.readinto(buffer)
                if count:
                    self.position += count
                    return count
            except (OSError, http.client.HTTPException) as failure:
                error = failure
            if self.response is not None:
                self.response.close()
                self.response = None
            stalled += 1
            if stalled == self.attempts:
                raise error or ValueError("the file ended before its size")
            time.sleep(2**stalled)
        return 0

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()


def read_url(url, query=None, attempts=5):
    """The body at a URL, or the answer to a GraphQL query sent there.

    A failed request is sent again, after the wait a throttling host asks for.
    """
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url)
        if query is not None:
            request.data = json.dumps(query).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            wait = getattr(error, "headers", None) and error.headers.get("Retry-After")
            time.sleep(max(2**attempt, int(wait) if wait and wait.isdigit() else 0))


def hub_files(source):
    """A Hugging Face dataset's files at the source's commit, how to open one and its size."""
    from huggingface_hub import HfApi, HfFileSystem

    info = HfApi(token=False).dataset_info(
        source["repository"], revision=source["revision"], files_metadata=True
    )
    if info.sha != source["revision"]:
        raise ValueError("the resolved dataset revision does not match the recipe")
    filesystem = HfFileSystem(token=False)
    sizes = {item.rfilename: item.size for item in info.siblings}

    def open_file(name):
        # Large reads keep the requests few, well within the Hub's rate limits
        # when many parts read at once.
        return filesystem.open(
            f"datasets/{source['repository']}@{source['revision']}/{name}",
            "rb",
            block_size=16 * 1024 * 1024,
        )

    return list(sizes), open_file, lambda names: {name: sizes[name] for name in names}


def gitlab_files(source):
    """A GitLab project's files at the source's commit, how to open one and their sizes."""
    project = "{}/api/v4/projects/{}".format(
        source["host"], urllib.parse.quote(source["repository"], safe="")
    )
    commit = json.loads(read_url(f"{project}/repository/commits/{source['revision']}"))
    if commit["id"] != source["revision"]:
        raise ValueError("the resolved dataset revision does not match the recipe")
    names = []
    for page in itertools.count(1):
        query = urllib.parse.urlencode(
            {
                "ref": source["revision"],
                "path": source["prefix"].rpartition("/")[0],
                "recursive": "true",
                "per_page": 100,
                "page": page,
            }
        )
        entries = json.loads(read_url(f"{project}/repository/tree?{query}"))
        names += [entry["path"] for entry in entries if entry["type"] == "blob"]
        if len(entries) < 100:
            break

    def open_file(name):
        url = "{}/{}/-/raw/{}/{}".format(
            source["host"], source["repository"], source["revision"], urllib.parse.quote(name)
        )
        return io.BufferedReader(RemoteFile(url), 16 * 1024 * 1024)

    def sizes(names):
        # One query asks for every file; a large file's size is that of the
        # content it stands for, not of its pointer.
        answer = json.loads(
            read_url(
                f"{source['host']}/api/graphql",
                {
                    "query": "query($project: ID!, $ref: String!, $paths: [String!]!) "
                    "{ project(fullPath: $project) { repository "
                    "{ blobs(ref: $ref, paths: $paths) { nodes { path rawSize } } } } }",
                    "variables": {
                        "project": source["repository"],
                        "ref": source["revision"],
                        "paths": list(names),
                    },
                },
            )
        )
        nodes = answer["data"]["project"]["repository"]["blobs"]["nodes"]
        found = {node["path"]: int(node["rawSize"]) for node in nodes}
        if set(found) != set(names):
            raise ValueError(f"{source['id']}'s host did not give the size of every file")
        return found

    return names, open_file, sizes


def source_shards(source):
    """The files a source reads at its commit, how to open one and their sizes."""
    names, open_file, sizes = (gitlab_files if "host" in source else hub_files)(source)
    form = source.get("format", "parquet")
    excluded = set(source.get("exclude_files", ()))
    shards = sorted(
        name
        for name in names
        if name.startswith(source["prefix"])
        and name.endswith(SOURCE_FORMATS[form])
        and name not in excluded
    )
    if not shards:
        raise ValueError(f"no {form} files found for {source['id']}")
    return shards, open_file, sizes


def source_sizes(source):
    """The bytes of each file the source reads, at its commit."""
    shards, _, sizes = source_shards(source)
    return sizes(shards)


def reading_order(source, shards, seed):
    """The source's files in the order its readers take them."""
    order = sorted(shards)
    random.Random(f"{seed}:{source['id']}").shuffle(order)
    return order


def iter_source(source, seed, part=None):
    """Yield the rows of a part's files, or of all the source's files."""
    from pyarrow import parquet

    shards, open_file, _ = source_shards(source)
    form = source.get("format", "parquet")
    shards = part["files"] if part is not None else reading_order(source, shards, seed)
    columns = source_columns(source)
    for shard in shards:
        generator = random.Random(f"{seed}:{source['id']}:{shard}")
        with open_file(shard) as handle:
            if form == "parquet":
                with parquet.ParquetFile(handle) as table:
                    present = set(table.schema_arrow.names)
                    groups = list(range(table.num_row_groups))
                    generator.shuffle(groups)
                    for group in groups:
                        for batch in table.iter_batches(
                            batch_size=64,
                            row_groups=[group],
                            columns=[name for name in columns if name in present],
                        ):
                            rows = batch.to_pylist()
                            generator.shuffle(rows)
                            for row in rows:
                                yield {**row, "source_file": shard}
            else:
                lines = read_json_lines(handle, form)
                while rows := list(itertools.islice(lines, 1024)):
                    generator.shuffle(rows)
                    for row in rows:
                        yield {
                            **{name: row.get(name) for name in columns},
                            "source_file": shard,
                        }


def field_value(row, name):
    value = row
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def passes_filters(row, source):
    values = {
        name: {key: field_value(row, key) for key in source.get(name, {})}
        for name in FILTERS
    }
    return (
        all(
            values["equal"][key] == value
            for key, value in source.get("equal", {}).items()
        )
        and all(
            finite_number(values["minimum"][key]) and values["minimum"][key] >= value
            for key, value in source.get("minimum", {}).items()
        )
        and all(
            finite_number(values["maximum"][key]) and values["maximum"][key] <= value
            for key, value in source.get("maximum", {}).items()
        )
        and (
            "any_minimum" not in source
            or any(
                finite_number(values["any_minimum"][key])
                and values["any_minimum"][key] >= value
                for key, value in source["any_minimum"].items()
            )
        )
    )


def source_text(row, source):
    """A row's text: one field, fields filled into a template, or a conversation.

    A conversation reads as instruction tuning writes one, a role and a
    colon before each turn. None if the row lacks what the source reads.
    """
    if "template" in source:
        values = {name: row.get(name) for name in template_fields(source["template"])}
        if not all(isinstance(value, str) for value in values.values()):
            return None
        return source["template"].format_map(values)
    if "conversation_field" in source:
        turns = row.get(source["conversation_field"])
        if not isinstance(turns, list) or not turns:
            return None
        messages = []
        for turn in turns:
            role = isinstance(turn, dict) and SPEAKERS.get(turn.get("role", turn.get("from")))
            content = isinstance(turn, dict) and turn.get("content", turn.get("value"))
            if not role or not isinstance(content, str):
                return None
            messages.append({"role": role, "content": content})
        return conversation_text(messages)[0]
    text = row.get(source.get("text_field", "text"))
    return text if isinstance(text, str) else None


def document_text(row, source, corpus):
    text = source_text(row, source)
    if text is None:
        return None
    text = unicodedata.normalize("NFC", text).strip()
    minimum = source.get("minimum_characters", corpus["minimum_characters"])
    maximum = source.get("maximum_characters", corpus["maximum_characters"])
    if not minimum <= len(text) <= maximum or not passes_filters(row, source):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.7:
        return None
    return text


def document_digest(text, seed):
    return hashlib.sha256(f"{seed}\0{' '.join(text.split())}".encode()).digest()


def validation_document(digest, fraction):
    return int.from_bytes(digest[:8], "big") < fraction * 2**64


@contextmanager
def output_directory(output):
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        yield staging
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def seen_documents(directory=None):
    database = sqlite3.connect(
        ":memory:" if directory is None else Path(directory) / "seen.sqlite3"
    )
    database.execute("CREATE TABLE documents (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    return database


def earlier_documents(recipe):
    """Yield the digest of every document in the earlier corpora the recipe leaves out.

    Each earlier corpus keeps the digest of every document it holds, so a
    later corpus can add documents that training has not read yet, as a
    larger budget or a new phase needs.
    """
    for entry in recipe["corpus"].get("excluded_corpora", []):
        directory = Path(entry["path"])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["seed"] != recipe["seed"]:
            raise ValueError(f"{directory} digests its documents with another seed")
        documents = directory / manifest["provenance"]["filename"]
        if file_digest(documents) != entry["documents_sha256"]:
            raise ValueError(f"the documents of {directory} do not match their checksum")
        for document in read_json_documents(documents):
            yield bytes.fromhex(document["digest"])


def accepted_documents(
    source,
    recipe,
    benchmarks,
    database,
    stats,
    training_fraction=None,
    part=None,
    earlier=None,
):
    """Yield new clean documents; with training_fraction, only training ones.

    Documents recorded in `earlier`, a database of earlier corpora's
    documents, are passed over.
    """
    for row in iter_source(source, recipe["seed"], part):
        text = document_text(row, source, recipe["corpus"])
        if text is None:
            stats["quality_rejected"] += 1
            continue
        if benchmarks.matches(text, source["language"]):
            stats["contaminated"] += 1
            continue
        digest = document_digest(text, recipe["seed"])
        if training_fraction is not None and validation_document(
            digest, training_fraction
        ):
            continue
        if (
            earlier is not None
            and earlier.execute(
                "SELECT 1 FROM documents WHERE digest = ?", (digest,)
            ).fetchone()
        ):
            stats["used_earlier"] += 1
            continue
        if (
            database.execute(
                "INSERT OR IGNORE INTO documents VALUES (?)", (digest,)
            ).rowcount
            == 0
        ):
            stats["duplicates"] += 1
            continue
        yield text, digest, row


def prepare_tokenizer(recipe_path, output, benchmarks_path):
    recipe = read_recipe(recipe_path)
    settings = recipe["tokenizer"]
    samples = None
    with output_directory(output) as staging, ExitStack() as stack:
        if "reuse_sha256" in settings:
            source = Path(settings["reuse_from"])
            if file_digest(source) != settings["reuse_sha256"]:
                raise ValueError("the reused tokenizer does not match its checksum")
            shutil.copyfile(source, staging / "tokenizer.json")
            tokenizer = Tokenizer.from_file(str(staging / "tokenizer.json"))
            if (
                tokenizer.get_vocab_size() != settings["vocabulary_size"]
                or tokenizer.token_to_id(BOS) is None
                or tokenizer.token_to_id(EOS) is None
            ):
                raise ValueError("the reused tokenizer does not match the recipe")
        else:
            samples = train_tokenizer(recipe, staging, stack, benchmarks_path)
        manifest = {
            "version": 1,
            "data_fingerprint": data_fingerprint(recipe_path),
            "benchmarks_sha256": file_digest(Path(benchmarks_path) / "sources.json"),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "reused": samples is None,
            "samples": samples,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def pre_tokenizer(settings):
    """How text splits before merging: GPT-2's split by default, or GPT-4's.

    Numbers split into groups of up to three digits under GPT-4's split, or
    into single digits when the recipe asks for them.
    """
    single = settings.get("single_digits", False)
    if settings.get("pretokenizer", "gpt2") == "gpt4":
        pattern = GPT4_SPLIT.replace("DIGITS", r"\p{N}" if single else r"\p{N}{1,3}")
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if single:
        return pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
    return byte_level


def train_tokenizer(recipe, staging, stack, benchmarks_path):
    benchmarks = BenchmarkFilter(benchmarks_path)
    sources = sources_by_id(recipe)
    database = stack.enter_context(closing(seen_documents(staging)))
    destinations = {
        language: stack.enter_context(
            (staging / f"{language}.jsonl").open("w", encoding="utf-8")
        )
        for language in LANGUAGES
    }
    samples = {}
    # The vocabulary follows the first corpus that training reads from.
    first = next(item for item in streamed(recipe) if item["purpose"] != "evaluation")
    shares = language_shares(recipe, first)
    for source_id, share in first["mix"].items():
        source = sources[source_id]
        stats = Counter()
        language = source["language"]
        # Each language's sample holds the same bytes, divided as the mix divides it.
        target = recipe["tokenizer"]["bytes_per_language"] * share // shares[language]
        for text, _, _ in accepted_documents(
            source,
            recipe,
            benchmarks,
            database,
            stats,
            training_fraction=validation_fraction(recipe, first),
        ):
            destinations[language].write(
                json.dumps({"text": text}, ensure_ascii=False) + "\n"
            )
            stats["bytes"] += len(text.encode("utf-8"))
            stats["documents"] += 1
            if stats["bytes"] >= target:
                break
        if stats["bytes"] < target:
            raise ValueError(f"{source_id} is short of the tokenizer sample budget")
        samples[source_id] = dict(stats)
    for destination in destinations.values():
        destination.flush()
    settings = recipe["tokenizer"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizer(settings)
    tokenizer.decoder = decoders.ByteLevel()
    # Unused special tokens hold places for markers that later training may
    # need, since the vocabulary cannot grow once a model learns from it.
    reserved = [f"<|reserved_{index}|>" for index in range(settings.get("reserved_tokens", 0))]
    trainer = trainers.BpeTrainer(
        vocab_size=settings["vocabulary_size"],
        min_frequency=2,
        show_progress=False,
        special_tokens=[BOS, EOS, *reserved],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    balanced = itertools.chain.from_iterable(
        itertools.zip_longest(
            *(read_documents(staging / f"{language}.jsonl") for language in LANGUAGES)
        )
    )
    tokenizer.train_from_iterator(
        (text for text in balanced if text is not None), trainer=trainer
    )
    if tokenizer.get_vocab_size() != recipe["tokenizer"]["vocabulary_size"]:
        raise ValueError(
            "the tokenizer sample cannot fill the requested vocabulary; add diverse text"
        )
    tokenizer.save(str(staging / "tokenizer.json"))
    stack.close()
    for filename in ("ja.jsonl", "en.jsonl", "seen.sqlite3"):
        (staging / filename).unlink()
    return samples


def source_parts(recipe):
    """Split the reading of every source into parts that run independently.

    A source used by several corpora is read once, in corpus order, and each
    part covers every corpus's demand on its own subset of the source's files.
    The files are dealt so that the parts hold about equal bytes, and each
    part's demand follows its bytes, so a part dealt only small files is not
    asked for what the large ones hold. The split follows only the recipe and
    the files at each source's commit, so any number of workers gives the same
    corpora. Parts read PART_SLACK beyond their share, so documents found
    earlier in another part or source can be dropped while merging.
    """
    sources = sources_by_id(recipe)
    demands = {}
    for item in streamed(recipe):
        for source_id in item["mix"]:
            demands.setdefault(source_id, {})[item["name"]] = source_targets(
                recipe, item, source_id
            )
    parts = []
    for source_id, wanted in demands.items():
        sizes = source_sizes(sources[source_id])
        order = reading_order(sources[source_id], sizes, recipe["seed"])
        total = sum(sum(targets.values()) for targets in wanted.values())
        count = min(
            len(order), max(1, math.ceil(total / recipe["corpus"]["part_tokens"]))
        )
        files, loads = [[] for _ in range(count)], [0] * count
        # Each file, the largest first, goes to the part holding the fewest bytes.
        for name in sorted(order, key=lambda name: -sizes[name]):
            lightest = loads.index(min(loads))
            files[lightest].append(name)
            loads[lightest] += sizes[name]
        position = {name: index for index, name in enumerate(order)}
        held = sum(loads)
        for index in range(count):
            fraction = loads[index] / held if held else 1 / count
            parts.append(
                {
                    "source": source_id,
                    "index": index,
                    "count": count,
                    "files": sorted(files[index], key=position.get),
                    "targets": {
                        name: {
                            split: math.ceil(target * (1 + PART_SLACK) * fraction)
                            for split, target in targets.items()
                        }
                        for name, targets in wanted.items()
                    },
                }
            )
    return parts


def part_directory(parts_root, part):
    return Path(parts_root) / f"{part['source']}-{part['index']:04d}"


def prepare_part(recipe_path, tokenizer_directory, benchmarks_path, part, parts_root):
    """Tokenize one part of a source into per-corpus pieces for merging."""
    recipe = read_recipe(recipe_path)
    source = sources_by_id(recipe)[part["source"]]
    corpora = {item["name"]: item for item in recipe["corpora"]}
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_directory) / "tokenizer.json"))
    bos, eos = tokenizer.token_to_id(BOS), tokenizer.token_to_id(EOS)
    benchmarks = BenchmarkFilter(benchmarks_path)
    output = part_directory(parts_root, part)
    output.mkdir(parents=True)
    scanned = Counter()
    summary, collected = {}, {}
    with closing(seen_documents()) as database, ExitStack() as opened:
        earlier = None
        if recipe["corpus"].get("excluded_corpora"):
            # The stage records the earlier corpora's documents here before any
            # part is read.
            location = (Path(parts_root) / "seen.sqlite3").resolve().as_uri()
            earlier = opened.enter_context(
                closing(sqlite3.connect(f"{location}?mode=ro", uri=True))
            )
        stream = accepted_documents(
            source, recipe, benchmarks, database, scanned, part=part, earlier=earlier
        )
        for name, targets in part["targets"].items():
            fraction = validation_fraction(recipe, corpora[name])
            stats = Counter()
            before = Counter(scanned)
            with ExitStack() as stack:
                pieces = {
                    split: stack.enter_context(
                        (output / f"{name}.{split}.bin").open("wb")
                    )
                    for split in targets
                }
                documents = stack.enter_context(
                    (output / f"{name}.documents.jsonl").open("w", encoding="utf-8")
                )
                while chunk := list(itertools.islice(stream, 64)):
                    encodings = tokenizer.encode_batch(
                        [entry[0] for entry in chunk], add_special_tokens=False
                    )
                    for (_, digest, row), encoding in zip(chunk, encodings):
                        split = (
                            "validation"
                            if validation_document(digest, fraction)
                            else "train"
                        )
                        if stats[f"{split}_tokens"] >= targets[split]:
                            continue
                        encoded = np.asarray(
                            [bos, *encoding.ids, eos], dtype=TOKEN_PIECE_DTYPE
                        )
                        pieces[split].write(encoded.tobytes())
                        documents.write(
                            json.dumps(
                                {
                                    "digest": digest.hex(),
                                    "id": row.get("id"),
                                    "url": row.get("url"),
                                    "split": split,
                                    "tokens": len(encoded),
                                    "source_file": row["source_file"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        stats[f"{split}_tokens"] += len(encoded)
                    if all(
                        stats[f"{split}_tokens"] >= target
                        for split, target in targets.items()
                    ):
                        break
            summary[name] = dict(scanned - before)
            collected[name] = {split: stats[f"{split}_tokens"] for split in targets}
    (output / "statistics.json").write_text(json.dumps(summary) + "\n")
    (output / "collected.json").write_text(json.dumps(collected) + "\n")
    # Written last, so a later run keeps only parts that finished.
    (output / "finished.json").write_text(json.dumps(part) + "\n")
    print(
        json.dumps({"part": f"{part['source']}-{part['index']}", **summary}), flush=True
    )
    return part


def prepare_corpus(
    recipe_path, tokenizer_directory, workdir, benchmarks_path, workers=1
):
    """Write every corpus read from the sources; no document appears in two of them.

    Corpora that select from a pool are written afterwards by the select stage.
    The parts an interrupted run finished are kept and not read again.
    """
    recipe = read_recipe(recipe_path)
    tokenizer_directory = Path(tokenizer_directory)
    tokenizer_manifest = json.loads((tokenizer_directory / "manifest.json").read_text())
    benchmark_digest = file_digest(Path(benchmarks_path) / "sources.json")
    fingerprint = data_fingerprint(recipe_path)
    if (
        tokenizer_manifest.get("data_fingerprint") != fingerprint
        or tokenizer_manifest["benchmarks_sha256"] != benchmark_digest
    ):
        raise ValueError(
            "use the tokenizer trained with this recipe and benchmark exclusions"
        )
    if (
        file_digest(tokenizer_directory / "tokenizer.json")
        != tokenizer_manifest["tokenizer_sha256"]
    ):
        raise ValueError("tokenizer checksum does not match")
    tokenizer = Tokenizer.from_file(str(tokenizer_directory / "tokenizer.json"))
    if tokenizer.get_vocab_size() != recipe["tokenizer"]["vocabulary_size"]:
        raise ValueError("tokenizer vocabulary does not match the recipe")
    paths = corpus_paths(recipe, workdir)
    parts = source_parts(recipe)
    parts_root = Path(workdir) / ".corpus-parts"
    finished = keep_finished_parts(
        parts_root,
        parts,
        {
            "data_fingerprint": fingerprint,
            "tokenizer_sha256": tokenizer_manifest["tokenizer_sha256"],
            "benchmarks_sha256": benchmark_digest,
        },
    )
    manifests, short = {}, []
    # The corpora are kept only once every one of them is whole.
    with ExitStack() as outputs:
        staging = {
            item["name"]: outputs.enter_context(output_directory(paths[item["name"]]))
            for item in streamed(recipe)
        }
        with closing(seen_documents(parts_root)) as database:
            # Documents of earlier corpora count as seen before any part is read.
            database.executemany(
                "INSERT OR IGNORE INTO documents VALUES (?)",
                ((digest,) for digest in earlier_documents(recipe)),
            )
            database.commit()

            def check(part=None):
                if part is not None:
                    finished.add((part["source"], part["index"]))
                if missing := certain_shortfalls(recipe, parts, finished, parts_root):
                    raise short_of_budget(missing)

            check()
            # Small parts go first, so a source too small for its share is
            # found within minutes instead of after the large parts.
            arguments = sorted(
                (
                    (recipe_path, tokenizer_directory, benchmarks_path, part, parts_root)
                    for part in parts
                    if (part["source"], part["index"]) not in finished
                ),
                key=lambda argument: sum(
                    sum(targets.values()) for targets in argument[3]["targets"].values()
                ),
            )
            if workers == 1:
                for argument in arguments:
                    prepare_part(*argument)
                    check(argument[3])
            else:
                run_parts(arguments, workers, check)
            for item in streamed(recipe):
                try:
                    manifests[item["name"]] = write_corpus(
                        recipe,
                        item,
                        parts,
                        parts_root,
                        tokenizer,
                        tokenizer_directory,
                        tokenizer_manifest,
                        benchmark_digest,
                        fingerprint,
                        database,
                        staging[item["name"]],
                    )
                except ShortCorpus as error:
                    short += error.args[0]
        if short:
            raise short_of_budget(short)
    shutil.rmtree(parts_root)
    return manifests


def keep_finished_parts(parts_root, parts, stage):
    """Ready the directory of parts, keeping those an interrupted run of the same stage finished.

    A part writes its record of itself last, so a part without it is read
    again. A stage with another recipe, tokenizer or benchmarks reads every
    part again. Return the finished parts.
    """
    record = parts_root / "stage.json"
    if parts_root.exists() and (
        not record.exists() or json.loads(record.read_text()) != stage
    ):
        shutil.rmtree(parts_root)
    parts_root.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(stage) + "\n")
    finished = set()
    for part in parts:
        directory = part_directory(parts_root, part)
        marker = directory / "finished.json"
        if marker.exists() and json.loads(marker.read_text()) == part:
            finished.add((part["source"], part["index"]))
        elif directory.exists():
            shutil.rmtree(directory)
    # The record of seen documents is rebuilt from the earlier corpora each run.
    (parts_root / "seen.sqlite3").unlink(missing_ok=True)
    return finished


def short_of_budget(short):
    return ValueError(
        "sources short of their clean corpus budget, which no source may "
        "repeat or pad to meet: " + "; ".join(short)
    )


def certain_shortfalls(recipe, parts, finished, parts_root):
    """Name each source that stays short of its share even if its unread parts fill theirs.

    What the finished parts gathered and the most the others may read bound
    what a source can give, before the merge drops any repeats.
    """
    reach = Counter()
    for part in parts:
        if (part["source"], part["index"]) in finished:
            gathered = json.loads(
                (part_directory(parts_root, part) / "collected.json").read_text()
            )
        else:
            gathered = part["targets"]
        for name, splits in gathered.items():
            for split, tokens in splits.items():
                reach[name, part["source"], split] += tokens
    return [
        f"{item['name']} {source_id} lacks at least "
        f"{target - reach[item['name'], source_id, split]:,} {split} tokens"
        for item in streamed(recipe)
        for source_id in item["mix"]
        for split, target in source_targets(recipe, item, source_id).items()
        if reach[item["name"], source_id, split] < target
    ]


def run_parts(arguments, workers, check=None):
    """Prepare each part in a process of its own, at most `workers` at a time.

    A part whose process fails or is ended, as one ended for lack of memory
    is, stops the stage at once instead of leaving it waiting for that part.
    `check` hears of each finished part and stops the stage by raising.
    """
    context = multiprocessing.get_context("fork")
    pending, running = list(arguments), {}
    try:
        while pending or running:
            while pending and len(running) < workers:
                argument = pending.pop(0)
                process = context.Process(target=prepare_part, args=argument)
                process.start()
                running[process] = argument[3]
            multiprocessing.connection.wait([process.sentinel for process in running])
            for process in [process for process in running if not process.is_alive()]:
                process.join()
                part = running.pop(process)
                if process.exitcode:
                    raise RuntimeError(
                        f"a corpus part's process ended with exit code {process.exitcode}"
                    )
                if check is not None:
                    check(part)
    finally:
        for process in running:
            process.terminate()
            process.join()


def read_json_documents(path):
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def read_piece(path):
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=TOKEN_PIECE_DTYPE)
    return np.memmap(path, mode="r", dtype=TOKEN_PIECE_DTYPE)


class ShortCorpus(ValueError):
    """Sources holding too few clean tokens for their shares of a corpus."""


def write_corpus(
    recipe,
    item,
    parts,
    parts_root,
    tokenizer,
    tokenizer_directory,
    tokenizer_manifest,
    benchmark_digest,
    fingerprint,
    database,
    staging,
):
    """Join the parts' pieces of one corpus in recipe order, dropping repeats.

    Every source short of its share is named before the corpus is given up.
    """
    dtype = np.dtype(TOKEN_PIECE_DTYPE)
    sources = sources_by_id(recipe)
    short = []
    with ExitStack() as stack:
        streams = {
            (split, language): stack.enter_context(
                (staging / f"{split}_{language}.bin").open("wb")
            )
            for split in ("train", "validation")
            for language in LANGUAGES
        }
        provenance = stack.enter_context(
            (staging / "documents.jsonl").open("w", encoding="utf-8")
        )
        counts = {language: Counter() for language in LANGUAGES}
        tokens = Counter()
        used = {}
        for source_id, share in item["mix"].items():
            source = sources[source_id]
            language = source["language"]
            targets = source_targets(recipe, item, source_id)
            stats = Counter()
            shards = set()
            for part in parts:
                if part["source"] != source_id:
                    continue
                directory = part_directory(parts_root, part)
                stats.update(
                    json.loads((directory / "statistics.json").read_text())[
                        item["name"]
                    ]
                )
                pieces = {
                    split: read_piece(directory / f"{item['name']}.{split}.bin")
                    for split in targets
                }
                offsets = Counter()
                for document in read_json_documents(
                    directory / f"{item['name']}.documents.jsonl"
                ):
                    split, size = document["split"], document["tokens"]
                    start = offsets[split]
                    offsets[split] += size
                    if stats[f"{split}_tokens"] >= targets[split]:
                        continue
                    if (
                        database.execute(
                            "INSERT OR IGNORE INTO documents VALUES (?)",
                            (bytes.fromhex(document["digest"]),),
                        ).rowcount
                        == 0
                    ):
                        stats["duplicates"] += 1
                        continue
                    streams[split, language].write(
                        pieces[split][start : start + size].tobytes()
                    )
                    provenance.write(
                        json.dumps(
                            {
                                "digest": document["digest"],
                                "source": source_id,
                                "id": document["id"],
                                "url": document["url"],
                                "split": split,
                                "tokens": size,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats[f"{split}_tokens"] += size
                    stats[f"{split}_documents"] += 1
                    tokens[split, language] += size
                    counts[language][split] += 1
                    shards.add(document["source_file"])
                database.commit()
            print(
                json.dumps(
                    {"corpus": item["name"], "source": source_id, **dict(stats)}
                ),
                flush=True,
            )
            missing = [
                f"{item['name']} {source_id} lacks {target - stats[f'{split}_tokens']:,} {split} tokens"
                for split, target in targets.items()
                if stats[f"{split}_tokens"] < target
            ]
            if missing:
                short += missing
                continue
            used[source_id] = {
                **source,
                "share": share,
                "targets": targets,
                "statistics": dict(stats),
                "shards": sorted(shards),
            }
        if short:
            raise ShortCorpus(short)
        stack.close()
        shutil.copyfile(
            tokenizer_directory / "tokenizer.json", staging / "tokenizer.json"
        )
        manifest = {
            "version": 1,
            "dtype": dtype.str,
            "seed": recipe["seed"],
            "corpus": item["name"],
            "data_fingerprint": fingerprint,
            "vocabulary_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "tokenizer_training": tokenizer_manifest,
            "sources": used,
            "benchmarks_sha256": benchmark_digest,
            "documents": counts,
            "duplicates_removed": sum(
                source["statistics"].get("duplicates", 0) for source in used.values()
            ),
            "provenance": {
                "filename": "documents.jsonl",
                "sha256": file_digest(staging / "documents.jsonl"),
            },
            "files": {
                f"{split}_{language}.bin": {
                    "tokens": tokens[split, language],
                    "sha256": file_digest(staging / f"{split}_{language}.bin"),
                }
                for split, language in streams
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def validate_corpus(recipe_path, workdir):
    recipe = read_recipe(recipe_path)
    model, _ = read_config(
        Path(recipe_path).resolve().parent / recipe["training_config"]
    )
    fingerprint = data_fingerprint(recipe_path)
    sources = sources_by_id(recipe)
    result = {}
    for item in recipe["corpora"]:
        data = corpus_paths(recipe, workdir)[item["name"]]
        corpus = Corpus(data)
        corpus.validate_context(model.context)
        manifest = corpus.manifest
        if (
            manifest.get("data_fingerprint") != fingerprint
            or manifest.get("corpus") != item["name"]
        ):
            raise ValueError("the corpus belongs to a different recipe")
        if "select" in item:
            pool = Corpus(
                corpus_paths(recipe, workdir)[item["select"]["pool"]], ("validation",)
            )
            chosen = manifest.get("selection", {})
            if (
                chosen.get("method") != item["select"]["method"]
                or chosen.get("pool_fingerprint") != pool.fingerprint
            ):
                raise ValueError("the corpus was not selected from this recipe's pool")
        if (
            file_digest(data / manifest["provenance"]["filename"])
            != manifest["provenance"]["sha256"]
        ):
            raise ValueError("document provenance checksum does not match")
        if set(manifest["sources"]) != set(item["mix"]):
            raise ValueError("the corpus sources do not match the recipe")
        for source_id in item["mix"]:
            saved = manifest["sources"][source_id]
            targets = source_targets(recipe, item, source_id)
            if (
                any(
                    saved.get(field) != value
                    for field, value in sources[source_id].items()
                )
                or saved["targets"] != targets
            ):
                raise ValueError(
                    "the corpus source or token budget does not match the recipe"
                )
            for split, target in targets.items():
                if saved["statistics"][f"{split}_tokens"] < target:
                    raise ValueError(
                        f"{source_id} is short of its {split} token budget"
                    )
        result[item["name"]] = {
            "files": manifest["files"],
            "sources": manifest["sources"],
        }
    return {"ready": True, "data_fingerprint": fingerprint, "corpora": result}


def write_sft_config(recipe_path, data, tokenizer_directory, output):
    from .instructions import InstructionCorpus

    recipe = read_recipe(recipe_path)
    if "sft" not in recipe:
        raise ValueError("this recipe has no instruction tuning settings")
    corpus = InstructionCorpus(data)
    model, _ = read_config(
        Path(recipe_path).resolve().parent / recipe["training_config"]
    )
    corpus.validate_context(model.context)
    tokenizer_manifest = json.loads(
        (Path(tokenizer_directory) / "manifest.json").read_text()
    )
    if (
        tokenizer_manifest.get("data_fingerprint") != data_fingerprint(recipe_path)
        or corpus.manifest["tokenizer_sha256"] != tokenizer_manifest["tokenizer_sha256"]
        or corpus.manifest["benchmarks_sha256"]
        != tokenizer_manifest["benchmarks_sha256"]
    ):
        raise ValueError(
            "SFT must use this recipe's tokenizer and benchmark exclusions"
        )
    sft = recipe["sft"]
    tokens_per_step = sft["batch_size"] * sft["accumulation_steps"] * model.context
    balanced_tokens = (
        2
        * model.context
        * min(len(corpus.rows["train"][language][0]) for language in LANGUAGES)
    )
    steps = max(2, math.ceil(sft["epochs"] * balanced_tokens / tokens_per_step))
    config = TrainConfig(
        steps=steps,
        batch_size=sft["batch_size"],
        accumulation_steps=sft["accumulation_steps"],
        learning_rate=sft["learning_rate"],
        min_learning_rate=sft["min_learning_rate"],
        warmup_steps=min(steps - 1, max(1, steps // 20)),
        eval_every=max(1, steps // 10),
        save_every=max(1, steps // 10),
        eval_batches=16,
        log_every=max(1, steps // 100),
        seed=recipe["seed"],
        # The recipe may tune with pretraining's optimizer, and read every
        # conversation once per epoch instead of drawing them at random.
        **{key: sft[key] for key in ("optimizer", "sampling") if key in sft},
    )
    text = ""
    for section, values in (("model", asdict(model)), ("training", asdict(config))):
        text += f"[{section}]\n"
        for key, value in values.items():
            if value is not None:
                text += f"{key} = {json.dumps(value)}\n"
        text += "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(text)
    return {
        "config": str(output),
        "steps": steps,
        "balanced_corpus_tokens": balanced_tokens,
        "scheduled_tokens": steps * tokens_per_step,
    }
