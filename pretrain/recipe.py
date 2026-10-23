"""A reproducible bilingual corpus and training recipe, independent of its host."""

import gzip
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import sqlite3
import tempfile
import tomllib
import unicodedata
from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import BOS, EOS, LANGUAGES, Corpus, file_digest, read_documents
from .instructions import benchmark_keys, contamination_text
from .model import LanguageModel
from .training import TrainConfig, read_config

SOURCE_FORMATS = {
    "parquet": ".parquet",
    "jsonl.zst": ".jsonl.zst",
    "jsonl.gz": ".jsonl.gz",
}
FILTERS = ("equal", "minimum", "maximum", "any_minimum")
SOURCE_FIELDS = {
    "id",
    "language",
    "repository",
    "revision",
    "prefix",
    "format",
    "license",
    "terms_url",
    "text_field",
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
        mix = item.get("mix")
        if not mix or any(source not in sources for source in mix):
            raise ValueError(f"{item['name']} mixes sources missing from the catalog")
        if any(type(share) is not int or share <= 0 for share in mix.values()):
            raise ValueError("source shares must be positive integer percentages")
        if any(
            sum(
                share
                for source, share in mix.items()
                if sources[source]["language"] == language
            )
            != 50
            for language in LANGUAGES
        ):
            raise ValueError("the recipe must allocate 50 percent to each language")
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
            unscored = pool["training_tokens"] * (100 - settings["scored_share"]) // 100
            if unscored < item["training_tokens"]:
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

    Every source of a corpus needs validation tokens in the same proportion to
    its training tokens, so a slightly larger fraction than that proportion
    fills the validation split just before training, without reading far past
    either budget.
    """
    for settings in (item, recipe["corpus"]):
        if "validation_fraction" in settings:
            return settings["validation_fraction"]
    ratio = 2 * validation_budget(recipe, item) / item["training_tokens"]
    return min(0.25, 1.25 * ratio / (1 + ratio))


def source_targets(recipe, item, share):
    return {
        "train": item["training_tokens"] * share // 100,
        "validation": validation_budget(recipe, item) * share // 50,
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


def recipe_fingerprint(path):
    recipe = read_recipe(path)
    return hashlib.sha256(
        json.dumps(
            {
                "recipe": recipe,
                "training_config_sha256": file_digest(
                    Path(path).resolve().parent / recipe["training_config"]
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


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

    def benchmark_command(checkpoint, selected):
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
        commands.append(benchmark_command(previous / "best.pt", tasks))
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
            checkpoint = runs[item["name"]] / "best.pt"
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
            commands.append(benchmark_command(checkpoint, tasks))
    if phases and "sft" in recipe:
        pretrained = previous / "best.pt"
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
                instruction_output / "best.pt",
                [*tasks, *recipe["evaluation"]["instruction_tasks"]],
            ),
        ]
    sources = sources_by_id(recipe)
    return {
        "recipe_fingerprint": recipe_fingerprint(path),
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
            item["training_tokens"] + 2 * validation_budget(recipe, item)
            for item in recipe["corpora"]
        ),
        "checkpoint_bytes_approximate": parameters * 12,
        "evaluation": recipe["evaluation"],
        "sft": recipe.get("sft"),
        "commands": commands,
    }


def source_columns(source):
    fields = {source.get("text_field", "text"), "id", "url"}
    for name in FILTERS:
        fields.update(key.split(".")[0] for key in source.get(name, {}))
    return sorted(fields)


def read_json_lines(handle, compression):
    if compression == "jsonl.zst":
        import zstandard

        stream = zstandard.ZstdDecompressor().stream_reader(handle)
    else:
        stream = gzip.GzipFile(fileobj=handle)
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def iter_source(source, seed, part=None):
    from huggingface_hub import HfApi, HfFileSystem
    from pyarrow import parquet

    info = HfApi(token=False).dataset_info(
        source["repository"], revision=source["revision"]
    )
    if info.sha != source["revision"]:
        raise ValueError("the resolved dataset revision does not match the recipe")
    form = source.get("format", "parquet")
    excluded = set(source.get("exclude_files", ()))
    shards = sorted(
        item.rfilename
        for item in info.siblings
        if item.rfilename.startswith(source["prefix"])
        and item.rfilename.endswith(SOURCE_FORMATS[form])
        and item.rfilename not in excluded
    )
    if not shards:
        raise ValueError(f"no {form} files found for {source['id']}")
    random.Random(f"{seed}:{source['id']}").shuffle(shards)
    if part is not None:
        if part["count"] > len(shards):
            raise ValueError(f"{source['id']} has fewer files than parts")
        shards = shards[part["index"] :: part["count"]]
    columns = source_columns(source)
    filesystem = HfFileSystem(token=False)
    for shard in shards:
        generator = random.Random(f"{seed}:{source['id']}:{shard}")
        # Large reads keep the requests few, well within the Hub's rate limits
        # when many parts read at once.
        with filesystem.open(
            f"datasets/{source['repository']}@{source['revision']}/{shard}",
            "rb",
            block_size=16 * 1024 * 1024,
        ) as handle:
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


def document_text(row, source, corpus):
    text = row.get(source.get("text_field", "text"))
    if not isinstance(text, str):
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


class BenchmarkFilter:
    def __init__(self, path):
        import ahocorasick

        self.automata = {}
        for language, keys in benchmark_keys(path).items():
            automaton = ahocorasick.Automaton()
            for key in keys:
                automaton.add_word(key, True)
            if keys:
                automaton.make_automaton()
                self.automata[language] = automaton

    def matches(self, text, language):
        automaton = self.automata.get(language)
        return (
            automaton is not None
            and next(automaton.iter(contamination_text(text)), None) is not None
        )


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


def accepted_documents(
    source, recipe, benchmarks, database, stats, training_fraction=None, part=None
):
    """Yield new clean documents; with training_fraction, only training ones."""
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
            "recipe_fingerprint": recipe_fingerprint(recipe_path),
            "benchmarks_sha256": file_digest(Path(benchmarks_path) / "sources.json"),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "reused": samples is None,
            "samples": samples,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


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
    for source_id, share in first["mix"].items():
        source = sources[source_id]
        stats = Counter()
        language = source["language"]
        target = recipe["tokenizer"]["bytes_per_language"] * share // 50
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
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=recipe["tokenizer"]["vocabulary_size"],
        min_frequency=2,
        show_progress=False,
        special_tokens=[BOS, EOS],
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
    The split follows only the recipe, so any number of workers gives the same
    corpora. Parts read PART_SLACK beyond their share, so documents found
    earlier in another part or source can be dropped while merging.
    """
    demands = {}
    for item in streamed(recipe):
        for source_id, share in item["mix"].items():
            demands.setdefault(source_id, {})[item["name"]] = source_targets(
                recipe, item, share
            )
    parts = []
    for source_id, wanted in demands.items():
        total = sum(sum(targets.values()) for targets in wanted.values())
        count = max(1, math.ceil(total / recipe["corpus"]["part_tokens"]))
        for index in range(count):
            parts.append(
                {
                    "source": source_id,
                    "index": index,
                    "count": count,
                    "targets": {
                        name: {
                            split: math.ceil(target * (1 + PART_SLACK) / count)
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
    summary = {}
    with closing(seen_documents()) as database:
        stream = accepted_documents(
            source, recipe, benchmarks, database, scanned, part=part
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
    (output / "statistics.json").write_text(json.dumps(summary) + "\n")
    print(
        json.dumps({"part": f"{part['source']}-{part['index']}", **summary}), flush=True
    )
    return part


def prepare_corpus(
    recipe_path, tokenizer_directory, workdir, benchmarks_path, workers=1
):
    """Write every corpus read from the sources; no document appears in two of them.

    Corpora that select from a pool are written afterwards by the select stage.
    """
    recipe = read_recipe(recipe_path)
    tokenizer_directory = Path(tokenizer_directory)
    tokenizer_manifest = json.loads((tokenizer_directory / "manifest.json").read_text())
    benchmark_digest = file_digest(Path(benchmarks_path) / "sources.json")
    fingerprint = recipe_fingerprint(recipe_path)
    if (
        tokenizer_manifest["recipe_fingerprint"] != fingerprint
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
    Path(workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workdir) as scratch:
        arguments = [
            (recipe_path, tokenizer_directory, benchmarks_path, part, Path(scratch))
            for part in parts
        ]
        if workers == 1:
            for argument in arguments:
                prepare_part(*argument)
        else:
            context = multiprocessing.get_context("fork")
            with context.Pool(workers, maxtasksperchild=1) as pool:
                for _ in pool.imap_unordered(run_part, arguments):
                    pass
        manifests = {}
        with closing(seen_documents(scratch)) as database:
            for item in streamed(recipe):
                manifests[item["name"]] = write_corpus(
                    recipe,
                    item,
                    parts,
                    Path(scratch),
                    tokenizer,
                    tokenizer_directory,
                    tokenizer_manifest,
                    benchmark_digest,
                    fingerprint,
                    database,
                    paths[item["name"]],
                )
    return manifests


def run_part(argument):
    return prepare_part(*argument)


def read_json_documents(path):
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def read_piece(path):
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=TOKEN_PIECE_DTYPE)
    return np.memmap(path, mode="r", dtype=TOKEN_PIECE_DTYPE)


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
    output,
):
    """Join the parts' pieces of one corpus in recipe order, dropping repeats."""
    dtype = np.dtype(TOKEN_PIECE_DTYPE)
    sources = sources_by_id(recipe)
    with output_directory(output) as staging, ExitStack() as stack:
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
            targets = source_targets(recipe, item, share)
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
            if any(
                stats[f"{split}_tokens"] < target for split, target in targets.items()
            ):
                raise ValueError(
                    f"{source_id} is short of its clean corpus budget; do not repeat or pad the source"
                )
            used[source_id] = {
                **source,
                "share": share,
                "targets": targets,
                "statistics": dict(stats),
                "shards": sorted(shards),
            }
            print(
                json.dumps(
                    {"corpus": item["name"], "source": source_id, **dict(stats)}
                ),
                flush=True,
            )
        stack.close()
        shutil.copyfile(
            tokenizer_directory / "tokenizer.json", staging / "tokenizer.json"
        )
        manifest = {
            "version": 1,
            "dtype": dtype.str,
            "seed": recipe["seed"],
            "corpus": item["name"],
            "recipe_fingerprint": fingerprint,
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
    fingerprint = recipe_fingerprint(recipe_path)
    sources = sources_by_id(recipe)
    result = {}
    for item in recipe["corpora"]:
        data = corpus_paths(recipe, workdir)[item["name"]]
        corpus = Corpus(data)
        corpus.validate_context(model.context)
        manifest = corpus.manifest
        if (
            manifest.get("recipe_fingerprint") != fingerprint
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
        for source_id, share in item["mix"].items():
            saved = manifest["sources"][source_id]
            targets = source_targets(recipe, item, share)
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
    return {"ready": True, "recipe_fingerprint": fingerprint, "corpora": result}


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
        tokenizer_manifest["recipe_fingerprint"] != recipe_fingerprint(recipe_path)
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
