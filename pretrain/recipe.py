"""A reproducible bilingual corpus and training recipe, independent of its host."""

import hashlib
import itertools
import json
import math
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


def read_recipe(path):
    path = Path(path).resolve()
    recipe = tomllib.loads(path.read_text(encoding="utf-8"))
    for name in ("training_tokens", "seed"):
        if type(recipe.get(name)) is not int or recipe[name] < (name != "seed"):
            raise ValueError(
                f"{name} must be a nonnegative integer"
                if name == "seed"
                else f"{name} must be a positive integer"
            )
    tokenizer, corpus = recipe["tokenizer"], recipe["corpus"]
    if (
        type(tokenizer["vocabulary_size"]) is not int
        or not 258 <= tokenizer["vocabulary_size"] <= 65536
    ):
        raise ValueError("the vocabulary must fit in unsigned 16-bit token IDs")
    for value in (
        tokenizer["bytes_per_language"],
        corpus["validation_tokens_per_language"],
        corpus["minimum_characters"],
    ):
        if type(value) is not int or value <= 0:
            raise ValueError("corpus and tokenizer budgets must be positive integers")
    if not 0 < corpus["validation_fraction"] < 0.5:
        raise ValueError("validation_fraction must be between zero and one half")
    if (
        type(corpus["maximum_characters"]) is not int
        or corpus["maximum_characters"] < corpus["minimum_characters"]
    ):
        raise ValueError("maximum_characters must cover minimum_characters")
    sources = recipe["sources"]
    if not sources or len({source["id"] for source in sources}) != len(sources):
        raise ValueError("each corpus source needs a unique identifier")
    for source in sources:
        if source["language"] not in LANGUAGES or source["kind"] not in (
            "web",
            "wiki",
            "math",
        ):
            raise ValueError("select a supported language and source kind")
        if type(source["share"]) is not int or source["share"] <= 0:
            raise ValueError("source shares must be positive integer percentages")
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("pin each source to a full dataset commit")
        if not source.get("license") or not source.get("terms_url"):
            raise ValueError("each source needs its license and terms")
    if any(
        sum(source["share"] for source in sources if source["language"] == language)
        != 50
        for language in LANGUAGES
    ):
        raise ValueError("the recipe must allocate 50 percent to each language")
    config_path = path.parent / recipe["training_config"]
    model, training = read_config(config_path)
    if recipe["seed"] != training.seed:
        raise ValueError("use the same seed for the recipe and training")
    tokens_per_step = training.batch_size * training.accumulation_steps * model.context
    if training.steps != math.ceil(recipe["training_tokens"] / tokens_per_step):
        raise ValueError(
            "training steps must cover the recipe's token budget without shortening it"
        )
    if recipe["evaluation"]["framework"] != "lm-eval":
        raise ValueError("evaluation uses the standard lm-evaluation-harness tasks")
    if type(recipe["sft"]["epochs"]) is not int or recipe["sft"]["epochs"] <= 0:
        raise ValueError("SFT epochs must be positive")
    return recipe


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
    pretraining_output = Path("runs") / Path(path).stem
    instruction_output = pretraining_output.with_name(
        Path(path).stem.replace("pretrain-", "instructions-", 1)
        if Path(path).stem.startswith("pretrain-")
        else "instructions-" + Path(path).stem
    )
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
            for stage in ("tokenizer", "corpus", "validate")
        ],
        [
            "python",
            "-m",
            "pretrain",
            "train",
            "--config",
            str(Path(path).parent / recipe["training_config"]),
            "--data",
            str(workdir / "corpus"),
            "--output",
            str(pretraining_output),
            "--device",
            "cuda",
            "--precision",
            "bfloat16",
        ],
        [
            "python",
            "-m",
            "pretrain",
            "benchmark",
            "--checkpoint",
            str(pretraining_output / "best.pt"),
            "--device",
            "cuda",
            "--tasks",
            *recipe["evaluation"]["base_tasks"],
        ],
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
            str(pretraining_output / "best.pt"),
            "--device",
            "cuda",
            "--precision",
            "bfloat16",
        ],
        [
            "python",
            "-m",
            "pretrain",
            "benchmark",
            "--checkpoint",
            str(instruction_output / "best.pt"),
            "--device",
            "cuda",
            "--tasks",
            *recipe["evaluation"]["base_tasks"],
            *recipe["evaluation"]["instruction_tasks"],
        ],
    ]
    return {
        "recipe_fingerprint": recipe_fingerprint(path),
        "parameters": parameters,
        "initialization": "random_weights",
        "model": asdict(model),
        "training": asdict(training),
        "nominal_training_tokens": recipe["training_tokens"],
        "scheduled_training_tokens": training.steps
        * training.batch_size
        * training.accumulation_steps
        * model.context,
        "tokenizer": recipe["tokenizer"],
        "sources": [
            {
                **source,
                "training_tokens": recipe["training_tokens"] * source["share"] // 100,
            }
            for source in recipe["sources"]
        ],
        "packed_dtype": "<u2",
        "minimum_packed_bytes": 2
        * (
            recipe["training_tokens"]
            + 2 * recipe["corpus"]["validation_tokens_per_language"]
        ),
        "checkpoint_bytes_approximate": parameters * 12,
        "evaluation": recipe["evaluation"],
        "sft": recipe["sft"],
        "commands": commands,
    }


def iter_source(source, seed):
    from huggingface_hub import HfApi, HfFileSystem
    from pyarrow import parquet

    info = HfApi(token=False).dataset_info(
        source["repository"], revision=source["revision"]
    )
    if info.sha != source["revision"]:
        raise ValueError("the resolved dataset revision does not match the recipe")
    shards = sorted(
        item.rfilename
        for item in info.siblings
        if item.rfilename.startswith(source["prefix"])
        and item.rfilename.endswith(".parquet")
    )
    if not shards:
        raise ValueError(f"no Parquet shards found for {source['id']}")
    random.Random(f"{seed}:{source['id']}").shuffle(shards)
    columns = ["text", "url"]
    if source["kind"] == "web":
        columns += ["language", "language_score", source["quality_field"]]
    elif source["kind"] == "wiki":
        columns += ["in_language", "id"]
    else:
        columns += ["language", "language_score"]
    filesystem = HfFileSystem(token=False)
    for shard in shards:
        with (
            filesystem.open(
                f"datasets/{source['repository']}@{source['revision']}/{shard}",
                "rb",
                block_size=1024 * 1024,
            ) as handle,
            parquet.ParquetFile(handle) as table,
        ):
            generator = random.Random(f"{seed}:{source['id']}:{shard}")
            groups = list(range(table.num_row_groups))
            generator.shuffle(groups)
            for group in groups:
                for batch in table.iter_batches(
                    batch_size=64, row_groups=[group], columns=columns
                ):
                    rows = batch.to_pylist()
                    generator.shuffle(rows)
                    for row in rows:
                        yield {**row, "source_file": shard}


def document_text(row, source, corpus):
    text = row.get("text")
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text).strip()
    if not corpus["minimum_characters"] <= len(text) <= corpus["maximum_characters"]:
        return None
    if source["kind"] in ("web", "math"):
        language = source.get("source_language", source["language"])
        score = row.get("language_score")
        if (
            row.get("language") != language
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0.8
        ):
            return None
    elif row.get("in_language") != source["language"]:
        return None
    if source["kind"] == "web":
        quality = row.get(source["quality_field"])
        if (
            not isinstance(quality, (int, float))
            or not math.isfinite(quality)
            or quality < source.get("minimum_quality", 0)
        ):
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


def accepted_documents(
    source, recipe, benchmarks, database, stats, training_only=False
):
    for row in iter_source(source, recipe["seed"]):
        text = document_text(row, source, recipe["corpus"])
        if text is None:
            stats["quality_rejected"] += 1
            continue
        if benchmarks.matches(text, source["language"]):
            stats["contaminated"] += 1
            continue
        digest = document_digest(text, recipe["seed"])
        split = (
            "validation"
            if validation_document(digest, recipe["corpus"]["validation_fraction"])
            else "train"
        )
        if training_only and split != "train":
            continue
        if (
            database.execute(
                "INSERT OR IGNORE INTO documents VALUES (?)", (digest,)
            ).rowcount
            == 0
        ):
            stats["duplicates"] += 1
            continue
        yield text, digest, split, row


def prepare_tokenizer(recipe_path, output, benchmarks_path):
    recipe = read_recipe(recipe_path)
    benchmarks = BenchmarkFilter(benchmarks_path)
    with output_directory(output) as staging, ExitStack() as stack:
        database = stack.enter_context(
            closing(sqlite3.connect(staging / "seen.sqlite3"))
        )
        database.execute(
            "CREATE TABLE documents (digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        destinations = {
            language: stack.enter_context(
                (staging / f"{language}.jsonl").open("w", encoding="utf-8")
            )
            for language in LANGUAGES
        }
        samples = {}
        for source in recipe["sources"]:
            stats = Counter()
            language = source["language"]
            target = recipe["tokenizer"]["bytes_per_language"] * source["share"] // 50
            for text, _, _, _ in accepted_documents(
                source, recipe, benchmarks, database, stats, training_only=True
            ):
                destinations[language].write(
                    json.dumps({"text": text}, ensure_ascii=False) + "\n"
                )
                stats["bytes"] += len(text.encode("utf-8"))
                stats["documents"] += 1
                if stats["bytes"] >= target:
                    break
            if stats["bytes"] < target:
                raise ValueError(
                    f"{source['id']} is short of the tokenizer sample budget"
                )
            samples[source["id"]] = dict(stats)
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
                *(
                    read_documents(staging / f"{language}.jsonl")
                    for language in LANGUAGES
                )
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
        manifest = {
            "version": 1,
            "recipe_fingerprint": recipe_fingerprint(recipe_path),
            "benchmarks_sha256": file_digest(Path(benchmarks_path) / "sources.json"),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "samples": samples,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        stack.close()
        for filename in ("ja.jsonl", "en.jsonl", "seen.sqlite3"):
            (staging / filename).unlink()
    return manifest


def prepare_corpus(recipe_path, tokenizer_directory, output, benchmarks_path):
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
    benchmarks = BenchmarkFilter(benchmarks_path)
    dtype = np.dtype("<u2")
    bos, eos = tokenizer.token_to_id(BOS), tokenizer.token_to_id(EOS)
    with output_directory(output) as staging, ExitStack() as stack:
        database = stack.enter_context(
            closing(sqlite3.connect(staging / "seen.sqlite3"))
        )
        database.execute(
            "CREATE TABLE documents (digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
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
        sources = {}
        for source in recipe["sources"]:
            language = source["language"]
            targets = {
                "train": recipe["training_tokens"] * source["share"] // 100,
                "validation": recipe["corpus"]["validation_tokens_per_language"]
                * source["share"]
                // 50,
            }
            stats = Counter()
            shards = set()
            documents = accepted_documents(source, recipe, benchmarks, database, stats)
            while chunk := list(itertools.islice(documents, 64)):
                encodings = tokenizer.encode_batch(
                    [item[0] for item in chunk], add_special_tokens=False
                )
                for (_, digest, split, row), encoding in zip(chunk, encodings):
                    if stats[f"{split}_tokens"] >= targets[split]:
                        continue
                    encoded = np.asarray([bos, *encoding.ids, eos], dtype=dtype)
                    streams[split, language].write(encoded.tobytes())
                    provenance.write(
                        json.dumps(
                            {
                                "digest": digest.hex(),
                                "source": source["id"],
                                "id": row.get("id"),
                                "url": row.get("url"),
                                "split": split,
                                "tokens": len(encoded),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats[f"{split}_tokens"] += len(encoded)
                    stats[f"{split}_documents"] += 1
                    tokens[split, language] += len(encoded)
                    counts[language][split] += 1
                    shards.add(row["source_file"])
                if all(
                    stats[f"{split}_tokens"] >= target
                    for split, target in targets.items()
                ):
                    break
                database.commit()
            if any(
                stats[f"{split}_tokens"] < target for split, target in targets.items()
            ):
                raise ValueError(
                    f"{source['id']} is short of its clean corpus budget; do not repeat or pad the source"
                )
            sources[source["id"]] = {
                **source,
                "targets": targets,
                "statistics": dict(stats),
                "shards": sorted(shards),
            }
            print(json.dumps({"source": source["id"], **dict(stats)}), flush=True)
        stack.close()
        (staging / "seen.sqlite3").unlink()
        shutil.copyfile(
            tokenizer_directory / "tokenizer.json", staging / "tokenizer.json"
        )
        manifest = {
            "version": 1,
            "dtype": dtype.str,
            "seed": recipe["seed"],
            "recipe_fingerprint": fingerprint,
            "vocabulary_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "tokenizer_training": tokenizer_manifest,
            "sources": sources,
            "benchmarks_sha256": benchmark_digest,
            "documents": counts,
            "duplicates_removed": sum(
                source["statistics"].get("duplicates", 0) for source in sources.values()
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


def validate_corpus(recipe_path, data):
    recipe = read_recipe(recipe_path)
    corpus = Corpus(data)
    model, _ = read_config(
        Path(recipe_path).resolve().parent / recipe["training_config"]
    )
    corpus.validate_context(model.context)
    manifest = corpus.manifest
    if manifest.get("recipe_fingerprint") != recipe_fingerprint(recipe_path):
        raise ValueError("the corpus belongs to a different recipe")
    if (
        file_digest(Path(data) / manifest["provenance"]["filename"])
        != manifest["provenance"]["sha256"]
    ):
        raise ValueError("document provenance checksum does not match")
    for source in recipe["sources"]:
        saved = manifest["sources"][source["id"]]
        targets = {
            "train": recipe["training_tokens"] * source["share"] // 100,
            "validation": recipe["corpus"]["validation_tokens_per_language"]
            * source["share"]
            // 50,
        }
        if (
            any(saved.get(field) != value for field, value in source.items())
            or saved["targets"] != targets
        ):
            raise ValueError(
                "the corpus source or token budget does not match the recipe"
            )
        for split, target in targets.items():
            if saved["statistics"][f"{split}_tokens"] < target:
                raise ValueError(f"{source['id']} is short of its {split} token budget")
    return {
        "ready": True,
        "recipe_fingerprint": manifest["recipe_fingerprint"],
        "files": manifest["files"],
        "sources": manifest["sources"],
    }


def write_sft_config(recipe_path, data, tokenizer_directory, output):
    from .instructions import InstructionCorpus

    recipe = read_recipe(recipe_path)
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
