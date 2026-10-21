"""Choose the pretraining documents a model learns most from, as CoLoR-Filter does.

A document scores the loss the prior model loses on it once the model is tuned
on the target data (Brandfonbrener et al., 2024, arXiv:2406.10670), so the
documents the tuning made most predictable rank first. Scoring a document with
two models costs about as much as training on it, so only a share of the pool
is scored. A linear classifier over token unigrams and hashed bigrams, the form
of fastText, learns the extremes of those scores and ranks the rest of the pool
cheaply, as DCLM and PreSelect rank theirs (arXiv:2406.11794, arXiv:2503.00808).
"""

import json
import random
import shutil
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import LANGUAGES, Corpus, file_digest
from .recipe import (
    corpus_paths,
    output_directory,
    read_json_documents,
    read_recipe,
    recipe_fingerprint,
    source_targets,
    sources_by_id,
)
from .training import autocast, load_model, select_precision

BUCKETS = 1 << 20
# Shares of the scored documents kept out of the classifier's examples: one to
# report how well it ranks, one to choose its learning rate.
HELD_OUT_SHARE = 20
TUNING_SHARE = 20
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03)
EPOCHS = 4


def in_share(digest, share, part):
    """Whether a document falls in a percentage of the pool, by one part of its digest.

    The first eight bytes choose the held-out split, so each later part gives a
    choice independent of that split and of the other parts.
    """
    value = int.from_bytes(bytes.fromhex(digest)[8 * part : 8 * part + 8], "big")
    return value < share * 2**64 // 100


def pool_documents(pool):
    """The pool's training documents by language, with their offsets in its streams."""
    manifest = pool.manifest
    languages = {
        name: source["language"] for name, source in manifest["sources"].items()
    }
    documents = {language: [] for language in LANGUAGES}
    offsets = Counter()
    for row in read_json_documents(pool.path / manifest["provenance"]["filename"]):
        if row["split"] == "train":
            language = languages[row["source"]]
            documents[language].append({**row, "start": offsets[language]})
            offsets[language] += row["tokens"]
    for language in LANGUAGES:
        if offsets[language] != len(pool.streams["train"][language]):
            raise ValueError("the pool's provenance does not match its training tokens")
    return documents


@torch.inference_mode()
def document_losses(model, tokens, starts, device, precision="float32", batch_size=8):
    """The mean loss of each document's tokens after its first.

    tokens holds the documents one after another from the given starts. The
    windows follow one another from the start, as in evaluate_windows, so the
    first predictions of a document see the end of the one before it, as they
    do in training.
    """
    context = model.config.context
    predicted = len(tokens) - 1
    begins = torch.as_tensor(np.asarray(starts, dtype=np.int64), device=device)
    totals = torch.zeros(len(begins), dtype=torch.float64, device=device)
    counts = torch.zeros(len(begins), dtype=torch.float64, device=device)
    full = predicted // context
    batches = [
        (list(range(first, min(first + batch_size, full))), context)
        for first in range(0, full, batch_size)
    ]
    if predicted % context:
        batches.append(([full], predicted % context))
    was_training = model.training
    model.eval()
    try:
        for windows, length in batches:
            x = torch.from_numpy(
                np.stack(
                    [
                        np.asarray(
                            tokens[window * context : window * context + length + 1],
                            dtype=np.int64,
                        )
                        for window in windows
                    ]
                )
            ).to(device)
            with autocast(device, precision):
                logits = model(x[:, :-1])
            losses = F.cross_entropy(
                logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="none"
            )
            positions = (
                torch.as_tensor(windows, device=device)[:, None] * context
                + 1
                + torch.arange(length, device=device)
            ).flatten()
            owners = torch.searchsorted(begins, positions, right=True) - 1
            counted = positions != begins[owners]
            owners = owners[counted]
            totals.index_add_(0, owners, losses[counted].double())
            counts.index_add_(0, owners, torch.ones_like(owners, dtype=torch.float64))
    finally:
        model.train(was_training)
    return (totals / counts).cpu().numpy()


class NgramClassifier(torch.nn.Module):
    """A linear classifier over a document's token unigrams and hashed bigrams."""

    def __init__(self, vocabulary_size, buckets=BUCKETS):
        super().__init__()
        if buckets < 2 or buckets & (buckets - 1):
            raise ValueError("the bigram buckets must be a power of two")
        self.vocabulary_size = vocabulary_size
        self.buckets = buckets
        self.weights = torch.nn.EmbeddingBag(vocabulary_size + buckets, 1, mode="mean")
        torch.nn.init.zeros_(self.weights.weight)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def features(self, documents):
        """The feature indices of token-ID documents and where each document starts."""
        shift = np.uint64(65 - self.buckets.bit_length())
        indices, offsets, position = [], [], 0
        for tokens in documents:
            tokens = np.asarray(tokens, dtype=np.uint64)
            pairs = tokens[:-1] * np.uint64(self.vocabulary_size) + tokens[1:]
            # Fibonacci hashing spreads the token pairs evenly over the buckets.
            bigrams = (pairs * np.uint64(0x9E3779B97F4A7C15)) >> shift
            indices.append(tokens.astype(np.int64))
            indices.append(bigrams.astype(np.int64) + self.vocabulary_size)
            offsets.append(position)
            position += 2 * len(tokens) - 1
        return torch.from_numpy(np.concatenate(indices)), torch.tensor(offsets)

    def forward(self, indices, offsets):
        return self.weights(indices, offsets).squeeze(-1) + self.bias

    @torch.inference_mode()
    def score(self, documents, device, batch_size=1024):
        scores = []
        for first in range(0, len(documents), batch_size):
            indices, offsets = self.features(documents[first : first + batch_size])
            scores.append(self(indices.to(device), offsets.to(device)).float().cpu())
        return torch.cat(scores).numpy() if scores else np.zeros(0, dtype=np.float32)


def train_classifier(
    documents, labels, vocabulary_size, device, learning_rate, epochs=EPOCHS, seed=0
):
    """Fit the classifier to tell documents labelled true from the others."""
    torch.manual_seed(seed)
    classifier = NgramClassifier(vocabulary_size).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    targets = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        order = generator.permutation(len(documents))
        for first in range(0, len(order), 64):
            chosen = order[first : first + 64]
            indices, offsets = classifier.features([documents[i] for i in chosen])
            loss = F.binary_cross_entropy_with_logits(
                classifier(indices.to(device), offsets.to(device)),
                targets[chosen].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    classifier.eval()
    return classifier


def rank_correlation(first, second):
    """Spearman's rank correlation of two equally long sequences."""
    if len(first) < 2:
        return float("nan")
    ranks = [
        np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(np.float64)
        for values in (first, second)
    ]
    return float(np.corrcoef(*ranks)[0, 1])


def top_overlap(predicted, actual, share):
    """The fraction of the predicted top share that is in the actual top share."""
    count = max(1, len(actual) * share // 100)
    order = [np.argsort(-np.asarray(values), kind="stable")[:count] for values in (predicted, actual)]
    return len(set(order[0]) & set(order[1])) / count


def rank_pool(pool, documents, settings, device, output):
    """Score part of the pool with both models, then rank the rest with a classifier.

    Returns, for each language, the unscored documents and their classifier
    scores, with a summary of how well the classifier ranks the held-out
    scored documents and what the ranking cost.
    """
    precision = select_precision(device)
    streams = pool.streams["train"]

    def tokens(document, language):
        start = document["start"]
        return streams[language][start : start + document["tokens"]]

    scored = {
        language: [
            item
            for item in documents[language]
            if in_share(item["digest"], settings["scored_share"], 1)
        ]
        for language in LANGUAGES
    }
    started = time.perf_counter()
    losses = {}
    # One model at a time is in memory.
    for role in ("prior", "conditional"):
        model, tokenizer, _ = load_model(settings[role]["checkpoint"], device)
        if tokenizer.to_str() != pool.tokenizer.to_str():
            raise ValueError(f"the {role} model needs the pool's tokenizer")
        for language in LANGUAGES:
            chosen = scored[language]
            if not chosen:
                raise ValueError(f"the pool has no scored {language} documents")
            stream = np.concatenate([tokens(item, language) for item in chosen])
            starts = np.cumsum([0, *(item["tokens"] for item in chosen[:-1])])
            losses[role, language] = document_losses(
                model, stream, starts, device, precision
            )
        del model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - started
    started = time.perf_counter()
    summary = {"languages": {}}
    ranked = {}
    classifiers = {}
    with output_directory(output) as staging:
        with (staging / "scores.jsonl").open("w", encoding="utf-8") as destination:
            for language in LANGUAGES:
                chosen = scored[language]
                scores = losses["prior", language] - losses["conditional", language]
                held_out = np.array(
                    [in_share(item["digest"], HELD_OUT_SHARE, 2) for item in chosen]
                )
                tuning = ~held_out & np.array(
                    [in_share(item["digest"], TUNING_SHARE, 3) for item in chosen]
                )
                fitting = ~held_out & ~tuning
                low, high = np.percentile(
                    scores[~held_out],
                    [settings["label_share"], 100 - settings["label_share"]],
                )
                examples = np.flatnonzero(fitting & ((scores <= low) | (scores >= high)))
                if not len(examples) or not len(np.flatnonzero(tuning)):
                    raise ValueError(f"too few scored {language} documents to learn from")
                texts = [tokens(item, language) for item in chosen]
                trials = []
                for learning_rate in LEARNING_RATES:
                    classifier = train_classifier(
                        [texts[i] for i in examples],
                        scores[examples] >= high,
                        pool.manifest["vocabulary_size"],
                        device,
                        learning_rate,
                    )
                    predicted = classifier.score(
                        [texts[i] for i in np.flatnonzero(tuning)], device
                    )
                    correlation = rank_correlation(predicted, scores[tuning])
                    trials.append(
                        (
                            correlation if np.isfinite(correlation) else -np.inf,
                            learning_rate,
                            classifier,
                        )
                    )
                correlation, learning_rate, classifier = max(
                    trials, key=lambda trial: trial[0]
                )
                classifiers[language] = {
                    key: value.cpu() for key, value in classifier.state_dict().items()
                }
                logits = classifier.score(texts, device)
                for index, item in enumerate(chosen):
                    destination.write(
                        json.dumps(
                            {
                                "digest": item["digest"],
                                "source": item["source"],
                                "language": language,
                                "tokens": item["tokens"],
                                "prior_loss": float(losses["prior", language][index]),
                                "conditional_loss": float(
                                    losses["conditional", language][index]
                                ),
                                "score": float(scores[index]),
                                "held_out": bool(held_out[index]),
                                "classifier": float(logits[index]),
                            }
                        )
                        + "\n"
                    )
                unscored = [
                    item
                    for item in documents[language]
                    if not in_share(item["digest"], settings["scored_share"], 1)
                ]
                ranked[language] = (
                    unscored,
                    classifier.score([tokens(item, language) for item in unscored], device),
                )
                summary["languages"][language] = {
                    "scored_documents": len(chosen),
                    "scored_tokens": int(sum(item["tokens"] for item in chosen)),
                    "score": {
                        "mean": float(scores.mean()),
                        "standard_deviation": float(scores.std()),
                    },
                    "examples": int(len(examples)),
                    "learning_rate": learning_rate,
                    "tuning_rank_correlation": correlation,
                    "held_out": {
                        "documents": int(held_out.sum()),
                        "rank_correlation": rank_correlation(
                            logits[held_out], scores[held_out]
                        ),
                        "top_overlap": top_overlap(
                            logits[held_out], scores[held_out], settings["label_share"]
                        ),
                    },
                    "unscored_documents": len(unscored),
                }
        torch.save(classifiers, staging / "classifiers.pt")
        summary["compute"] = {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "precision": precision,
            "scoring_seconds": scoring_seconds,
            "scored_tokens_per_model": int(
                sum(
                    stats["scored_tokens"] for stats in summary["languages"].values()
                )
            ),
            "classifier_seconds": time.perf_counter() - started,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return ranked, summary


def choose_documents(recipe, item, ranked):
    """The digests of the unscored pool documents a corpus takes, source by source.

    Each source fills its share of the corpus in the order of the selection:
    the classifier's ranking, or a seeded shuffle for a random sample.
    """
    sources = sources_by_id(recipe)
    chosen = set()
    statistics = {}
    for source_id, share in item["mix"].items():
        unscored, scores = ranked[sources[source_id]["language"]]
        order = [index for index, row in enumerate(unscored) if row["source"] == source_id]
        if item["select"]["method"] == "learnable":
            order.sort(key=lambda index: -scores[index])
        else:
            random.Random(f"{recipe['seed']}:{item['name']}:{source_id}").shuffle(order)
        target = source_targets(recipe, item, share)["train"]
        tokens = documents = 0
        for index in order:
            if tokens >= target:
                break
            chosen.add(unscored[index]["digest"])
            tokens += unscored[index]["tokens"]
            documents += 1
        if tokens < target:
            raise ValueError(
                f"{source_id} is short of the unscored documents {item['name']} needs"
            )
        statistics[source_id] = {
            "train_tokens": tokens,
            "train_documents": documents,
            "pool_train_tokens": int(sum(unscored[i]["tokens"] for i in order)),
            "pool_train_documents": len(order),
        }
    return chosen, statistics


def write_selection(recipe, item, pool, ranked, summary, fingerprint, output):
    """Write the documents a corpus selects in pool order, with the pool's held-out split."""
    chosen, statistics = choose_documents(recipe, item, ranked)
    sources = sources_by_id(recipe)
    with output_directory(output) as staging:
        tokens = Counter()
        counts = {language: Counter() for language in LANGUAGES}
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
            offsets = Counter()
            filename = pool.manifest["provenance"]["filename"]
            for row in read_json_documents(pool.path / filename):
                split = row["split"]
                language = sources[row["source"]]["language"]
                start = offsets[split, language]
                offsets[split, language] += row["tokens"]
                if split == "train" and row["digest"] not in chosen:
                    continue
                streams[split, language].write(
                    np.asarray(
                        pool.streams[split][language][start : start + row["tokens"]]
                    ).tobytes()
                )
                provenance.write(json.dumps(row, ensure_ascii=False) + "\n")
                tokens[split, language] += row["tokens"]
                counts[language][split] += 1
        shutil.copyfile(pool.path / "tokenizer.json", staging / "tokenizer.json")
        selection = {
            "method": item["select"]["method"],
            "pool": item["select"]["pool"],
            "pool_fingerprint": pool.fingerprint,
            "scored_share": recipe["selection"]["scored_share"],
        }
        if selection["method"] == "learnable":
            selection.update(
                prior=recipe["selection"]["prior"],
                conditional=recipe["selection"]["conditional"],
                label_share=recipe["selection"]["label_share"],
                ranking=summary,
            )
        manifest = {
            **{
                key: pool.manifest[key]
                for key in (
                    "version",
                    "dtype",
                    "seed",
                    "vocabulary_size",
                    "tokenizer_sha256",
                    "tokenizer_training",
                    "benchmarks_sha256",
                )
            },
            "corpus": item["name"],
            "recipe_fingerprint": fingerprint,
            "selection": selection,
            "sources": {
                source_id: {
                    **pool.manifest["sources"][source_id],
                    "targets": source_targets(recipe, item, share),
                    "statistics": {
                        key: value
                        for key, value in pool.manifest["sources"][source_id][
                            "statistics"
                        ].items()
                        if key.startswith("validation_")
                    }
                    | statistics[source_id],
                }
                for source_id, share in item["mix"].items()
            },
            "documents": counts,
            "provenance": {
                "filename": "documents.jsonl",
                "sha256": file_digest(staging / "documents.jsonl"),
            },
            "files": {
                f"{split}_{language}.bin": {
                    "tokens": tokens[split, language],
                    "sha256": file_digest(staging / f"{split}_{language}.bin"),
                }
                for split in ("train", "validation")
                for language in LANGUAGES
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def select_corpora(recipe_path, workdir, device):
    """Rank each pool once and write every corpus that selects from it."""
    recipe = read_recipe(recipe_path)
    settings = recipe["selection"]
    for role in ("prior", "conditional"):
        if file_digest(settings[role]["checkpoint"]) != settings[role]["sha256"]:
            raise ValueError(f"the {role} checkpoint does not match its checksum")
    fingerprint = recipe_fingerprint(recipe_path)
    paths = corpus_paths(recipe, workdir)
    result = {}
    for name in sorted({item["select"]["pool"] for item in recipe["corpora"] if "select" in item}):
        pool = Corpus(paths[name])
        if (
            pool.manifest.get("recipe_fingerprint") != fingerprint
            or pool.manifest.get("corpus") != name
        ):
            raise ValueError("the pool belongs to a different recipe")
        ranked, summary = rank_pool(
            pool,
            pool_documents(pool),
            settings,
            device,
            Path(workdir) / f"selection-{name}",
        )
        corpora = {}
        for item in recipe["corpora"]:
            if item.get("select", {}).get("pool") == name:
                manifest = write_selection(
                    recipe, item, pool, ranked, summary, fingerprint, paths[item["name"]]
                )
                corpora[item["name"]] = {
                    source_id: source["statistics"]
                    for source_id, source in manifest["sources"].items()
                }
        result[name] = {"ranking": summary, "corpora": corpora}
    return result
