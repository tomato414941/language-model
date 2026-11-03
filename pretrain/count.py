"""Counted continuations that COUNT adds to each token's embedding.

COUNT (Magnusson et al., arXiv 2610.20456) looks up the tokens that most often
followed each token's last few tokens in a counted corpus, and adds their
embeddings to the token's own, weighted by how often each followed. A table
holds, for every history of one to orders - 1 tokens that occurs at least
min_history times, the tokens that followed it at least min_count times, the
most frequent first, with the share of the history's occurrences each took; a
row of the most frequent tokens stands for the empty history. A history stays
within one document and may begin with the token that starts it.

A history is known by a 48-bit key made from two polynomial hashes modulo
primes below 2**31, which NumPy and PyTorch compute alike without overflow.
"""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import BOS, file_digest

PRIMES = (2_147_483_647, 2_147_483_629)
BASES = (1_000_003, 998_244_353)
# Of the 62 bits the two hashes give, the low ones are dropped so that a key
# and the token after it fit in one 64-bit number while counting.
KEY_SHIFT = 14
FOLLOWING_BITS = 16
# Counting splits the keys by their top bits so that each pass fits in memory.
BUCKET_SHIFT = 32
BUCKETS = 1 << 16
CHUNK_TOKENS = 1 << 22
# The bytes a counted pair takes while its pass sorts and ranks it, by a
# generous estimate.
BYTES_PER_PAIR = 64
# Tables count at most this many tokens to an n-gram, so a lookup reads no
# further back than this many tokens before its position.
COUNT_HISTORY = 8


def tokenizer_digest(tokenizer):
    """The digest of a tokenizer's definition, which a table records."""
    return hashlib.sha256(tokenizer.to_str().encode()).hexdigest()


def _previous(values):
    """Each position's value at the position before it, and zero at the first."""
    if isinstance(values, torch.Tensor):
        return F.pad(values[..., :-1], (1, 0))
    return np.concatenate([np.zeros_like(values[..., :1]), values[..., :-1]], axis=-1)


def history_keys(tokens, orders):
    """The key of the history of each length from 1 to orders - 1 ending at each position.

    tokens is a NumPy array or a tensor of 64-bit token ids whose last axis is
    the sequence; a position too close to the start gets a key of a shorter
    history, which the caller leaves out.
    """
    following = tokens + 1
    first = second = following * 0
    keys = []
    for _ in range(1, orders):
        first = (_previous(first) * BASES[0] + following) % PRIMES[0]
        second = (_previous(second) * BASES[1] + following) % PRIMES[1]
        keys.append(((first << 31) | second) >> KEY_SHIFT)
    return keys


def document_begins(tokens, start):
    """The position where each position's document begins, or -1 when it began before the window."""
    if isinstance(tokens, torch.Tensor):
        positions = torch.arange(tokens.shape[-1], device=tokens.device)
        return torch.cummax(torch.where(tokens == start, positions, -1), dim=-1).values
    positions = np.arange(tokens.shape[-1])
    return np.maximum.accumulate(np.where(tokens == start, positions, -1), axis=-1)


def _chunk_pairs(stream, begin, orders, start, low, high):
    """The pairs of a history and the token after it, for the histories ending in one chunk whose keys fall in [low, high) buckets."""
    first = max(0, begin - (orders - 2))
    end = min(len(stream), begin + CHUNK_TOKENS + 1)
    tokens = np.asarray(stream[first:end], dtype=np.int64)
    history, following = tokens[:-1], tokens[1:].astype(np.uint64)
    positions = np.arange(len(history))
    begins = document_begins(history, start)
    own = positions >= begin - first
    pairs = {}
    for length, key in enumerate(history_keys(history, orders), 1):
        earliest = positions - length + 1
        bucket = key >> BUCKET_SHIFT
        chosen = own & (earliest >= 0) & (begins <= earliest) & (bucket >= low) & (bucket < high)
        pairs[length] = (key[chosen].astype(np.uint64) << np.uint64(FOLLOWING_BITS)) | following[chosen]
    return pairs


def _rows(pairs, candidates, min_history, min_count):
    """The table rows of one order: keys, candidate tokens and their shares, in key order."""
    values, counts = np.unique(pairs, return_counts=True)
    histories = (values >> np.uint64(FOLLOWING_BITS)).astype(np.int64)
    following = (values & np.uint64((1 << FOLLOWING_BITS) - 1)).astype(np.int64)
    if not len(values):
        return (
            np.zeros(0, np.int64),
            np.zeros((0, candidates), np.uint16),
            np.zeros((0, candidates), np.float16),
        )
    new = np.empty(len(values), dtype=bool)
    new[0], new[1:] = True, histories[1:] != histories[:-1]
    starts = np.flatnonzero(new)
    totals = np.add.reduceat(counts, starts)
    group = np.cumsum(new) - 1
    kept = (counts >= min_count) & (totals[group] >= min_history)
    group, following, counts = group[kept], following[kept], counts[kept]
    # By history, then the most frequent first; equal counts keep token order.
    order = np.lexsort((-counts, group))
    group, following, counts = group[order], following[order], counts[order]
    first = np.ones(len(group), dtype=bool)
    first[1:] = group[1:] != group[:-1]
    place = np.arange(len(group))
    rank = place - np.maximum.accumulate(np.where(first, place, 0))
    top = rank < candidates
    group, following, counts, rank, first = group[top], following[top], counts[top], rank[top], first[top]
    row = np.cumsum(first) - 1
    held = group[first]
    ids = np.zeros((len(held), candidates), np.uint16)
    shares = np.zeros((len(held), candidates), np.float16)
    ids[row, rank] = following
    shares[row, rank] = counts / totals[group]
    return histories[starts[held]], ids, shares


def build_table(corpora, output, orders, candidates, min_history, min_count, memory_gb, threads):
    """Count the training text of prepared corpora into a table at output."""
    from .data import Corpus

    if type(orders) is not int or not 2 <= orders <= COUNT_HISTORY:
        raise ValueError(f"orders must be an integer from 2 to {COUNT_HISTORY}")
    for name, value in (("candidates", candidates), ("min_history", min_history), ("min_count", min_count), ("threads", threads)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not memory_gb > 0:
        raise ValueError("memory_gb must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("the table's directory already exists")
    opened = [Corpus(path, splits=("train",)) for path in corpora]
    if len({corpus.manifest["tokenizer_sha256"] for corpus in opened}) != 1:
        raise ValueError("the corpora must share one tokenizer")
    tokenizer = opened[0].tokenizer
    vocabulary = tokenizer.get_vocab_size()
    if vocabulary > 1 << FOLLOWING_BITS:
        raise ValueError(f"a table holds vocabularies of at most {1 << FOLLOWING_BITS} tokens")
    start = tokenizer.token_to_id(BOS)
    streams = [stream for corpus in opened for stream in corpus.streams["train"].values()]
    tasks = [(stream, begin) for stream in streams for begin in range(0, len(stream) - 1, CHUNK_TOKENS)]
    unigram = np.zeros(vocabulary, dtype=np.int64)
    pairs_total = 0
    with ThreadPoolExecutor(threads) as pool:
        for counted in pool.map(
            lambda task: np.bincount(np.asarray(task[0][task[1] + 1 : task[1] + 1 + CHUNK_TOKENS]), minlength=vocabulary),
            tasks,
        ):
            unigram += counted
        pairs_total = sum(len(stream) - 1 for stream in streams) * (orders - 1)
        passes = max(1, math.ceil(pairs_total * BYTES_PER_PAIR / (memory_gb * 2**30)))
        bounds = np.linspace(0, BUCKETS, passes + 1).round().astype(np.int64)
        rows = {length: [] for length in range(1, orders)}
        for low, high in zip(bounds[:-1], bounds[1:]):
            found = {length: [] for length in range(1, orders)}
            for pairs in pool.map(lambda task: _chunk_pairs(*task, orders, start, low, high), tasks):
                for length, part in pairs.items():
                    found[length].append(part)
            for length, part in zip(
                found,
                pool.map(
                    lambda length: _rows(np.concatenate(found[length]), candidates, min_history, min_count),
                    list(found),
                ),
            ):
                rows[length].append(part)
            del found
    common = np.argsort(-unigram, kind="stable")[:candidates]
    arrays = {
        "candidates_0": np.zeros(candidates, np.uint16),
        "shares_0": np.zeros(candidates, np.float16),
    }
    arrays["candidates_0"][: len(common)] = common
    arrays["shares_0"][: len(common)] = unigram[common] / max(unigram.sum(), 1)
    for length, parts in rows.items():
        arrays[f"keys_{length}"] = np.concatenate([part[0] for part in parts])
        arrays[f"candidates_{length}"] = np.concatenate([part[1] for part in parts])
        arrays[f"shares_{length}"] = np.concatenate([part[2] for part in parts])
    output.mkdir(parents=True)
    files = {}
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
        files[f"{name}.npy"] = file_digest(output / f"{name}.npy")
    manifest = {
        "version": 1,
        "orders": orders,
        "candidates": candidates,
        "min_history": min_history,
        "min_count": min_count,
        "document_start": start,
        "vocabulary_size": vocabulary,
        "tokenizer": tokenizer_digest(tokenizer),
        "hash": {"primes": list(PRIMES), "bases": list(BASES), "key_shift": KEY_SHIFT},
        "counted": [
            {"corpus": corpus.path.name, "file": f"train_{language}.bin", **corpus.manifest["files"][f"train_{language}.bin"]}
            for corpus in opened
            for language in corpus.streams["train"]
        ],
        "rows": {str(length): len(arrays[f"keys_{length}"]) for length in range(1, orders)},
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"table": str(output), "digest": CountTable.load(output).digest, "rows": manifest["rows"]}


class CountTable:
    """A built table: its manifest, and its arrays as tensors ready for the model's device."""

    def __init__(self, manifest, tensors):
        self.manifest = manifest
        self.tensors = tensors
        self.digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise ValueError("unsupported count table")
        if manifest["hash"] != {"primes": list(PRIMES), "bases": list(BASES), "key_shift": KEY_SHIFT}:
            raise ValueError("the count table was keyed by another hash")
        token = np.int16 if manifest["vocabulary_size"] <= 1 << 15 else np.int32
        tensors = {}
        for name, digest in manifest["files"].items():
            if file_digest(directory / name) != digest:
                raise ValueError(f"count table checksum does not match: {name}")
            array = np.load(directory / name)
            if name.startswith("candidates_"):
                array = array.astype(token)
            tensors[name.removesuffix(".npy")] = torch.from_numpy(array)
        return cls(manifest, tensors)

    def check_tokenizer(self, tokenizer):
        if self.manifest["tokenizer"] != tokenizer_digest(tokenizer):
            raise ValueError("the count table was counted with another tokenizer")

    def checkpoint(self):
        """The table as a checkpoint keeps it."""
        return {"manifest": json.dumps(self.manifest, sort_keys=True), "tensors": self.tensors}

    @classmethod
    def from_checkpoint(cls, saved):
        return cls(json.loads(saved["manifest"]), dict(saved["tensors"]))


class CountInput(nn.Module):
    """The share-weighted embeddings of each position's counted continuations."""

    def __init__(self, table):
        super().__init__()
        self.orders = table.manifest["orders"]
        self.document_start = table.manifest["document_start"]
        # The table goes wherever the model goes, but is no part of its weights.
        for name, tensor in table.tensors.items():
            self.register_buffer(name, tensor, persistent=False)

    def lookup(self, tokens):
        """Each position's candidates and weights, on a last axis of orders times candidates."""
        positions = torch.arange(tokens.shape[-1], device=tokens.device)
        begins = document_begins(tokens, self.document_start)
        shape = (*tokens.shape, self.candidates_0.shape[-1])
        ids, shares = [self.candidates_0.expand(shape)], [self.shares_0.expand(shape)]
        for length, key in enumerate(history_keys(tokens, self.orders), 1):
            keys = getattr(self, f"keys_{length}")
            if not len(keys):
                ids.append(self.candidates_0.expand(shape))
                shares.append(torch.zeros_like(self.shares_0).expand(shape))
                continue
            row = torch.searchsorted(keys, key).clamp_(max=len(keys) - 1)
            earliest = positions - length + 1
            found = (earliest >= 0) & (begins <= earliest) & (keys[row] == key)
            ids.append(getattr(self, f"candidates_{length}")[row])
            shares.append(getattr(self, f"shares_{length}")[row] * found.unsqueeze(-1))
        # Each order's row counts once, an order without a row as nothing.
        return torch.cat(ids, -1), torch.cat(shares, -1) / self.orders

    def forward(self, tokens, weight):
        ids, shares = self.lookup(tokens)
        gathered = F.embedding_bag(
            ids.flatten(0, -2).long(),
            weight,
            per_sample_weights=shares.flatten(0, -2).to(weight.dtype),
            mode="sum",
        )
        return gathered.view(*tokens.shape, weight.shape[-1])
