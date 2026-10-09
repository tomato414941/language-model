import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

LANGUAGES = ("ja", "en")
BOS, EOS = "<|bos|>", "<|eos|>"
TOKEN_DTYPE = np.dtype("<u4")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_documents(path):
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        if path.suffix == ".jsonl":
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                document = json.loads(line)
                if not isinstance(document, dict) or not isinstance(
                    document.get("text"), str
                ):
                    raise TypeError(
                        f"{path}:{number}: each JSON object needs a text string"
                    )
                yield document["text"]
        else:
            paragraph = []
            for line in source:
                if line.strip():
                    paragraph.append(line.rstrip("\r\n"))
                elif paragraph:
                    yield "\n".join(paragraph)
                    paragraph = []
            if paragraph:
                yield "\n".join(paragraph)


def prepare_data(
    japanese,
    english,
    output,
    vocabulary_size=16384,
    validation_fraction=0.05,
    seed=42,
    provenance=None,
    tokenizer_bytes_per_language=8 * 1024 * 1024,
    tokenizer_path=None,
):
    if type(vocabulary_size) is not int or not 258 <= vocabulary_size <= 2**32 - 1:
        raise ValueError("vocabulary_size must be an integer of at least 258")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if (
        type(tokenizer_bytes_per_language) is not int
        or tokenizer_bytes_per_language < 4
    ):
        raise ValueError(
            "tokenizer_bytes_per_language must be an integer of at least four"
        )
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; select a new dataset directory")
    inputs = {
        language: {"filename": Path(path).name, "sha256": file_digest(path)}
        for language, path in (("ja", japanese), ("en", english))
    }
    source_manifest = None
    if provenance is not None:
        source_manifest = json.loads(Path(provenance).read_text(encoding="utf-8"))
        for source in inputs.values():
            metadata = source_manifest.get("files", {}).get(source["filename"], {})
            if metadata.get("sha256") != source["sha256"]:
                raise ValueError("source checksum does not match the provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    # Tokenizers consumes the iterator on a worker thread. Database operations
    # remain sequential: ingestion, tokenizer training, then token-file encoding.
    connection = sqlite3.connect(staging / "documents.sqlite3", check_same_thread=False)
    try:
        connection.execute(
            "CREATE TABLE documents (digest TEXT PRIMARY KEY, text TEXT, language TEXT, split TEXT)"
        )
        duplicates = 0
        for language, source in (("ja", japanese), ("en", english)):
            for document in read_documents(source):
                text = unicodedata.normalize("NFC", document).strip()
                if not text:
                    continue
                normalized = " ".join(text.split())
                digest = hashlib.sha256(f"{seed}\0{normalized}".encode()).hexdigest()
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, 'train')",
                    (digest, text, language),
                )
                duplicates += cursor.rowcount == 0
        connection.commit()
        connection.execute(
            "CREATE INDEX language_split ON documents(language, split, digest)"
        )
        counts = {}
        for language in LANGUAGES:
            count = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE language = ?", (language,)
            ).fetchone()[0]
            if count < 2:
                raise ValueError(
                    f"{language} needs at least two distinct, nonempty documents"
                )
            held_out = min(count - 1, max(1, int(count * validation_fraction)))
            connection.execute(
                "UPDATE documents SET split = 'validation' WHERE digest IN "
                "(SELECT digest FROM documents WHERE language = ? ORDER BY digest LIMIT ?)",
                (language, held_out),
            )
            counts[language] = {"train": count - held_out, "validation": held_out}
        connection.commit()

        def documents(language, split):
            return (
                row[0]
                for row in connection.execute(
                    "SELECT text FROM documents WHERE language = ? AND split = ? ORDER BY digest",
                    (language, split),
                )
            )

        # Train our own byte BPE on training documents only. Every byte has a token,
        # so unseen Japanese/English text can still be encoded without an UNK token.
        tokenizer = (
            Tokenizer.from_file(str(tokenizer_path))
            if tokenizer_path is not None
            else Tokenizer(models.BPE())
        )
        if tokenizer_path is None:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
        elif tokenizer.get_vocab_size() != vocabulary_size or any(
            tokenizer.token_to_id(token) is None for token in (BOS, EOS)
        ):
            raise ValueError(
                "the reused tokenizer must match the vocabulary and BOS/EOS tokens"
            )
        trainer = trainers.BpeTrainer(
            vocab_size=vocabulary_size,
            min_frequency=2,
            show_progress=False,
            special_tokens=[BOS, EOS],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        per_language = min(
            25000, *(counts[language]["train"] for language in LANGUAGES)
        )
        tokenizer_sample = {
            language: {"documents": 0, "bytes": 0} for language in LANGUAGES
        }

        def tokenizer_documents(language):
            stats = tokenizer_sample[language]
            for document in itertools.islice(
                documents(language, "train"), per_language
            ):
                remaining = tokenizer_bytes_per_language - stats["bytes"]
                if remaining < 4:
                    break
                text = document.encode("utf-8")[:remaining].decode(
                    "utf-8", errors="ignore"
                )
                stats["documents"] += 1
                stats["bytes"] += len(text.encode("utf-8"))
                yield text

        balanced = itertools.chain.from_iterable(
            itertools.zip_longest(tokenizer_documents("ja"), tokenizer_documents("en"))
        )
        if tokenizer_path is None:
            tokenizer.train_from_iterator(
                (text for text in balanced if text is not None), trainer=trainer
            )
            tokenizer.save(str(staging / "tokenizer.json"))
            tokenizer_metadata = {
                "max_bytes_per_language": tokenizer_bytes_per_language,
                "samples": tokenizer_sample,
            }
        else:
            shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
            tokenizer_metadata = {
                "reused": True,
                "source_filename": Path(tokenizer_path).name,
                "source_sha256": file_digest(tokenizer_path),
            }
        manifest = {
            "version": 1,
            "dtype": TOKEN_DTYPE.str,
            "seed": seed,
            "vocabulary_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": file_digest(staging / "tokenizer.json"),
            "tokenizer_training": tokenizer_metadata,
            "duplicates_removed": duplicates,
            "inputs": inputs,
            "provenance": source_manifest,
            "documents": counts,
            "files": {},
        }
        bos, eos = tokenizer.token_to_id(BOS), tokenizer.token_to_id(EOS)
        for split in ("train", "validation"):
            for language in LANGUAGES:
                filename = f"{split}_{language}.bin"
                token_count = 0
                with (staging / filename).open("wb") as destination:
                    source = documents(language, split)
                    while batch := list(itertools.islice(source, 256)):
                        for encoding in tokenizer.encode_batch(
                            batch, add_special_tokens=False
                        ):
                            tokens = np.asarray(
                                [bos, *encoding.ids, eos], dtype=TOKEN_DTYPE
                            )
                            destination.write(tokens.tobytes())
                            token_count += len(tokens)
                manifest["files"][filename] = {
                    "tokens": token_count,
                    "sha256": file_digest(staging / filename),
                }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        connection.close()
        (staging / "documents.sqlite3").unlink()
        os.rename(staging, output)
        return manifest
    finally:
        connection.close()
        if staging.exists():
            shutil.rmtree(staging)


class Corpus:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("version") != 1
            or self.manifest.get("dtype") != TOKEN_DTYPE.str
        ):
            raise ValueError("unsupported dataset format")
        tokenizer_path = self.path / "tokenizer.json"
        if file_digest(tokenizer_path) != self.manifest["tokenizer_sha256"]:
            raise ValueError("dataset tokenizer checksum does not match")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if self.tokenizer.get_vocab_size() != self.manifest["vocabulary_size"]:
            raise ValueError("dataset vocabulary does not match the tokenizer")
        self.streams = {}
        for split in ("train", "validation"):
            self.streams[split] = {}
            for language in LANGUAGES:
                filename = f"{split}_{language}.bin"
                metadata = self.manifest["files"][filename]
                path = self.path / filename
                if path.stat().st_size != metadata["tokens"] * TOKEN_DTYPE.itemsize:
                    raise ValueError(f"dataset token count does not match: {filename}")
                if file_digest(path) != metadata["sha256"]:
                    raise ValueError(f"dataset checksum does not match: {filename}")
                self.streams[split][language] = np.memmap(
                    path, mode="r", dtype=TOKEN_DTYPE
                )
        self.fingerprint = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()

    def validate_context(self, context):
        for split, streams in self.streams.items():
            for language, stream in streams.items():
                if len(stream) <= context:
                    raise ValueError(
                        f"{split}/{language} needs more than {context} tokens; add documents or shorten context"
                    )

    def batch(self, split, batch_size, context, generator, device, language=None):
        languages = (
            [language] * batch_size
            if language
            else [
                LANGUAGES[i]
                for i in torch.randint(
                    len(LANGUAGES), (batch_size,), generator=generator
                ).tolist()
            ]
        )
        windows = []
        for selected in languages:
            stream = self.streams[split][selected]
            start = torch.randint(
                len(stream) - context, (1,), generator=generator
            ).item()
            windows.append(
                np.asarray(stream[start : start + context + 1], dtype=np.int64)
            )
        tokens = torch.from_numpy(np.stack(windows))
        if device.type == "cuda":
            tokens = tokens.pin_memory()
        tokens = tokens.to(device, non_blocking=device.type == "cuda")
        return tokens[:, :-1], tokens[:, 1:]
