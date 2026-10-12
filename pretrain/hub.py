import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath


def _json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        raise ValueError(
            "run ID must be a single name containing letters, numbers, dots, underscores or hyphens"
        )
    return value


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _checkpoint(path):
    import torch
    from tokenizers import Tokenizer

    from .model import ModelConfig
    from .training import TrainConfig

    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or saved.get("version") != 1:
        raise ValueError("use a version-one pretraining checkpoint")
    ModelConfig(**saved["model_config"])
    TrainConfig(**saved["train_config"])
    tokenizer = Tokenizer.from_str(saved["tokenizer"])
    if any(tokenizer.token_to_id(token) is None for token in ("<|bos|>", "<|eos|>")):
        raise ValueError("checkpoint tokenizer must contain BOS and EOS tokens")
    if not isinstance(saved["model"], dict) or not saved["model"]:
        raise ValueError("checkpoint must contain model weights")
    return saved


def _resumable(saved):
    return {"optimizer", "scaler", "sampling_rng", "torch_rng", "cuda_rng"}.issubset(
        saved
    )


def prepare_checkpoint(
    checkpoint, run_id, resume_checkpoint=None, summary=None, source_commit=None
):
    run_id = _run_id(run_id)
    checkpoint = Path(checkpoint).resolve(strict=True)
    saved = _checkpoint(checkpoint)
    prefix = "runs/" + run_id + "/"
    files = {prefix + "model.pt": checkpoint}
    resume_name = "model.pt" if _resumable(saved) else None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint).resolve(strict=True)
        latest = (
            saved if resume_checkpoint == checkpoint else _checkpoint(resume_checkpoint)
        )
        if not _resumable(latest):
            raise ValueError(
                "resume checkpoint must contain optimizer and random-number states"
            )
        for field in (
            "model_config",
            "train_config",
            "dataset_fingerprint",
            "tokenizer",
        ):
            if latest[field] != saved[field]:
                raise ValueError(
                    "model and resume checkpoint must belong to the same training run"
                )
        if resume_checkpoint != checkpoint:
            files[prefix + "last.pt"] = resume_checkpoint
            resume_name = "last.pt"
    files[prefix + "tokenizer.json"] = saved["tokenizer"].encode()
    files[prefix + "config.json"] = _json(
        {"model": saved["model_config"], "training": saved["train_config"]}
    )
    if summary is not None:
        files[prefix + "summary.json"] = _json(json.loads(Path(summary).read_text()))
    source = Path(__file__).parent
    for name in ("__init__.py", "model.py", "data.py", "training.py"):
        files[prefix + "source/pretrain/" + name] = source / name
    files[prefix + "source/pyproject.toml"] = source.parent / "pyproject.toml"
    manifest = {
        "version": 1,
        "run_id": run_id,
        "format": "pretrain-pytorch-v1",
        "checkpoint": prefix + "model.pt",
        "resume_checkpoint": prefix + resume_name if resume_name else None,
        "step": saved["step"],
        "tokens_seen": saved["tokens_seen"],
        "dataset_fingerprint": saved["dataset_fingerprint"],
        "source_commit": source_commit,
        "files": {
            name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            if isinstance(value, bytes)
            else {"sha256": _digest(value), "bytes": value.stat().st_size}
            for name, value in files.items()
        },
    }
    files[prefix + "manifest.json"] = _json(manifest)
    return manifest, files


def _validate_manifest(manifest, run_id):
    prefix = "runs/" + _run_id(run_id) + "/"
    if (
        manifest.get("version") != 1
        or manifest.get("format") != "pretrain-pytorch-v1"
        or manifest.get("run_id") != run_id
    ):
        raise ValueError("use the requested training run's manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 32:
        raise ValueError("manifest must list the checkpoint files")
    for name, entry in files.items():
        path = PurePosixPath(name)
        if (
            not name.startswith(prefix)
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
        ):
            raise ValueError("checkpoint paths must stay within the training run")
        if (
            not isinstance(entry, dict)
            or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("sha256", "")))
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
        ):
            raise ValueError("each checkpoint file needs its size and SHA-256")
    if manifest.get("checkpoint") not in files or (
        manifest.get("resume_checkpoint") is not None
        and manifest["resume_checkpoint"] not in files
    ):
        raise ValueError("manifest must include the model and any resume checkpoint")
    return manifest


def _api(token=None, required=False):
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if required and not token:
        raise ValueError(
            "provide HF_TOKEN through your approved Hugging Face connection"
        )
    return HfApi(token=token or False)


def _download(api, repo_id, name, revision, output):
    from huggingface_hub import hf_hub_download

    output = Path(output).resolve()
    if not (output / name).resolve().is_relative_to(output):
        raise ValueError("download paths must stay within the output directory")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=name,
            revision=revision,
            token=api.token,
            local_dir=output,
        )
    )


def publish_checkpoint(
    checkpoint,
    repo_id,
    run_id,
    resume_checkpoint=None,
    summary=None,
    source_commit=None,
    token=None,
):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.errors import EntryNotFoundError

    manifest, files = prepare_checkpoint(
        checkpoint, run_id, resume_checkpoint, summary, source_commit
    )
    api = _api(token, required=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not info.private:
        raise ValueError("use a private model repository for these checkpoints")
    previous = None
    with tempfile.TemporaryDirectory(prefix="model-hub-manifest-") as temporary:
        try:
            path = _download(
                api, repo_id, "runs/" + run_id + "/manifest.json", info.sha, temporary
            )
            previous = _validate_manifest(json.loads(path.read_text()), run_id)
        except EntryNotFoundError:
            pass
    if previous == manifest:
        revision, reused = info.sha, True
    else:
        operations = [
            CommitOperationAdd(
                path_in_repo=name,
                path_or_fileobj=str(value) if isinstance(value, Path) else value,
            )
            for name, value in files.items()
        ]
        operations += [
            CommitOperationDelete(path_in_repo=name)
            for name in (previous or {}).get("files", {})
            if name not in files
        ]
        committed = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Save training run " + run_id,
            parent_commit=info.sha,
        )
        revision, reused = committed.oid, False
    return {
        "repo_id": repo_id,
        "run_id": run_id,
        "revision": revision,
        "url": "https://huggingface.co/"
        + repo_id
        + "/tree/"
        + revision
        + "/runs/"
        + run_id,
        "checkpoint": manifest["checkpoint"],
        "resume_checkpoint": manifest["resume_checkpoint"],
        "step": manifest["step"],
        "tokens_seen": manifest["tokens_seen"],
        "reused": reused,
    }


def download_checkpoint(repo_id, run_id, output, revision=None, token=None):
    _run_id(run_id)
    api = _api(token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision)
    pinned = info.sha
    output = Path(output).resolve()
    manifest_path = _download(
        api, repo_id, "runs/" + run_id + "/manifest.json", pinned, output
    )
    manifest = _validate_manifest(json.loads(manifest_path.read_text()), run_id)
    for name, entry in manifest["files"].items():
        path = _download(api, repo_id, name, pinned, output)
        if path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
            raise ValueError(
                "downloaded contents do not match the expected checksum: " + name
            )
    return {
        "repo_id": repo_id,
        "run_id": run_id,
        "revision": pinned,
        "checkpoint": str(output / manifest["checkpoint"]),
        "resume_checkpoint": str(output / manifest["resume_checkpoint"])
        if manifest["resume_checkpoint"]
        else None,
        "verified": True,
    }


def publish_training_run(output, repo_id, run_id=None, source_commit=None):
    output = Path(output)
    last = output / "last.pt"
    checkpoint = output / "best.pt" if (output / "best.pt").is_file() else last
    return publish_checkpoint(
        checkpoint,
        repo_id,
        run_id or output.name,
        resume_checkpoint=last,
        source_commit=source_commit,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Save and restore versioned pretraining checkpoints on Hugging Face"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--checkpoint", type=Path, required=True)
    upload.add_argument("--resume-checkpoint", type=Path)
    upload.add_argument("--summary", type=Path)
    upload.add_argument("--source-commit")
    download = commands.add_parser("download")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--revision")
    for command in (upload, download):
        command.add_argument("--repo-id", required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "upload":
        result = publish_checkpoint(
            args.checkpoint,
            args.repo_id,
            args.run_id,
            args.resume_checkpoint,
            args.summary,
            args.source_commit,
        )
    else:
        result = download_checkpoint(
            args.repo_id, args.run_id, args.output, args.revision
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
