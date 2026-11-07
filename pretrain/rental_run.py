"""A recipe as a run for the rental tool, which carries it to its end on rented GPUs.

A rental settings file, such as configs/pretrain-v7-rental.toml, names the
recipe and says where it runs: the machine to rent, the rates its trainings
were measured at there, and the run's policy. write_run turns the recipe's plan
into the run's steps, from GPU tests through the trainings, the average and the
benchmarks to RULER, each with the files it reads and leaves, and writes
spec.json with code.tar.gz beside it: the committed tree, the built data's small
files and the evaluations' questions, the same bytes for the same commit and data.
The packed data stays in the store at the paths the plan reads it from, and is
checked against the recipe before a run is written.
"""

import gzip
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

from .recipe import data_fingerprint, inspect_recipe, read_config

PYTHON = ".venv/bin/python"
GPU_TESTS = [PYTHON, "-m", "unittest", "-v", "-k", "文書ごとの注意", "-k", "on_cuda", "tests.test_pretrain"]
# The benchmarks read the tasks' questions from the code's cache, and llm-jp-eval keeps its checkout beside it.
EVALUATION = {"HF_HOME": "huggingface", "HF_DATASETS_OFFLINE": "1", "LLM_JP_EVAL_HOME": "llm-jp-eval-home"}
QUESTIONS = Path("data/evaluation-v5/huggingface")


def option(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def data_files(directory, path):
    """The packed files of built data, with the digest and size its manifest records for each."""
    manifest = json.loads((directory / "manifest.json").read_text())
    width = int(manifest["dtype"][2:])
    files = []
    for name, entry in sorted(manifest["files"].items()):
        if manifest.get("kind") == "instructions":
            size = entry["rows"] * (manifest["context"] + 1) * (1 if name.endswith("_mask.bin") else width)
        else:
            size = entry["tokens"] * width
        files.append({"path": f"{path}/{name}", "key": f"{path}/{name}", "sha256": entry["sha256"], "bytes": size})
    return files


def check_built(config, built, names):
    """Refuse built data that belongs to another recipe, tokenizer or set of benchmark exclusions."""
    fingerprint = data_fingerprint(config)
    tokenizer = json.loads((built / "tokenizer/manifest.json").read_text())
    if tokenizer["data_fingerprint"] != fingerprint:
        raise ValueError("the tokenizer belongs to another recipe")
    for name in names:
        manifest = json.loads((built / name / "manifest.json").read_text())
        if manifest.get("kind") == "instructions":
            if (manifest["tokenizer_sha256"], manifest["benchmarks_sha256"]) != (
                    tokenizer["tokenizer_sha256"], tokenizer["benchmarks_sha256"]):
                raise ValueError(f"{name} uses another tokenizer or other benchmark exclusions")
        elif manifest["data_fingerprint"] != fingerprint or manifest["tokenizer_sha256"] != tokenizer["tokenizer_sha256"]:
            raise ValueError(f"{name} belongs to another recipe or tokenizer")


def run_steps(plan, rates, built, workdir):
    """The plan's trainings, average and benchmarks as steps, with the GPU tests first and RULER last."""
    def local(path):
        # The plan reads its data and the tuning's settings under the workdir; they were built elsewhere.
        return built / Path(path).relative_to(workdir) if Path(path).is_relative_to(workdir) else Path(path)

    steps, files, ends, tuned = [{"name": "gpu-tests", "command": GPU_TESTS}], [], {}, None
    for command in plan["commands"]:
        kind, rest = command[3], command[4:]
        if kind == "train":
            settings, data, output = option(command, "--config"), option(command, "--data"), option(command, "--output")
            model, training = read_config(local(settings))
            begun = option(command, "--continue-from")
            start = ends[str(Path(begun).parent)] if begun else 0
            ends[output] = target = int(option(command, "--max-steps") or training.steps)
            outputs = [f"{output}/last.pt", f"{output}/metrics.jsonl"]
            if training.snapshot_every and target == training.steps:
                # The decay's snapshots, which the average reads.
                outputs.append(f"{output}/snapshots")
            read = data_files(local(data), data)
            files += read
            steps.append({
                "name": Path(output).name, "command": [PYTHON, "-m", "pretrain", "train", *rest, "--threads", "4"],
                "inputs": [item["path"] for item in read] + [path for path in (
                    option(command, "--initialize-from"), begun) if path],
                "outputs": outputs, "rate": rates[settings],
                "units": (target - start) * training.batch_size * training.accumulation_steps * model.context
                * training.patch_size,
            })
            tuned = output
        elif kind == "average":
            run, output = option(command, "--run"), option(command, "--output")
            steps.append({"name": "average", "command": [PYTHON, "-m", "pretrain", "average", *rest],
                          "inputs": [f"{run}/last.pt", f"{run}/snapshots"], "outputs": [output]})
        elif kind == "benchmark":
            checkpoint = Path(option(command, "--checkpoint"))
            label = checkpoint.parent.name + ("" if checkpoint.stem == "last" else f"-{checkpoint.stem}")
            steps.append({
                "name": f"benchmark-{label}",
                "command": [PYTHON, "-m", "pretrain", "benchmark", *rest, "--batch-tokens", "16384", "--threads", "4"],
                "environment": EVALUATION, "inputs": [str(checkpoint)], "outputs": [option(command, "--output")],
            })
    # RULER at 8k and 16k for the tuned model, for reference; it fetches its own essays and questions.
    steps.append({"name": "ruler", "command": [PYTHON, "-m", "pretrain", "ruler", "--checkpoint", f"{tuned}/last.pt",
                                               "--device", "cuda", "--output", f"{tuned}/ruler.json"],
                  "environment": {"HF_HOME": "huggingface"}, "inputs": [f"{tuned}/last.pt"],
                  "outputs": [f"{tuned}/ruler.json"]})
    return steps, files


def write_code(path, built, workdir, names, questions=QUESTIONS):
    """The committed tree, the built data's small files and the questions, as a tarball of the same bytes each time."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    when = int(subprocess.run(["git", "show", "-s", "--format=%ct", "HEAD"], capture_output=True, text=True,
                              check=True).stdout)
    tree = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    extra = [(f"{workdir}/{name}/{small}", built / name / small) for name in names
             for small in ("manifest.json", "tokenizer.json") if (built / name / small).exists()]
    extra.append((f"{workdir}/sft.toml", built / "sft.toml"))
    extra += [(f"huggingface/{item.relative_to(questions)}", item)
              for item in sorted(questions.rglob("*")) if item.is_file()]

    def entry(name, size):
        info = tarfile.TarInfo(name)
        info.size, info.mtime, info.mode = size, when, 0o644
        return info

    with open(path, "wb") as target, gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as packed, \
            tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
        with tarfile.open(fileobj=io.BytesIO(tree)) as source:
            for member in source.getmembers():
                bundle.addfile(member, source.extractfile(member) if member.isfile() else None)
        marker = f"{commit}\n".encode()
        bundle.addfile(entry("SOURCE_COMMIT", len(marker)), io.BytesIO(marker))
        for name, local in sorted(extra):
            with open(local, "rb") as source:
                bundle.addfile(entry(name, local.stat().st_size), source)
    return commit


def write_run(settings_path, built, directory, bucket, questions=QUESTIONS):
    """Write spec.json and code.tar.gz for the recipe the settings name, from its data built in built."""
    settings_path, built, directory = Path(settings_path), Path(built), Path(directory)
    settings = tomllib.loads(settings_path.read_text())
    config = settings_path.parent / settings["recipe"]
    workdir = Path("data") / config.stem
    plan = inspect_recipe(config, workdir)
    steps, files = run_steps(plan, settings["rates"], built, workdir)
    names = sorted({Path(item["path"]).parent.name for item in files})
    check_built(config, built, names)
    directory.mkdir(parents=True, exist_ok=True)
    commit = write_code(directory / "code.tar.gz", built, workdir, names, questions)
    spec = {
        "name": f"language-model-{directory.name}",
        "provider": settings["machine"],
        "store": {"kind": "r2", "bucket": bucket, "prefix": f"stages/{config.stem}/{directory.name}"},
        "code": {"path": "code.tar.gz"},
        "setup": [["bash", "scripts/setup_environment.sh"]],
        "files": files,
        "steps": steps,
        "policy": settings.get("policy", {}),
    }
    (directory / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    return {"spec": str(directory / "spec.json"), "commit": commit, "steps": [step["name"] for step in steps],
            "files": len(files), "gigabytes": round(sum(item["bytes"] for item in files) / 1e9, 1)}
