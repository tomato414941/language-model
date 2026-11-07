"""A recipe's experiment as a run for the rental tool, which carries it to its end on rented GPUs.

A rental settings file, such as configs/pretrain-v7-rental.toml, names the
recipe and says where its experiment runs: the machine to rent, the rate each
training went at there, and the run's policy. write_run takes the experiment's
commands from the recipe's plan as the run's steps, each with the files it reads
and leaves, and writes spec.json with code.tar.gz beside it: the committed tree,
the small files of the data the preparation built, and the evaluations'
questions, the same bytes for the same commit and data. The data's packed files
stay in the store at the paths the plan reads them from, and are checked against
the recipe before a run is written. Each machine sets up the environment that
the lock file pins, then passes its check, the tests that only a CUDA GPU runs,
before any step: a machine whose GPU computes the model wrongly fails it. The
check runs again whenever a step fails, so that a machine that broke under the
step, as one that lost its GPU, takes the blame, and a step that fails on a
sound machine does not blame it.
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
# Document attention's fast path against the dense mask, and a short training.
GPU_TESTS = (
    "tests.test_pretrain.PretrainingTests.test_GPUでは文書ごとの注意をマスクと同じ結果で速い計算に任せる",
    "tests.test_pretrain.PretrainingTests.test_trains_and_evaluates_on_cuda",
)
SETUP = [["bash", "scripts/setup_environment.sh"]]
CHECK = [[PYTHON, "-m", "unittest", *GPU_TESTS]]
# The benchmarks read the tasks' questions from the code's own cache, and llm-jp-eval keeps its checkout beside it;
# RULER fetches its essays and questions itself.
ENVIRONMENTS = {
    "benchmark": {"HF_HOME": "huggingface", "HF_DATASETS_OFFLINE": "1", "LLM_JP_EVAL_HOME": "llm-jp-eval-home"},
    "ruler": {"HF_HOME": "huggingface"},
}
# The questions every recipe's evaluations read, as Hugging Face keeps them.
QUESTIONS = Path("data/evaluation-questions")


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


class Experiment:
    """The experiment's commands as the run's steps, and what they read of the data built under workdir, here in built:
    the packed files, which the store holds, and the small files, which travel in the code."""

    def __init__(self, plan, rates, built, workdir):
        self.built, self.workdir = built, workdir
        self.steps, self.files, self.small, made = [], {}, {}, set()
        trainings = {command["name"] for command in plan["experiment"] if "training" in command}
        if set(rates) != trainings:
            raise ValueError(f"the settings give rates for {sorted(rates)}, and the trainings are {sorted(trainings)}")
        for command in plan["experiment"]:
            inputs = []
            for path in command["reads"]:
                if any(path == other or path.startswith(f"{other}/") for other in made):
                    inputs.append(path)
                else:
                    inputs += self.read_built(command["name"], path)
            step = {"name": command["name"], "command": [PYTHON, *command["argv"][1:]], "inputs": inputs,
                    "outputs": command["leaves"]}
            if command["kind"] in ENVIRONMENTS:
                step["environment"] = ENVIRONMENTS[command["kind"]]
            if "training" in command:
                step["rate"], step["units"] = rates[command["name"]], self.tokens(command["training"])
            self.steps.append(step)
            made.update(command["leaves"])

    def local(self, path):
        """Where built data the plan reads at path is here."""
        if not Path(path).is_relative_to(self.workdir):
            raise ValueError(f"{path} is neither built data nor what an earlier step leaves")
        return self.built / Path(path).relative_to(self.workdir)

    def read_built(self, name, path):
        """The store's files a step reads at path: packed data's files, or none for a small file, which the code has."""
        local = self.local(path)
        if local.is_file():
            self.small[path] = local
            return []
        files = data_files(local, path)
        self.files.update((item["path"], item) for item in files)
        for small in ("manifest.json", "tokenizer.json"):
            if (local / small).exists():
                self.small[f"{path}/{small}"] = local / small
        return [item["path"] for item in files]

    def tokens(self, training):
        """The tokens a training goes through, in which it reports its progress."""
        settings = Path(training["settings"])
        model, config = read_config(self.local(settings) if settings.is_relative_to(self.workdir) else settings)
        steps = (training["to_step"] or config.steps) - training["from_step"]
        return steps * config.batch_size * config.accumulation_steps * model.context * config.patch_size


def write_code(path, small, questions=QUESTIONS):
    """The committed tree, the built data's small files and the questions, as a tarball of the same bytes each time."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    when = int(subprocess.run(["git", "show", "-s", "--format=%ct", "HEAD"], capture_output=True, text=True,
                              check=True).stdout)
    tree = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    extra = sorted([*small.items(), *((f"huggingface/{item.relative_to(questions)}", item)
                                      for item in questions.rglob("*") if item.is_file())])

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
        for name, local in extra:
            with open(local, "rb") as source:
                bundle.addfile(entry(name, local.stat().st_size), source)
    return commit


def write_run(settings_path, built, directory, bucket, questions=QUESTIONS):
    """Write spec.json and code.tar.gz for the experiment of the recipe the settings name, on the data built in built."""
    settings_path, built, directory = Path(settings_path), Path(built), Path(directory)
    settings = tomllib.loads(settings_path.read_text())
    config = settings_path.parent / settings["recipe"]
    workdir = Path("data") / config.stem
    experiment = Experiment(inspect_recipe(config, workdir), settings["rates"], built, workdir)
    check_built(config, built, sorted({Path(item["path"]).parent.name for item in experiment.files.values()}))
    directory.mkdir(parents=True, exist_ok=True)
    commit = write_code(directory / "code.tar.gz", experiment.small, questions)
    files = list(experiment.files.values())
    spec = {
        "name": f"language-model-{directory.name}",
        "provider": settings["machine"],
        "store": {"kind": "r2", "bucket": bucket, "prefix": f"stages/{config.stem}/{directory.name}"},
        "code": {"path": "code.tar.gz"},
        "setup": SETUP,
        "check": CHECK,
        "files": files,
        "steps": experiment.steps,
        "policy": settings.get("policy", {}),
    }
    (directory / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    return {"spec": str(directory / "spec.json"), "commit": commit, "steps": [step["name"] for step in experiment.steps],
            "files": len(files), "gigabytes": round(sum(item["bytes"] for item in files) / 1e9, 1)}
