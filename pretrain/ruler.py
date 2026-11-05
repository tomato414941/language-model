"""RULER (Hsieh et al., arXiv 2404.06654) as its authors run it now, and the score the pass line reads.

The authors mark their repository's own pipeline as deprecated and point to
their NeMo-Skills branch for RULER v1. That branch has RULER's code write 100
questions of each of its 13 tasks at a length, with the model's tokenizer,
appends each task's answer prefix to the question, asks it as the user's turn
of a conversation and scores the reply by RULER's string matching. The model is
served by vLLM, and the commands follow the authors' example. A model is asked
at lengths beyond its context as it is, as the authors' instructions raise the
limit rather than cut the question.

The score is the mean of each length's 13-task mean, as the authors' table
averages its lengths.
"""

import json
import statistics
from pathlib import Path

RULER_REPOSITORY = "https://github.com/NVIDIA/RULER"
RULER_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
NEMO_SKILLS_REPOSITORY = "https://github.com/NVIDIA-NeMo/Skills"
NEMO_SKILLS_BRANCH = "chsieh/ruler-remove-prefix"
NEMO_SKILLS_COMMIT = "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3"
# The vLLM of the authors' example cluster configuration, with the Transformers of its time.
VLLM_VERSION = "0.10.1.1"
TRANSFORMERS_VERSION = "4.55.4"
# Everything else in the environment as it stood on the day of the authors' NeMo-Skills commit: later releases of
# NeMo-Run's dependencies dropped modules it imports. The environment has pip, which the preparation calls, and
# the packages RULER's data code imports, which the authors' containers hold.
DEPENDENCIES_AS_OF = "2025-12-13T00:00:00Z"
DATA_PACKAGES = ("wonderwords", "html2text", "tenacity", "beautifulsoup4", "tqdm", "nltk", "scipy", "pyyaml")
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
    "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
# For now, and fixed for v7's pass line. 4,096 is left out: the authors' QA generator (qa.py) loops forever when
# a question's own documents alone exceed the length, which this project's tokenizer reaches with HotpotQA at
# 4,096. Later generations' lines may choose other lengths.
LENGTHS = (8192, 16384)
QUESTIONS_PER_TASK = 100
# The authors' example: greedy replies of up to 4,096 tokens.
GENERATION = ("++inference.tokens_to_generate=4096", "++inference.top_p=1.0", "++inference.temperature=0.0")
REPLY_TOKENS = 4096
# NeMo-Skills runs each step in this machine's environment, without containers; it still looks up the names of
# the containers of the authors' example configuration.
CONTAINERS = ("trtllm", "vllm", "sglang", "megatron", "sandbox", "nemo-skills", "verl", "nemo-rl")
CLUSTER_CONFIG = "executor: none\ncontainers:\n" + "".join(f'  {name}: ""\n' for name in CONTAINERS)
CLUSTER = "local"


def setup_name(name, length):
    return f"{name}-{length}"


def prepare_command(name, model, length, data_dir, config_dir, ruler_dir):
    """The authors' data command for one length; ruler_dir holds RULER at RULER_COMMIT with its downloads."""
    return [
        "ns", "prepare_data", "ruler", f"--cluster={CLUSTER}", f"--config_dir={config_dir}",
        f"--expname=ruler-data-{setup_name(name, length)}", f"--data_dir={data_dir}",
        f"--setup={setup_name(name, length)}", f"--tokenizer_path={model}", f"--max_seq_length={length}",
        f"--tmp_data_dir={ruler_dir}",
    ]


def eval_command(name, model, length, data_dir, output_dir, config_dir):
    """The authors' evaluation command for one length, on one GPU.

    The server holds the longest question with the longest reply, and runs the
    model in bfloat16, as it was trained, where vLLM could take a float32
    checkpoint at float16.
    """
    server = f"--max-model-len {max(LENGTHS) + REPLY_TOKENS} --dtype bfloat16 --trust-remote-code"
    return [
        "ns", "eval", f"--cluster={CLUSTER}", f"--config_dir={config_dir}",
        f"--expname=ruler-{setup_name(name, length)}", f"--data_dir={data_dir}", f"--output_dir={output_dir}",
        f"--benchmarks=ruler.{setup_name(name, length)}", f"--model={model}", "--server_nodes=1",
        "--server_gpus=1", "--server_type=vllm", f"--server_args={server}", *GENERATION, "++skip_filled=True",
    ]


def read_results(output_dir, name):
    """The record of one model's RULER replies: each length's tasks, their questions' scores and the score."""
    lengths = {}
    for length in LENGTHS:
        tasks = {}
        for task in TASKS:
            # NeMo-Skills writes each length's tasks within a folder of that length.
            setup = setup_name(name, length)
            folder = Path(output_dir) / "eval-results" / f"ruler.{setup}" / f"ruler.{setup}.{task}"
            files = sorted(folder.glob("output*.jsonl"))
            if len(files) != 1:
                raise ValueError(f"expected one output file in {folder}, found {len(files)}")
            scores = [float(json.loads(line)["is_correct"]) for line in files[0].read_text(encoding="utf-8").splitlines() if line]
            if len(scores) != QUESTIONS_PER_TASK:
                raise ValueError(f"{folder.name} holds {len(scores)} replies, not {QUESTIONS_PER_TASK}")
            tasks[task] = {"questions": len(scores), "accuracy": statistics.fmean(scores), "scores": scores}
        lengths[str(length)] = {"accuracy": statistics.fmean(task["accuracy"] for task in tasks.values()), "tasks": tasks}
    return {
        "procedure": {
            "ruler": RULER_COMMIT, "nemo_skills": NEMO_SKILLS_COMMIT, "vllm": VLLM_VERSION,
            "lengths": list(LENGTHS), "questions_per_task": QUESTIONS_PER_TASK, "generation": list(GENERATION),
        },
        "lengths": lengths,
        "accuracy": statistics.fmean(length["accuracy"] for length in lengths.values()),
        "questions": sum(task["questions"] for length in lengths.values() for task in length["tasks"].values()),
    }
