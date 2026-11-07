"""Ask Japanese tasks through llm-jp-eval, as its published workflow for other inference engines does.

llm-jp-eval builds each task's prompts and scores the replies; a model it
cannot load writes the replies itself, as llm-jp-eval-inference's modules do:
the tool preprocesses the datasets and dumps the prompts, this module
continues each prompt greedily for the task's output length and saves the
replies in that format, and the tool's evaluation reads them. The release is
pinned, its default settings stand, and every question is asked, but in a smoke
run, which asks the first few of each dataset and says so in its record.

Tasks asked in the conversation format put the prompt in a user turn, as
llm-jp-eval-inference's chat template option does; the others continue the
prompt as plain text. An instruction-tuned model may be asked all of them in the
conversation format as well, the form it learned. The template leaves out the
datasets llm-jp-eval does not process for commercial use, JHumanEval among them,
so asking for one is refused before any reply is written.

By default the tool scores only what precedes a reply's first blank line. Tasks
that judge the whole reply, as their authors' code does, are scored that way as
well, by the tool's own setting for a dataset's extraction. The default scores
stay, to compare with others who keep the defaults.
"""

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from .conversation import chat_prompt, require_markers
from .evaluation import NativeLM

REPOSITORY = "https://github.com/llm-jp/llm-jp-eval.git"
RELEASE = "v2.1.5"
# These tasks ask in the conversation format that the API serves.
CONVERSATION_DATASETS = ("mif_eval_ja",)
# These tasks judge the whole reply, so they are scored on it as well.
WHOLE_REPLY_DATASETS = ("mif_eval_ja",)
WHOLE_REPLY = "(?s)^(.*)$"


class Request:
    """The shape of a harness request that NativeLM.generate_until reads."""

    def __init__(self, prompt, limit):
        self.args = (prompt, {"until": [], "max_gen_toks": limit})


def checkout(home):
    """The pinned release with its locked environment, fetched once."""
    home = Path(home)
    if not (home / "pyproject.toml").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", RELEASE, REPOSITORY, str(home)],
            check=True,
        )
    subprocess.run(["uv", "sync", "--quiet", "--frozen"], cwd=home, check=True)
    return home


def tool(home, *arguments):
    subprocess.run(["uv", "run", "--frozen", "--no-sync", "python", *arguments], cwd=home, check=True)


def write_config(home, work, datasets, results=None, limit=None):
    """llm-jp-eval's template settings over every question of the chosen datasets, or the first `limit` of each."""
    catalog = yaml.safe_load((home / "eval_configs/all_datasets.yaml").read_text())
    categories = {}
    for name, category in catalog["categories"].items():
        chosen = [dataset for dataset in category["datasets"] if dataset in datasets]
        if chosen:
            categories[name] = {**category, "datasets": chosen}
    selection = work / "datasets.yaml"
    selection.write_text(
        yaml.safe_dump(
            {"datasets": list(datasets), "categories": categories, "dataset_info_overrides": {}},
            allow_unicode=True,
        )
    )
    config = yaml.safe_load((home / "configs/config_template.yaml").read_text())
    config.update(
        run_name="language-model",
        output_dir=str(work / "files"),
        eval_dataset_config_path=str(selection),
        max_num_samples=-1 if limit is None else limit,
        exporters={"local": {"export_output_table": True, "output_top_n": None}},
    )
    if results is not None:
        config["inference_result_dir"] = str(results)
    path = work / ("evaluation.yaml" if results is not None else "prompts.yaml")
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    return path


def dumped_prompts(work, datasets):
    """The directory of the dumped prompts, refusing the datasets llm-jp-eval left out."""
    [prompts] = (work / "files/datasets").glob("*/evaluation/test/prompts_*")
    missing = [name for name in datasets if not (prompts / f"{name}.eval-prompt.json").exists()]
    if missing:
        raise ValueError(
            f"llm-jp-eval wrote no prompts for {', '.join(missing)}; its settings leave out "
            "the datasets it does not process for commercial use"
        )
    return prompts


def generate_replies(backend, prompts, results, datasets, conversation=CONVERSATION_DATASETS):
    """Write each dataset's replies and the run's record where llm-jp-eval's evaluation reads them."""
    results.mkdir(parents=True, exist_ok=True)
    profiles, times, dumped = {}, {}, {}
    for name in datasets:
        data = json.loads((prompts / f"{name}.eval-prompt.json").read_text(encoding="utf-8"))
        dumped = dumped or data["config"]
        limit = data["output_length"] + data["config"].get("output_length_delta", 0)
        texts = []
        for sample in data["samples"]:
            text = sample["prompt"]
            if name in conversation:
                messages = [{"role": "user", "content": text}]
                if sample.get("system_prompt"):
                    messages.insert(0, {"role": "system", "content": sample["system_prompt"]})
                text = chat_prompt(messages)
            texts.append(text)
        started = time.monotonic()
        replies = backend.generate_until([Request(text, limit) for text in texts])
        times[name] = time.monotonic() - started
        written = copy.deepcopy(data)
        for sample, reply in zip(written["samples"], replies, strict=True):
            sample["generated"] = reply
        (results / f"{name}.eval-generated.json").write_text(
            json.dumps(written, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        lengths = [len(backend.tok_encode(text)) for text in texts]
        profiles[name] = {
            "num_prompt": len(texts),
            "max_input_len": max(lengths),
            "sum_input_len": sum(lengths),
            "max_output_len": limit,
            "max_seq_len": max(lengths) + limit,
            "sum_seq_len": sum(lengths) + limit * len(texts),
        }
    total = {key: sum(profile[key] for profile in profiles.values()) for key in ("num_prompt", "sum_input_len", "sum_seq_len")}
    for key in ("max_input_len", "max_output_len", "max_seq_len"):
        total[key] = max(profile[key] for profile in profiles.values())
    profiles["(total)"] = total
    infer = sum(times.values())
    (results / "_config.json").write_text(
        json.dumps(
            {
                "run_name": "language-model",
                "generation_config": {"do_sample": False},
                "inference_result_info": {
                    "dump_prompts_config": dumped,
                    "dataset_profiles": profiles,
                    "benchmark": {"lib": "pretrain", "infer": infer},
                    "time_profile": {"(infer)": infer, **times},
                },
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def score_whole_replies(home, work, replies, datasets, limit=None):
    """The tool's scores of these datasets' replies, each taken whole."""
    whole = work / "whole-reply"
    results = whole / "replies"
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy(replies / "_config.json", results)
    for name in datasets:
        # The tool needs the dataset in its output directory even to score saved replies.
        tool(home, "scripts/preprocess_dataset.py", "--dataset-name", name, "--output-dir", str(whole / "files"))
        data = json.loads((replies / f"{name}.eval-generated.json").read_text(encoding="utf-8"))
        data["answer_extract_pattern"] = WHOLE_REPLY
        (results / f"{name}.eval-generated.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    tool(home, "scripts/evaluate_llm.py", "eval", "--config",
         str(write_config(home, whole, datasets, results, limit)))
    [scored] = (whole / "files/results").glob("result_*.json")
    return json.loads(scored.read_text(encoding="utf-8"))["evaluation"]["scores"]


def ask_llm_jp(home, backend, datasets, conversation, work, limit=None):
    """One evaluation by the tool: its prompts, the model's replies and the tool's scores."""
    work.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        tool(home, "scripts/preprocess_dataset.py", "--dataset-name", name, "--output-dir", str(work / "files"))
    tool(home, "scripts/evaluate_llm.py", "dump", "--config", str(write_config(home, work, datasets, limit=limit)))
    prompts = dumped_prompts(work, datasets)
    results = work / "replies"
    generate_replies(backend, prompts, results, datasets, conversation)
    tool(home, "scripts/evaluate_llm.py", "eval", "--config",
         str(write_config(home, work, datasets, results, limit)))
    [scored] = (work / "files/results").glob("result_*.json")
    output = json.loads(scored.read_text(encoding="utf-8"))
    output["framework"] = {"name": "llm-jp-eval", "version": RELEASE}
    output["evaluation_scope"] = "full" if limit is None else "smoke"
    output["conversation_format_datasets"] = [name for name in datasets if name in conversation]
    judged_whole = [name for name in datasets if name in WHOLE_REPLY_DATASETS]
    if judged_whole:
        output["whole_reply_scores"] = score_whole_replies(home, work, results, judged_whole, limit)
    return output


def evaluate_llm_jp(model, tokenizer, device, precision, batch_tokens, datasets, work, also_as_conversation=False,
                    limit=None):
    """Scores, settings and every reply of llm-jp-eval's evaluation of the chosen datasets, of every question or of
    the first `limit` of each.

    With also_as_conversation, the datasets asked as plain text are asked in
    the conversation format too, under as_conversation.
    """
    if not datasets:
        raise ValueError("select at least one llm-jp-eval dataset")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    plain = [name for name in datasets if name not in CONVERSATION_DATASETS]
    if also_as_conversation or len(plain) != len(datasets):
        require_markers(tokenizer)
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    home = checkout(os.environ.get("LLM_JP_EVAL_HOME") or work / "llm-jp-eval")
    backend = NativeLM(model, tokenizer, device, precision, batch_tokens)
    output = ask_llm_jp(home, backend, datasets, CONVERSATION_DATASETS, work, limit)
    if also_as_conversation and plain:
        output["as_conversation"] = ask_llm_jp(home, backend, plain, plain, work / "as-conversation", limit)
    return output
