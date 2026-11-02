"""Score saved M-IFEval-Ja replies with the authors' own code.

The authors' code runs unchanged, from its commit, in an environment of its own.
It first scores the authors' published replies of one model and compares every
result with their published results, which shows this environment scores as
theirs did. The prompts come from the evaluation's record, as asked: seven of the
authors' input file's prompts say 文 where their published replies, llm-jp-eval and
the record say 行, with the same rules.
"""

import json
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/lightblue-tech/M-IFEval.git"
COMMIT = "10b874dbf8c4ca25ed5d876e1d3226e7f3f9d0dc"
# The authors' pins of what evaluation_main imports; the rest of their
# requirements serve to generate replies.
PACKAGES = (
    "absl-py==1.4.0",
    "immutabledict==4.2.1",
    "janome==0.5.0",
    "ja_sentence_segmenter==0.0.2",
    "langdetect==1.0.9",
    "nltk==3.9.1",
    "spacy==3.7.5",
)
# The models their README downloads, which their instruction modules load on import.
MODELS = tuple(
    f"https://github.com/explosion/spacy-models/releases/download/{name}-3.7.0/{name}-3.7.0-py3-none-any.whl"
    for name in ("es_core_news_sm", "xx_sent_ud_sm")
)
PUBLISHED = "ja_input_response_data_Qwen__Qwen2.5-0.5B-Instruct-GPTQ-Int4"
LEVELS = ("strict", "loose")


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def prepare(work):
    """The authors' commit and an environment with their pins; returns its Python."""
    home, environment = work / "code", work / "environment"
    if not (home / "evaluation_main.py").exists():
        home.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--quiet", cwd=home)
        run("git", "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT, cwd=home)
        run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=home)
    python = environment / "bin/python"
    if not python.exists():
        run("uv", "venv", "--quiet", "--python", "3.12", str(environment))
        run("uv", "pip", "install", "--quiet", "--python", str(python), *PACKAGES, *MODELS)
    return home, python


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def replies(path):
    """The record's M-IFEval-Ja questions: each prompt as asked, its rules and whole reply."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in record["llm_jp_eval"]["records"] if row["target_dataset"] == "mif_eval_ja"]


def questions(rows):
    return [
        {"key": index, "prompt": row["input"], **{k: json.loads(row["true"])[k] for k in ("instruction_id_list", "kwargs")}}
        for index, row in enumerate(rows)
    ]


def score(home, python, asked, answered, directory):
    """Each level's results from the authors' evaluation_main."""
    write_lines(directory / "input.jsonl", asked)
    write_lines(directory / "responses.jsonl", answered)
    run(
        str(python), "-m", "evaluation_main",
        f"--input_data={directory / 'input.jsonl'}",
        f"--input_response_data={directory / 'responses.jsonl'}",
        f"--output_dir={directory}",
        cwd=home,
    )
    return {level: read_lines(directory / f"eval_results_{level}.jsonl") for level in LEVELS}


def agreement(home, python, asked, directory):
    """How many of the authors' published results this environment repeats, by level."""
    answered = read_lines(home / "data" / f"{PUBLISHED}.jsonl")
    ours = score(home, python, asked, answered, directory)
    counts = {}
    for level in LEVELS:
        theirs = read_lines(home / "evaluations" / PUBLISHED / f"eval_results_{level}.jsonl")
        by_prompt = {row["prompt"]: row["follow_instruction_list"] for row in theirs}
        counts[level] = sum(by_prompt.get(row["prompt"]) == row["follow_instruction_list"] for row in ours[level])
    return counts


def official_scores(records, work):
    """Each model's official scores, and how many published results this environment repeats.

    records maps a model's name to its evaluation record. A record that does not
    exist is listed as missing, and the others are still scored.
    """
    work = Path(work).resolve()
    loaded = {name: replies(Path(path)) for name, path in records.items() if Path(path).exists()}
    if not loaded:
        raise ValueError("none of the evaluation records exists")
    home, python = prepare(work)
    asked = questions(next(iter(loaded.values())))
    output = {"commit": COMMIT, "questions": len(asked), "missing": sorted(set(records) - set(loaded))}
    output["published_results_repeated"] = agreement(home, python, asked, work / "published")
    for name, rows in loaded.items():
        answered = [{"prompt": row["input"], "response": row["output"]} for row in rows]
        scored = score(home, python, questions(rows), answered, work / name)
        output[name] = {
            f"mif_eval_ja_mifeval_{level}": sum(row["follow_all_instructions"] for row in scored[level]) / len(scored[level])
            for level in LEVELS
        }
    return output


def require_repeated(output):
    """Refuse scores from an environment that did not repeat the authors' results, or with a model missing."""
    repeated = output["published_results_repeated"]
    if output["missing"] or any(count != output["questions"] for count in repeated.values()):
        raise ValueError("the authors' published results were not repeated, or a model's replies are missing")
