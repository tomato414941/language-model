"""Grade replies to Japanese chat evaluations with a judge model, as their publishers do.

ELYZA-tasks-100 gives each of its 100 tasks an example answer and grading
notes; ELYZA's published automatic grading asks a judge for a score from 1
to 5 with the prompt below (zenn.dev/elyza/articles/5e7d9373c32a98) and
keeps only replies it can read as a number. Japanese MT-Bench is graded as
Stability AI's FastChat grades it with its Japanese judge prompts, which
cover the first turn: a score from 1 to 10 in "[[評価]]", with GPT-4's
reference answers for mathematics, reasoning and programming, and judgments
without a score left out. Both publishers judged with GPT-4; the default
here is the dated GPT-4o that the Swallow evaluation judges Japanese
MT-Bench with, and scores compare only under the same judge.

Replies are decoded greedily from the conversation format, so a
checkpoint's scores do not vary between runs; the judge reads at
temperature 0.
"""

import csv
import hashlib
import io
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .benchmarks import download, source_url

SOURCES = {
    "elyza_tasks_100": {
        "repository": "elyza/ELYZA-tasks-100",
        "revision": "240fc800c4695d86ce0f32a713e199cd2ba66ab6",
        "license": "CC-BY-SA-4.0",
        "files": {"test.csv": "51412fbfc955e06e59f6cae2cb564b43758cb5360d6dd65715e6558861dd524c"},
    },
    "ja_mt_bench": {
        "host": "github",
        "repository": "Stability-AI/FastChat",
        "revision": "e37eb28ee490480f6f4989274ed47c55f7dfce82",
        "license": "Apache-2.0",
        "files": {
            "fastchat/llm_judge/data/japanese_mt_bench/question_full.jsonl": "8f56c8bd5ff93d2120409bc8f77f0020e673c74548c9637f2aace12a323834e5",
            "fastchat/llm_judge/data/japanese_mt_bench/reference_answer/gpt-4.jsonl": "0631685c4ebc66390970588b658e4d19047fafb0a00ec1d841616686c07b0d6b",
            "fastchat/llm_judge/data/judge_ja_prompts.jsonl": "a3ef723f4e282ced514c8919773549675f0c0f87a61e5a5f46c320b048803b40",
        },
    },
}
TASKS = tuple(SOURCES)
JUDGE_MODEL = "openai/gpt-4o-2024-08-06"
JUDGE_URL = "https://openrouter.ai/api/v1"
# As FastChat answers MT-Bench's questions.
MAX_REPLY_TOKENS = 1024
# FastChat's categories whose judge compares the reply with a reference answer.
REFERENCE_CATEGORIES = ("math", "reasoning", "coding")
SCORE = re.compile(r"\[\[(\d+\.?\d*)\]\]")
SCORE_BACKUP = re.compile(r"\[(\d+\.?\d*)\]")
ELYZA_PROMPT = """あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{input_text}

# 正解例
{output_text}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 回答
{pred}
"""


def fetch_sources(output):
    """Download each evaluation's files, checked against their digests."""
    output = Path(output)
    for name, source in SOURCES.items():
        for path, expected in source["files"].items():
            target = output / name / Path(path).name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                continue
            data = download(source_url(source, path))
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"{name} source checksum does not match: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_questions(directory, name):
    """Each question's identifier, its first user turn and what its judge needs."""
    directory = Path(directory) / name
    for path, expected in SOURCES[name]["files"].items():
        if hashlib.sha256((directory / Path(path).name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} source checksum does not match: {path}")
    if name == "elyza_tasks_100":
        rows = csv.DictReader(io.StringIO((directory / "test.csv").read_text(encoding="utf-8")))
        return [
            {"id": index, "question": row["input"], "example": row["output"], "aspect": row["eval_aspect"]}
            for index, row in enumerate(rows)
        ]
    references = {row["question_id"]: row for row in read_jsonl(directory / "gpt-4.jsonl")}
    return [
        {
            "id": row["question_id"],
            "category": row["category"],
            "question": row["turns"][0],
            "reference": references[row["question_id"]]["choices"][0]["turns"][0]
            if row["category"] in REFERENCE_CATEGORIES
            else None,
        }
        for row in read_jsonl(directory / "question_full.jsonl")
    ]


def judge_messages(name, question, reply, prompts=None):
    """The messages that ask the judge to grade a reply, in its publisher's words."""
    if name == "elyza_tasks_100":
        text = ELYZA_PROMPT.format(
            input_text=question["question"],
            output_text=question["example"],
            eval_aspect=question["aspect"],
            pred=reply,
        )
        return [{"role": "user", "content": text}]
    prompt = prompts["single-math-v1" if question["reference"] is not None else "single-v1"]
    fields = {"question": question["question"], "answer": reply}
    if question["reference"] is not None:
        fields["ref_answer_1"] = question["reference"]
    return [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["prompt_template"].format(**fields)},
    ]


def read_score(name, judgment):
    """The score a judgment gives, or None where its publisher's reading finds none."""
    if name == "elyza_tasks_100":
        try:
            return int(judgment)
        except ValueError:
            return None
    match = SCORE.search(judgment) or SCORE_BACKUP.search(judgment)
    return float(match.group(1)) if match else None


def ask_judge(messages, model, url, key, attempts=6):
    """The judge's reply at temperature 0, retried with growing waits when the service fails."""
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": 2048}
    ).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(
            url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.load(response)
            return reply["choices"][0]["message"]["content"], reply.get("model")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 2**attempt))


def generate_replies(backend, questions):
    """Each question's reply, decoded greedily from the conversation format."""
    requests = [
        SimpleNamespace(
            args=(
                backend.apply_chat_template([{"role": "user", "content": question["question"]}]),
                {"until": [], "do_sample": False, "temperature": 0.0, "max_gen_toks": MAX_REPLY_TOKENS},
            )
        )
        for question in questions
    ]
    return backend.generate_until(requests)


def judge_replies(
    backend, directory, tasks=None, limit=None, model=JUDGE_MODEL, url=JUDGE_URL, key=None, workers=8
):
    """Ask each task's questions, have the judge grade the replies, and report each task's mean score."""
    tasks = TASKS if tasks is None else tasks
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("the judge needs an API key in OPENROUTER_API_KEY")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    result = {
        "judge": {"model": model, "url": url, "temperature": 0},
        "generation": {"decoding": "greedy", "max_reply_tokens": MAX_REPLY_TOKENS},
        "evaluation_scope": "full" if limit is None else "smoke",
        "tasks": {},
    }
    for name in tasks:
        if name not in SOURCES:
            raise ValueError(f"unknown judged evaluation: {name}")
        questions = load_questions(directory, name)[:limit]
        prompts = None
        if name == "ja_mt_bench":
            prompts = {row["name"]: row for row in read_jsonl(Path(directory) / name / "judge_ja_prompts.jsonl")}
        replies = generate_replies(backend, questions)
        with ThreadPoolExecutor(workers) as pool:
            judgments = list(
                pool.map(
                    lambda pair: ask_judge(judge_messages(name, *pair, prompts), model, url, key),
                    zip(questions, replies),
                )
            )
        items = []
        for question, reply, (judgment, served) in zip(questions, replies, judgments):
            items.append(
                {
                    "id": question["id"],
                    **({"category": question["category"]} if "category" in question else {}),
                    "reply": reply,
                    "judgment": judgment,
                    "judge_model": served,
                    "score": read_score(name, judgment),
                }
            )
        scored = [item for item in items if item["score"] is not None]
        summary = {
            "score": statistics.fmean(item["score"] for item in scored) if scored else None,
            "scored": len(scored),
            "unscored": len(items) - len(scored),
        }
        if name == "ja_mt_bench":
            summary["by_category"] = {
                category: statistics.fmean(item["score"] for item in scored if item["category"] == category)
                for category in sorted({item["category"] for item in scored})
            }
        result["tasks"][name] = {**summary, "items": items}
    return result
