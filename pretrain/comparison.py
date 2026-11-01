"""Compare two standard evaluations question by question, by language.

Each evaluation records, per question, every score its task averages and
the correct answer's bits per byte. The comparison reads each task's accuracy
as the standard reads it now, so a revised choice needs no new evaluation.
Pairing the same questions removes their difficulty from the difference, so
a smaller difference stands out from the noise.
"""

import math
import statistics
from collections import Counter

from .evaluation import STANDARD_TASKS, result_keys, task_reading

# docs/evaluation.md's pass line, in standard errors of the difference between
# two separately trained models. A language's mean must rise by more than the
# first. With about twenty tasks, the second keeps a task from falling past it
# by chance alone; at two, one of them would about four times in ten.
MEAN_ERRORS = 2
TASK_ERRORS = 3
# Answering in the asked form alone raises this task's score, so a language's
# mean is also shown without it.
FORM_TASKS = {"ja": ("jmmlu",)}


def members(result, name):
    """A task, or a group's tasks in order, through the groups within it."""
    children = result["group_subtasks"].get(name)
    if not children:
        return [name]
    return [task for child in children for task in members(result, child)]


def records(result, name, metric):
    """Each question's key, correctness and answer bits; a group's tasks in order.

    None if the evaluation did not record this accuracy for its questions; the
    bits are None for a task whose answers are not scored by their bits.
    """
    keys, correct, bits = [], [], []
    for member in members(result, name):
        items = result["items"].get(member)
        if items is None or metric not in items["correct"]:
            return None
        # A task's version changes when the way it asks does.
        keys += [(member, result["versions"].get(member), doc) for doc in items["doc_id"]]
        correct += items["correct"][metric]
        if bits is not None and "gold_bits_per_byte" in items:
            bits += items["gold_bits_per_byte"]
        else:
            bits = None
    return keys, correct, bits


def paired(first, second):
    """The mean difference, second minus first, and its standard error."""
    differences = [b - a for a, b in zip(first, second, strict=True)]
    error = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
    return statistics.fmean(differences), error


def compare_task(base, other, name, metric):
    """The difference from base to other on one task, or why they cannot be paired."""
    ours, theirs = records(base, name, metric), records(other, name, metric)
    if ours is None or theirs is None:
        return {"not_compared": f"an evaluation did not record {metric} for each question"}
    (keys, correct, bits), (other_keys, other_correct, other_bits) = ours, theirs
    if keys != other_keys:
        return {"not_compared": "the evaluations asked different questions"}
    accuracy, accuracy_error = paired(correct, other_correct)
    read = {
        "metric": metric,
        "questions": len(keys),
        "base": base["results"][name][result_keys(metric)[0]],
        "other": other["results"][name][result_keys(metric)[0]],
        "difference": accuracy,
        "stderr": accuracy_error,
        "z": accuracy / accuracy_error if accuracy_error else None,
    }
    if bits is not None and other_bits is not None:
        answer, answer_error = paired(bits, other_bits)
        read.update(
            bits_difference=answer,
            bits_stderr=answer_error,
            bits_z=answer / answer_error if answer_error else None,
        )
    return read


def compare_benchmarks(base, other):
    """Differences from base to other on each task either evaluation read, by language.

    A task that cannot be paired is listed with the reason, so that adding or
    changing tasks leaves the others comparable.
    """
    asked = [[name for tasks in result["by_language"].values() for name in tasks] for result in (base, other)]
    comparison = {}
    for name in dict.fromkeys(asked[0] + asked[1]):
        language, metric = task_reading(name)
        if all(name in tasks for tasks in asked):
            read = compare_task(base, other, name, metric)
        else:
            read = {"not_compared": "only one of the evaluations asked it"}
        comparison.setdefault(language, {})[name] = read
    return comparison


def llm_jp_scores(result):
    """Each llm-jp-eval task's score and questions.

    M-IFEval-Ja reads its authors' scoring code when it has been run on the
    saved replies, or else the tool's score of the whole reply, or else its default.
    """
    evaluation = result.get("llm_jp_eval")
    if not evaluation:
        return {}
    scores = evaluation["evaluation"]["scores"]
    questions = Counter(record["target_dataset"] for record in evaluation["records"])
    read = {
        name: (scores[f"{name}_exact_match"], questions[name])
        for name in ("kuci", "jmmlu")
        if f"{name}_exact_match" in scores
    }
    for source in (evaluation.get("official_scores"), evaluation.get("whole_reply_scores"), scores):
        if source and "mif_eval_ja_mifeval_strict" in source:
            read["mif_eval_ja"] = (source["mif_eval_ja_mifeval_strict"], questions["mif_eval_ja"])
            break
    return read


def generation_scores(result):
    """The score and questions of each task the pass line reads, by language."""
    read = {"ja": dict(llm_jp_scores(result)), "en": {}}
    for language, tasks in result["by_language"].items():
        for name, row in tasks.items():
            if name in STANDARD_TASKS:
                questions = sum(result["n-samples"][task]["effective"] for task in members(result, name))
                read[language][name] = (row["accuracy"], questions)
    if "ifeval" in result["results"]:
        read["en"]["ifeval"] = (
            result["results"]["ifeval"]["prompt_level_strict_acc,none"],
            result["n-samples"]["ifeval"]["effective"],
        )
    return read


def score_error(score, questions):
    """A score's standard error, which a score of zero or one still has."""
    return math.sqrt(max(score * (1 - score), 1 / questions) / questions)


def pass_line(base, other):
    """Whether other passes docs/evaluation.md's pass line on scores against base.

    Only tasks both evaluations read count; the rest are listed.
    """
    first, second = generation_scores(base), generation_scores(other)
    languages = {}
    for language in first:
        names = [name for name in first[language] if name in second[language]]
        tasks, variance, dropped = {}, 0.0, []
        for name in names:
            (old, old_questions), (new, new_questions) = first[language][name], second[language][name]
            spread = score_error(old, old_questions) ** 2 + score_error(new, new_questions) ** 2
            variance += spread
            tasks[name] = {"base": old, "other": new, "difference": new - old, "fall_limit": TASK_ERRORS * math.sqrt(spread)}
            if new - old < -tasks[name]["fall_limit"]:
                dropped.append(name)
        mean = lambda kept, side: statistics.fmean(tasks[name][side] for name in kept)
        difference = mean(names, "other") - mean(names, "base")
        kept = [name for name in names if name not in FORM_TASKS.get(language, ())]
        languages[language] = {
            "base": mean(names, "base"),
            "other": mean(names, "other"),
            "difference": difference,
            "difference_without_form_tasks": mean(kept, "other") - mean(kept, "base"),
            "readable": MEAN_ERRORS * math.sqrt(variance) / len(names),
            "rose": difference > MEAN_ERRORS * math.sqrt(variance) / len(names),
            "dropped": dropped,
            "only_one_read": sorted(set(first[language]) ^ set(second[language])),
            "tasks": tasks,
        }
    return {
        "passed": all(read["rose"] and not read["dropped"] for read in languages.values()),
        "languages": languages,
    }
