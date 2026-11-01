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


def llm_jp_scores(evaluation):
    """Each llm-jp-eval task's score, questions and form.

    KUCI and JMMLU read the conversation format when it was asked. M-IFEval-Ja,
    always asked that way, reads its authors' scoring code when it has been run
    on the saved replies, or else the tool's score of the whole reply, or else
    its default.
    """
    read = {}
    for source, form in ((evaluation, "text"), (evaluation.get("as_conversation"), "conversation")):
        if not source:
            continue
        scores = source["evaluation"]["scores"]
        questions = Counter(record["target_dataset"] for record in source["records"])
        for name in ("kuci", "jmmlu"):
            if f"{name}_exact_match" in scores:
                read[name] = (scores[f"{name}_exact_match"], questions[name], form)
    questions = Counter(record["target_dataset"] for record in evaluation["records"])
    for source in (evaluation.get("official_scores"), evaluation.get("whole_reply_scores"), evaluation["evaluation"]["scores"]):
        if source and "mif_eval_ja_mifeval_strict" in source:
            read["mif_eval_ja"] = (source["mif_eval_ja_mifeval_strict"], questions["mif_eval_ja"], "conversation")
            break
    return read


def generation_scores(result):
    """The score, questions and form of each task the pass line reads, by language.

    A task asked in the conversation format as well reads that form; the rest
    read the form they were asked in.
    """
    read = {"ja": dict(llm_jp_scores(result["llm_jp_eval"])) if result.get("llm_jp_eval") else {}, "en": {}}
    for source, form in ((result, "text"), (result.get("as_conversation"), "conversation")):
        if not source:
            continue
        for language, tasks in source["by_language"].items():
            for name, row in tasks.items():
                if name in STANDARD_TASKS:
                    questions = sum(source["n-samples"][task]["effective"] for task in members(source, name))
                    read[language][name] = (row["accuracy"], questions, form)
    if "ifeval" in result["results"]:
        read["en"]["ifeval"] = (
            result["results"]["ifeval"]["prompt_level_strict_acc,none"],
            result["n-samples"]["ifeval"]["effective"],
            "conversation",
        )
    return read


def score_error(score, questions):
    """A score's standard error, which a score of zero or one still has."""
    return math.sqrt(max(score * (1 - score), 1 / questions) / questions)


def pass_line(base, other):
    """Whether other passes docs/evaluation.md's pass line on scores against base.

    Only tasks both evaluations read in the same form count. The rest are listed,
    and leave the line unmet: the comparison is then incomplete.
    """
    first, second = generation_scores(base), generation_scores(other)
    languages = {}
    for language in first:
        names = [
            name for name in first[language]
            if name in second[language] and first[language][name][2] == second[language][name][2]
        ]
        tasks, variance, dropped = {}, 0.0, []
        for name in names:
            (old, old_questions, _), (new, new_questions, _) = first[language][name], second[language][name]
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
            "not_compared": sorted((set(first[language]) | set(second[language])) - set(names)),
            "tasks": tasks,
        }
    return {
        "passed": all(read["rose"] and not read["dropped"] and not read["not_compared"] for read in languages.values()),
        "languages": languages,
    }
