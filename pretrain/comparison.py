"""Compare two standard evaluations question by question, by language.

Each evaluation records, per question, every accuracy its task reports and
the correct answer's bits per byte. The comparison reads each task's accuracy
as the standard reads it now, so a revised choice needs no new evaluation.
Pairing the same questions removes their difficulty from the difference, so
a smaller difference stands out from the noise.
"""

import math
import statistics

from .evaluation import task_reading


def records(result, name, metric):
    """Each question's key, correctness and answer bits; a group's tasks in order.

    None if the evaluation did not record this accuracy for its questions.
    """
    keys, correct, bits = [], [], []
    for member in result["group_subtasks"].get(name) or [name]:
        items = result["items"].get(member)
        if items is None or metric not in items["correct"]:
            return None
        # A task's version changes when the way it asks does.
        keys += [(member, result["versions"].get(member), doc) for doc in items["doc_id"]]
        correct += items["correct"][metric]
        bits += items["gold_bits_per_byte"]
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
    answer, answer_error = paired(bits, other_bits)
    return {
        "metric": metric,
        "questions": len(keys),
        "base": base["results"][name][f"{metric},none"],
        "other": other["results"][name][f"{metric},none"],
        "difference": accuracy,
        "stderr": accuracy_error,
        "z": accuracy / accuracy_error if accuracy_error else None,
        "bits_difference": answer,
        "bits_stderr": answer_error,
        "bits_z": answer / answer_error if answer_error else None,
    }


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
