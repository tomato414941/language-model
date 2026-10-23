"""Compare two standard evaluations question by question, by language.

Each evaluation records the accuracy it read from every task and, per
question, whether the model answered right and the correct answer's bits per
byte. Pairing the same questions removes their difficulty from the
difference, so a smaller difference stands out from the noise.
"""

import math
import statistics


def records(result, name):
    """Each question's key, correctness and answer bits; a group's tasks in order."""
    keys, correct, bits = [], [], []
    for member in result["group_subtasks"].get(name) or [name]:
        items = result["items"][member]
        keys += [(member, doc) for doc in items["doc_id"]]
        correct += items["correct"]
        bits += items["gold_bits_per_byte"]
    return keys, correct, bits


def paired(first, second):
    """The mean difference, second minus first, and its standard error."""
    differences = [b - a for a, b in zip(first, second, strict=True)]
    error = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
    return statistics.fmean(differences), error


def compare_benchmarks(base, other):
    """Differences from base to other on the tasks both evaluations read alike."""
    comparison = {}
    for language, tasks in base["by_language"].items():
        for name, ours in tasks.items():
            theirs = other["by_language"].get(language, {}).get(name)
            if theirs is None:
                continue
            if theirs["metric"] != ours["metric"]:
                raise ValueError(f"{name}: the evaluations read different accuracies")
            keys, correct, bits = records(base, name)
            other_keys, other_correct, other_bits = records(other, name)
            if keys != other_keys:
                raise ValueError(f"{name}: the evaluations asked different questions")
            accuracy, accuracy_error = paired(correct, other_correct)
            answer, answer_error = paired(bits, other_bits)
            comparison.setdefault(language, {})[name] = {
                "metric": ours["metric"],
                "questions": len(keys),
                "base": ours["accuracy"],
                "other": theirs["accuracy"],
                "difference": accuracy,
                "stderr": accuracy_error,
                "z": accuracy / accuracy_error if accuracy_error else None,
                "bits_difference": answer,
                "bits_stderr": answer_error,
                "bits_z": answer / answer_error if answer_error else None,
            }
    return comparison
