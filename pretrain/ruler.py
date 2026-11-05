"""RULER (Hsieh et al., arXiv 2404.06654) as lm-evaluation-harness implements it, and the score the pass line reads.

The harness writes RULER's 13 tasks itself, 500 questions of each at each length, sizing them with the model's
tokenizer, and scores the replies by RULER's string matching. The project's own backend asks them, as it asks the
other harness tasks: an instruction-tuned model reads them in the conversation format, where the harness puts each
task's answer prefix at the start of the reply, and a question longer than the model's context keeps its end, as
the harness's own backends do. The model runs as the project's code builds it, so parts that read the tokens before
each step, such as a count table, work as they do in training.

The score is the mean of each length's 13-task mean.
"""

import statistics
import tempfile
from importlib.metadata import version

TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot",
)
# For now: v7 is measured at these lengths for reference, and v8's line compares v8 with v7 at the same ones.
LENGTHS = (8192, 16384)
QUESTIONS_PER_TASK = 500


def ruler_record(samples, conversation, limit=None):
    """The record of one model's RULER replies, from the harness's logged samples: each length's tasks with their
    questions' scores and replies, each length's mean over the tasks, and the mean over the lengths."""
    lengths = {}
    for task in TASKS:
        by_length = {}
        for sample in sorted(samples[task], key=lambda sample: sample["doc_id"]):
            length = sample["doc"]["max_length"]
            by_length.setdefault(length, []).append((float(sample[str(length)]), sample["resps"][0][0]))
        if sorted(by_length) != sorted(LENGTHS):
            raise ValueError(f"{task} holds lengths {sorted(by_length)}, not {sorted(LENGTHS)}")
        for length, answered in by_length.items():
            if limit is None and len(answered) != QUESTIONS_PER_TASK:
                raise ValueError(f"{task} at {length} holds {len(answered)} replies, not {QUESTIONS_PER_TASK}")
            scores, replies = [score for score, _ in answered], [reply for _, reply in answered]
            read = lengths.setdefault(str(length), {"tasks": {}})
            read["tasks"][task] = {
                "questions": len(scores), "accuracy": statistics.fmean(scores), "scores": scores, "replies": replies,
            }
    for read in lengths.values():
        read["accuracy"] = statistics.fmean(task["accuracy"] for task in read["tasks"].values())
    return {
        "procedure": {
            "harness": "lm-evaluation-harness", "version": version("lm_eval"), "lengths": list(LENGTHS),
            "form": "conversation" if conversation else "text", "scope": "full" if limit is None else "smoke",
        },
        "lengths": lengths,
        "accuracy": statistics.fmean(read["accuracy"] for read in lengths.values()),
        "questions": sum(task["questions"] for read in lengths.values() for task in read["tasks"].values()),
    }


def evaluate_ruler(model, tokenizer, device, precision="bfloat16", batch_tokens=65536, conversation=True, limit=None):
    """Ask the model RULER's tasks at LENGTHS through the harness and return its record."""
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    from .evaluation import NativeLM
    from .huggingface import write_tokenizer

    with tempfile.TemporaryDirectory() as directory:
        # The harness sizes each question with this tokenizer, the model's own.
        manager = TaskManager(metadata={"max_seq_lengths": list(LENGTHS), "tokenizer": str(write_tokenizer(tokenizer, directory))})
        result = evaluator.simple_evaluate(
            model=NativeLM(model, tokenizer, device, precision, batch_tokens),
            tasks=list(TASKS),
            task_manager=manager,
            limit=limit,
            log_samples=True,
            apply_chat_template=conversation,
        )
    return ruler_record(result["samples"], conversation, limit)
