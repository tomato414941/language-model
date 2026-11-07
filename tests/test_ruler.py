import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

from pretrain.ruler import LENGTHS, QUESTIONS_PER_TASK, TASKS, evaluate_ruler, ruler_record
from tests.test_benchmarks import byte_tokenizer


def logged_samples(score, questions=QUESTIONS_PER_TASK):
    """The harness's logged samples of every task, each question's score at its length given by score(task, length, index)."""
    return {
        task: [
            {"doc_id": order * questions + index, "doc": {"max_length": length}, str(length): score(task, length, index),
             "4096": -1.0, "resps": [[f"{task} {length} {index}"]]}
            for order, length in enumerate(LENGTHS)
            for index in range(questions)
        ]
        for task in TASKS
    }


class RulerTests(unittest.TestCase):
    def test_長さごとに13課題の点を平均し長さの平均を点にする(self):
        # Every other task scores each reply 1, except one part-scored reply; the second length halves every score.
        def score(task, length, index):
            whole = float(TASKS.index(task) % 2 == 0) * (0.5 if index == 0 else 1.0)
            return whole if length == LENGTHS[0] else whole / 2

        record = ruler_record(logged_samples(score), conversation=True)
        first, second = (record["lengths"][str(length)] for length in LENGTHS)
        self.assertEqual(first["tasks"]["niah_single_1"]["questions"], QUESTIONS_PER_TASK)
        self.assertAlmostEqual(first["tasks"]["niah_single_1"]["accuracy"], (QUESTIONS_PER_TASK - 0.5) / QUESTIONS_PER_TASK)
        per_task = (QUESTIONS_PER_TASK - 0.5) / QUESTIONS_PER_TASK
        self.assertAlmostEqual(first["accuracy"], 7 * per_task / 13)
        self.assertAlmostEqual(second["accuracy"], 7 * per_task / 13 / 2)
        self.assertAlmostEqual(record["accuracy"], (7 * per_task / 13) * 0.75)
        self.assertEqual(record["questions"], len(LENGTHS) * len(TASKS) * QUESTIONS_PER_TASK)
        self.assertEqual(record["procedure"]["form"], "conversation")
        # Each question's reply is kept beside its score.
        self.assertEqual(second["tasks"]["ruler_vt"]["replies"][3], f"ruler_vt {LENGTHS[1]} 3")

    def test_返答が足りない課題があれば記録を作らない(self):
        samples = logged_samples(lambda task, length, index: 1.0)
        samples["ruler_qa_hotpot"] = samples["ruler_qa_hotpot"][:-1]
        with self.assertRaisesRegex(ValueError, "replies"):
            ruler_record(samples, conversation=True)

    def test_長さとモデル自身のトークナイザを渡し会話形式かどうかを選んで問う(self):
        tokenizer = byte_tokenizer(reserved_tokens=8)
        seen = {}

        class Manager:
            def __init__(self, metadata):
                seen["metadata"] = metadata
                # The harness reads the tokenizer while the evaluation runs.
                loaded = AutoTokenizer.from_pretrained(metadata["tokenizer"])
                seen["encoded"] = loaded("abc")["input_ids"]

        def evaluate(model, tasks, task_manager, log_samples, apply_chat_template, limit=None, samples=None):
            # As the harness does: of each task's questions, the first `limit` or those `samples` names.
            seen.update(tasks=tasks, conversation=apply_chat_template)
            every = logged_samples(lambda task, length, index: 1.0)
            return {"samples": {task: [sample for sample in every[task]
                                       if (samples is None and (limit is None or sample["doc_id"] < limit))
                                       or (samples is not None and sample["doc_id"] in samples[task])]
                                for task in tasks}}

        for conversation in (True, False):
            with patch("lm_eval.tasks.TaskManager", Manager), patch("lm_eval.evaluator.simple_evaluate", evaluate):
                record = evaluate_ruler(object(), tokenizer, "cpu", "float32", conversation=conversation, limit=2)
            self.assertEqual(seen["metadata"]["max_seq_lengths"], list(LENGTHS))
            self.assertEqual(seen["encoded"], [tokenizer.token_to_id("<|bos|>"), *tokenizer.encode("abc").ids])
            self.assertEqual(seen["tasks"], list(TASKS))
            self.assertEqual(seen["conversation"], conversation)
            self.assertEqual(record["procedure"]["scope"], "smoke")
            self.assertEqual(record["accuracy"], 1.0)
            # A smoke run asks the same few questions at every length.
            self.assertEqual({length: read["tasks"]["ruler_cwe"]["questions"] for length, read in record["lengths"].items()},
                             {str(length): 2 for length in LENGTHS})
            self.assertFalse(Path(seen["metadata"]["tokenizer"]).exists())


if __name__ == "__main__":
    unittest.main()
