import json
import tempfile
import unittest
from pathlib import Path

from pretrain.ruler import (
    LENGTHS,
    QUESTIONS_PER_TASK,
    TASKS,
    eval_command,
    prepare_command,
    read_results,
)


def write_outputs(root, name, score):
    """NeMo-Skills' layout of evaluated replies, each task's replies scored score(length, task, index)."""
    for length in LENGTHS:
        for task in TASKS:
            folder = Path(root) / "eval-results" / f"ruler.{name}-{length}.{task}"
            folder.mkdir(parents=True)
            lines = [json.dumps({"is_correct": score(length, task, index)}) for index in range(QUESTIONS_PER_TASK)]
            (folder / "output.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class RulerTests(unittest.TestCase):
    def test_各長さのデータを作り作者の例と同じ設定で問うコマンドを組む(self):
        for length in LENGTHS:
            prepared = prepare_command("v5", "models/v5", length, "data", "configs", "ruler")
            asked = eval_command("v5", "models/v5", length, "data", "results", "configs")
            self.assertIn(f"--setup=v5-{length}", prepared)
            self.assertIn(f"--max_seq_length={length}", prepared)
            self.assertIn(f"--benchmarks=ruler.v5-{length}", asked)
            self.assertIn("--server_type=vllm", asked)
            for setting in ("++inference.tokens_to_generate=4096", "++inference.top_p=1.0", "++inference.temperature=0.0"):
                self.assertIn(setting, asked)
            # The server holds the longest question with the longest reply, in the precision of training.
            server = next(part for part in asked if part.startswith("--server_args="))
            self.assertIn(f"--max-model-len {max(LENGTHS) + 4096}", server)
            self.assertIn("--dtype bfloat16", server)
            self.assertIn("--trust-remote-code", server)

    def test_長さごとに13課題の点を平均し長さの平均を点にする(self):
        with tempfile.TemporaryDirectory() as root:
            # At each length, half the tasks score every reply 1 and the rest 0, except a part-scored reply.
            write_outputs(root, "v5", lambda length, task, index: float(TASKS.index(task) % 2 == 0) * (0.5 if index == 0 else 1.0))
            record = read_results(root, "v5")
        task = record["lengths"]["4096"]["tasks"]["niah_single_1"]
        self.assertEqual(task["questions"], QUESTIONS_PER_TASK)
        self.assertAlmostEqual(task["accuracy"], 0.995)
        self.assertAlmostEqual(record["lengths"]["8192"]["accuracy"], 7 * 0.995 / 13)
        self.assertAlmostEqual(record["accuracy"], 7 * 0.995 / 13)
        self.assertEqual(record["questions"], len(LENGTHS) * len(TASKS) * QUESTIONS_PER_TASK)

    def test_返答が足りない課題があれば記録を作らない(self):
        with tempfile.TemporaryDirectory() as root:
            write_outputs(root, "v5", lambda length, task, index: 1.0)
            short = Path(root) / "eval-results" / f"ruler.v5-{LENGTHS[-1]}.qa_2" / "output.jsonl"
            short.write_text(json.dumps({"is_correct": 1.0}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replies"):
                read_results(root, "v5")


if __name__ == "__main__":
    unittest.main()
