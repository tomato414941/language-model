import importlib.util
import tempfile
import unittest
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "ahocorasick", "language_model_evaluation")):
    raise unittest.SkipTest("Install the shared evaluation's dependencies")

from pretrain.overlap import EvaluationOverlap
from tests.helpers import evaluation

QUESTION = "Which of the following best explains why the sky appears blue to an observer on the ground at noon?"
PARAGRAPH = "The quick brown fox jumps over the lazy dog, and then it runs far away into the old green forest."
PROMPT = "Write a short story about a lighthouse keeper, and do not use any commas in your whole reply."
JAPANESE_PROMPT = "灯台守についての短い物語を書いてください。読点を一つも使わないでください。"


class EvaluationOverlapTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = evaluation(self, Path(temporary.name) / "evaluation", {"en": [QUESTION]}, [PARAGRAPH],
                                    {"en": [PROMPT], "ja": [JAPANESE_PROMPT]})
        self.overlap = EvaluationOverlap(self.directory)

    def test_評価の問題と文章と指示のプロンプトを含む文章を見つける(self):
        for held in (
            f"Some notes.\n{QUESTION}\nMore notes.",
            f"A page.\n{PARAGRAPH}\nThe end.",
            f"Asked: {PROMPT}",
            f"依頼：{JAPANESE_PROMPT}",
        ):
            self.assertTrue(self.overlap.holds(held), held)
        self.assertFalse(self.overlap.holds("An explanation of natural science and history. " * 5))

    def test_書き方を変えた評価の文章も見つける(self):
        """改行や大文字、記号の付け方を変えても、評価の文章の一続きを持つ文章を見つける。"""
        rewritten = PARAGRAPH.upper().replace(", ", " -\n").replace(" ", "  ")
        self.assertTrue(self.overlap.holds(f"Quoted:\n{rewritten}\n"))

    def test_何で見つけたかを成果物の名前と作者のコードの版で示す(self):
        from language_model_evaluation import instructions

        names = [path.name for path in self.directory.iterdir() if path.name.startswith("overlap-")]
        self.assertEqual(self.overlap.made_from, {"overlap": names, "instructions": instructions.REVISION})


if __name__ == "__main__":
    unittest.main()
