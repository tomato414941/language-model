import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pretrain.__main__ import main
from pretrain.comparison import llm_jp_scores
from pretrain.m_ifeval import COMMIT, PUBLISHED, official_scores, require_repeated

# Stands in for the authors' evaluation_main: a reply follows its instruction when
# it holds the word its rules ask for, strictly in the same case, loosely in any.
AUTHORS_CODE = r"""
import json
import sys
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


flags = dict(item.removeprefix("--").split("=", 1) for item in sys.argv[1:])
replied = {row["prompt"]: row["response"] for row in read_lines(flags["input_response_data"])}
for level in ("strict", "loose"):
    rows = []
    for row in read_lines(flags["input_data"]):
        word, reply = row["kwargs"][0]["word"], replied[row["prompt"]]
        followed = [word in reply if level == "strict" else word.lower() in reply.lower()]
        rows.append({"prompt": row["prompt"], "follow_all_instructions": all(followed), "follow_instruction_list": followed})
    written = "".join(json.dumps(row) + "\n" for row in rows)
    Path(flags["output_dir"], f"eval_results_{level}.jsonl").write_text(written, encoding="utf-8")
"""
WORDS = {"問1": "Yes", "問2": "No", "問3": "Maybe"}


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class OfficialScoringTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "code"
        self.home.mkdir()
        (self.home / "evaluation_main.py").write_text(AUTHORS_CODE, encoding="utf-8")
        write_lines(
            self.home / "data" / f"{PUBLISHED}.jsonl",
            [{"prompt": prompt, "response": word} for prompt, word in WORDS.items()],
        )
        self.publish(loose=[[True], [True], [True]])
        prepared = patch("pretrain.m_ifeval.prepare", return_value=(self.home, Path(sys.executable)))
        prepared.start()
        self.addCleanup(prepared.stop)
        # Follows one instruction, follows another only loosely, and follows the third in neither.
        self.partial = self.record("partial", ["Yes", "no", "なし"])
        self.whole = self.record("whole", list(WORDS.values()))

    def publish(self, loose):
        for level, followed in (("strict", [[True], [True], [True]]), ("loose", loose)):
            write_lines(
                self.home / "evaluations" / PUBLISHED / f"eval_results_{level}.jsonl",
                [{"prompt": prompt, "follow_instruction_list": row} for prompt, row in zip(WORDS, followed, strict=True)],
            )

    def record(self, name, answers):
        rows = [{"target_dataset": "kuci", "input": "別の課題", "output": "1", "true": "1"}]
        for (prompt, word), answer in zip(WORDS.items(), answers, strict=True):
            rules = {"instruction_id_list": ["word"], "kwargs": [{"word": word}], "ignored": "field"}
            rows.append({"target_dataset": "mif_eval_ja", "input": prompt, "output": answer, "true": json.dumps(rules)})
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"llm_jp_eval": {"records": rows}}), encoding="utf-8")
        return path

    def scored(self, *arguments):
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown), contextlib.redirect_stderr(io.StringIO()):
            main(["score-m-ifeval", "--output", str(self.root / "official.json"), "--work", str(self.root / "work"), *arguments])
        return json.loads(shown.getvalue())

    def test_各モデルの返答を問うた文面と規則で採点し水準ごとの割合を返す(self):
        """Scores each model's replies of the record's questions, by level."""
        output = official_scores({"partial": self.partial, "whole": self.whole}, self.root / "work")
        self.assertEqual(output["commit"], COMMIT)
        self.assertEqual(output["questions"], 3)
        self.assertEqual(output["missing"], [])
        self.assertAlmostEqual(output["partial"]["mif_eval_ja_mifeval_strict"], 1 / 3)
        self.assertAlmostEqual(output["partial"]["mif_eval_ja_mifeval_loose"], 2 / 3)
        self.assertEqual(output["whole"], {"mif_eval_ja_mifeval_strict": 1, "mif_eval_ja_mifeval_loose": 1})

    def test_作者の公開の結果をいくつ再現したかを水準ごとに返す(self):
        """Counts the authors' published results that the environment repeats, by level."""
        self.publish(loose=[[True], [False], [True]])
        output = official_scores({"partial": self.partial}, self.root / "work")
        self.assertEqual(output["published_results_repeated"], {"strict": 3, "loose": 2})

    def test_記録のないモデルを欠けとして示し残りを採点する(self):
        """Lists a model without a record as missing and scores the others."""
        output = official_scores({"partial": self.partial, "absent": self.root / "absent.json"}, self.root / "work")
        self.assertEqual(output["missing"], ["absent"])
        self.assertIn("partial", output)
        self.assertNotIn("absent", output)

    def test_記録が一つもなければ断る(self):
        """Refuses when none of the records exists."""
        with self.assertRaises(ValueError):
            official_scores({"absent": self.root / "absent.json"}, self.root / "work")

    def test_相対パスの作業場所でも作者のコードの場所から書き込める(self):
        """Scores with a relative work directory, though the authors' code runs from its own."""
        with contextlib.chdir(self.root):
            output = official_scores({"whole": self.whole}, "relative-work")
        self.assertEqual(output["whole"]["mif_eval_ja_mifeval_strict"], 1)

    def test_コマンドは公開の結果を再現すると点を書き出して表示する(self):
        """Writes and shows the scores when the published results repeat."""
        shown = self.scored(f"partial={self.partial}")
        self.assertEqual(shown, json.loads((self.root / "official.json").read_text(encoding="utf-8")))
        self.assertEqual(shown["published_results_repeated"], {"strict": 3, "loose": 3})
        require_repeated(shown)

    def test_コマンドは再現できないと点を書き出してから失敗する(self):
        """Keeps the scores to look at, and fails, when a published result does not repeat."""
        self.publish(loose=[[True], [False], [True]])
        with self.assertRaises(SystemExit) as stopped:
            self.scored(f"partial={self.partial}")
        self.assertNotEqual(stopped.exception.code, 0)
        written = json.loads((self.root / "official.json").read_text(encoding="utf-8"))
        self.assertEqual(written["published_results_repeated"]["loose"], 2)

    def test_コマンドはモデルの記録が欠けると点を書き出してから失敗する(self):
        """Keeps the other models' scores, and fails, when a model's record is missing."""
        with self.assertRaises(SystemExit) as stopped:
            self.scored(f"partial={self.partial}", f"absent={self.root / 'absent.json'}")
        self.assertNotEqual(stopped.exception.code, 0)
        written = json.loads((self.root / "official.json").read_text(encoding="utf-8"))
        self.assertEqual(written["missing"], ["absent"])
        self.assertIn("partial", written)

    def test_合格の線は公式の点を読む(self):
        """Reads the official score in place of the tool's scores of the same replies."""
        official = official_scores({"partial": self.partial}, self.root / "work")["partial"]
        evaluation = {
            "evaluation": {"scores": {"mif_eval_ja_mifeval_strict": 0.9}},
            "whole_reply_scores": {"mif_eval_ja_mifeval_strict": 0.8},
            "official_scores": official,
            "records": [{"target_dataset": "mif_eval_ja"}] * 3,
        }
        score, questions, form = llm_jp_scores(evaluation)["mif_eval_ja"]
        self.assertAlmostEqual(score, 1 / 3)
        self.assertEqual((questions, form), (3, "conversation"))


if __name__ == "__main__":
    unittest.main()
