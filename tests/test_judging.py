import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pretrain import judging

ELYZA_CSV = (
    "input,output,eval_aspect\n"
    "仕事の熱意を取り戻すアイデアを挙げてください。,新しい技能を学ぶ。,- 5つでなければ1点減点\n"
    "雨の日の過ごし方を教えてください。,読書をする。,- 屋外の案は1点減点\n"
)
QUESTIONS = [
    {"question_id": 81, "category": "writing", "turns": ["旅行記を書いてください。", "書き直してください。"]},
    {"question_id": 111, "category": "math", "turns": ["三角形の面積は？", "外接円の面積は？"]},
]
REFERENCES = [{"question_id": 111, "choices": [{"index": 0, "turns": ["面積は3です。", "5π"]}]}]
PROMPTS = [
    {
        "name": "single-v1",
        "system_prompt": "あなたは有益な助手です。",
        "prompt_template": "[質問]\n{question}\n[回答]\n{answer}",
        "output_format": "[[評価]]",
    },
    {
        "name": "single-math-v1",
        "system_prompt": "あなたは有益な助手です。",
        "prompt_template": "[質問]\n{question}\n[参考]\n{ref_answer_1}\n[回答]\n{answer}",
        "output_format": "[[評価]]",
    },
]


class ReplyBackend:
    """Answers every question with the same reply, asked in a marked format."""

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def generate_until(self, requests):
        self.prompts.extend(request.args[0] for request in requests)
        return ["返答です。" for _ in requests]


class JudgingTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        files = {
            ("elyza_tasks_100", "test.csv"): ELYZA_CSV.encode(),
            ("ja_mt_bench", "question_full.jsonl"): "\n".join(json.dumps(row, ensure_ascii=False) for row in QUESTIONS).encode(),
            ("ja_mt_bench", "gpt-4.jsonl"): "\n".join(json.dumps(row, ensure_ascii=False) for row in REFERENCES).encode(),
            ("ja_mt_bench", "judge_ja_prompts.jsonl"): "\n".join(json.dumps(row, ensure_ascii=False) for row in PROMPTS).encode(),
        }
        sources = {}
        for (task, name), data in files.items():
            (self.root / task).mkdir(exist_ok=True)
            (self.root / task / name).write_bytes(data)
            source = sources.setdefault(task, {**judging.SOURCES[task], "files": {}})
            source["files"][f"data/{name}"] = hashlib.sha256(data).hexdigest()
        patcher = patch.dict(judging.SOURCES, sources)
        patcher.start()
        self.addCleanup(patcher.stop)

    def judge(self, verdicts):
        """Answer the judge's requests in order with the given verdicts, keeping what it was asked."""
        asked = []

        def respond(request, timeout):
            body = json.loads(request.data)
            asked.append(body)
            verdict = verdicts[len(asked) - 1]
            return io.BytesIO(json.dumps({"model": body["model"], "choices": [{"message": {"content": verdict}}]}).encode())

        return asked, patch("urllib.request.urlopen", side_effect=respond)

    def test_elyza_tasks_are_graded_with_elyza_prompt_and_unreadable_scores_left_out(self):
        """Asks each task, grades it with ELYZA's prompt at temperature 0, and averages the numeric scores."""
        backend = ReplyBackend()
        asked, urlopen = self.judge(["4", "とても良い"])
        with urlopen:
            result = judging.judge_replies(backend, self.root, ["elyza_tasks_100"], key="k", workers=1)
        self.assertEqual(backend.prompts[0], "<user>仕事の熱意を取り戻すアイデアを挙げてください。<assistant>")
        first = asked[0]["messages"][0]["content"]
        self.assertTrue(first.startswith("あなたは採点者です。"))
        self.assertIn("# 正解例\n新しい技能を学ぶ。", first)
        self.assertIn("問題固有の採点基準\n- 5つでなければ1点減点", first)
        self.assertIn("# 回答\n返答です。", first)
        self.assertEqual(asked[0]["temperature"], 0)
        task = result["tasks"]["elyza_tasks_100"]
        self.assertEqual((task["score"], task["scored"], task["unscored"]), (4, 1, 1))
        self.assertEqual([item["score"] for item in task["items"]], [4, None])

    def test_japanese_mt_bench_grades_the_first_turn_with_references_for_mathematics(self):
        """Grades the first turn, gives the math judge GPT-4's reference, and averages scores by category."""
        backend = ReplyBackend()
        asked, urlopen = self.judge(["良い。評価：[[7]]", "誤り。評価：[[2]]"])
        with urlopen:
            result = judging.judge_replies(backend, self.root, ["ja_mt_bench"], key="k", workers=1)
        self.assertEqual(backend.prompts, ["<user>旅行記を書いてください。<assistant>", "<user>三角形の面積は？<assistant>"])
        self.assertEqual(asked[0]["messages"][0], {"role": "system", "content": "あなたは有益な助手です。"})
        self.assertNotIn("[参考]", asked[0]["messages"][1]["content"])
        self.assertIn("[参考]\n面積は3です。", asked[1]["messages"][1]["content"])
        task = result["tasks"]["ja_mt_bench"]
        self.assertEqual(task["score"], 4.5)
        self.assertEqual(task["by_category"], {"math": 2.0, "writing": 7.0})

    def test_judging_needs_an_api_key(self):
        """Refuses to judge without the judge service's key."""
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            judging.judge_replies(ReplyBackend(), self.root, ["elyza_tasks_100"])


if __name__ == "__main__":
    unittest.main()
