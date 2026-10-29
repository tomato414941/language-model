import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from pretrain.conversation import chat_prompt
from pretrain.evaluation import NativeLM
from pretrain.llm_jp_eval import dumped_prompts, generate_replies, write_config
from pretrain.model import LanguageModel, ModelConfig
from tests.test_benchmarks import byte_tokenizer


def prompts(name, texts, output_length):
    return {
        "target_dataset": name,
        "output_length": output_length,
        "config": {"output_length_delta": 2},
        "samples": [{"input": text, "prompt": f"問：{text}", "gold": "A", "metadata": {}} for text in texts],
    }


class LlmJpEvalTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_各データセットの問いに続きを書きllm_jp_evalの形式で保存する(self):
        """Continues each dumped prompt for its output length and saves the replies where the evaluation reads them."""
        tokenizer = byte_tokenizer(reserved_tokens=8)
        model = LanguageModel(tokenizer.get_vocab_size(), ModelConfig(width=8, heads=2, layers=1, context=64))
        dumped = self.root / "prompts"
        dumped.mkdir()
        for name, texts, length in (("kuci", ["一", "二"], 1), ("mif_eval_ja", ["三"], 5)):
            (dumped / f"{name}.eval-prompt.json").write_text(json.dumps(prompts(name, texts, length)), encoding="utf-8")
        asked = []

        def reply(sequences, max_new_tokens, *args, **kwargs):
            for sequence in sequences:
                asked.append((tokenizer.decode(sequence[1:], skip_special_tokens=False), max_new_tokens))
            return [tokenizer.encode("A").ids for _ in sequences]

        with patch.object(model, "continue_greedily", side_effect=reply):
            generate_replies(NativeLM(model, tokenizer, torch.device("cpu")), dumped, self.root / "replies", ["kuci", "mif_eval_ja"])
        self.assertEqual(
            sorted(asked),
            sorted(
                [("問：一", 3), ("問：二", 3), (chat_prompt([{"role": "user", "content": "問：三"}]), 7)]
            ),
        )
        replies = json.loads((self.root / "replies/kuci.eval-generated.json").read_text(encoding="utf-8"))
        self.assertEqual([sample["generated"] for sample in replies["samples"]], ["A", "A"])
        self.assertEqual(replies["target_dataset"], "kuci")
        record = json.loads((self.root / "replies/_config.json").read_text(encoding="utf-8"))
        self.assertEqual(record["inference_result_info"]["dump_prompts_config"], {"output_length_delta": 2})
        self.assertEqual(record["inference_result_info"]["dataset_profiles"]["(total)"]["num_prompt"], 3)

    def test_選んだデータセットの全問を既定の設定で評価する設定を書く(self):
        """Writes the template's settings over every question of the chosen datasets and their categories."""
        home = self.root / "llm-jp-eval"
        (home / "eval_configs").mkdir(parents=True)
        (home / "configs").mkdir()
        (home / "eval_configs/all_datasets.yaml").write_text(
            yaml.safe_dump(
                {
                    "datasets": ["kuci", "jmmlu", "jnli"],
                    "categories": {
                        "CR": {"description": "", "default_metric": "exact_match", "metrics": {}, "datasets": ["kuci"]},
                        "HE-JA": {"description": "", "default_metric": "exact_match", "metrics": {}, "datasets": ["jmmlu"]},
                        "NLI": {"description": "", "default_metric": "exact_match", "metrics": {}, "datasets": ["jnli"]},
                    },
                }
            )
        )
        (home / "configs/config_template.yaml").write_text(yaml.safe_dump({"max_num_samples": 10, "num_few_shots": 4}))
        work = self.root / "work"
        work.mkdir()
        config = yaml.safe_load(write_config(home, work, ["kuci", "jmmlu"], self.root / "replies").read_text())
        self.assertEqual((config["max_num_samples"], config["num_few_shots"]), (-1, 4))
        self.assertEqual(config["inference_result_dir"], str(self.root / "replies"))
        selection = yaml.safe_load(Path(config["eval_dataset_config_path"]).read_text())
        self.assertEqual(selection["datasets"], ["kuci", "jmmlu"])
        self.assertEqual(set(selection["categories"]), {"CR", "HE-JA"})
        self.assertEqual(selection["dataset_info_overrides"], {})

    def test_llm_jp_evalが問いを書き出さなかったデータセットは答えを書く前に断る(self):
        """Refuses the datasets llm-jp-eval left out of the dump, before any reply is written."""
        dumped = self.root / "work/files/datasets/2.1.5/evaluation/test/prompts_abc"
        dumped.mkdir(parents=True)
        (dumped / "kuci.eval-prompt.json").write_text("{}")
        self.assertEqual(dumped_prompts(self.root / "work", ["kuci"]), dumped)
        with self.assertRaisesRegex(ValueError, "jhumaneval"):
            dumped_prompts(self.root / "work", ["kuci", "jhumaneval"])


if __name__ == "__main__":
    unittest.main()
