import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers", "language_model_evaluation")):
    raise unittest.SkipTest("Install training dependencies")

import torch
from language_model_evaluation import definition, flores

from pretrain.balance import measure
from pretrain.comparison import compare_results
from pretrain.model import LanguageModel, ModelConfig
from tests.test_knowledge import byte_tokenizer

# The same sentences in English and Japanese, by their ids.
SENTENCES = {"1": ("Rain falls on the hills.", "雨が丘に降る。"), "2": ("The river runs to the sea.", "川は海へ流れる。")}


class BalanceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        torch.manual_seed(0)
        self.tokenizer = byte_tokenizer()
        self.model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=64))
        sentences = flores.fetch(self.directory, "devtest", self.download)
        self.definition = {"name": "toy", "balance": ["jpn_Jpan", "eng_Latn"],
                           "flores": {"split": "devtest", "text": sentences.name}}

    @staticmethod
    def download(path, directory):
        """A file of FLORES+ of SENTENCES, a sentence a line as it gives them."""
        column = list(flores.LANGUAGES).index(Path(path).stem)
        local = Path(directory) / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(json.dumps({"id": key, "text": texts[column]}, ensure_ascii=False) + "\n"
                                 for key, texts in SENTENCES.items()), encoding="utf-8")
        return local

    def bits(self, model, text):
        """The bits that the model takes to read a text by itself after the BOS token, read directly."""
        tokens = [self.tokenizer.token_to_id("<|bos|>"), *self.tokenizer.encode(text, add_special_tokens=False).ids]
        with torch.no_grad():
            logits = model(torch.tensor([tokens[:-1]])).log_softmax(-1)[0]
        return -sum(logits[k, token].item() for k, token in enumerate(tokens[1:])) / math.log(2)

    def test_釣り合いは日本語の文を読むbitsを同じ文を英語で読むbitsで割ったもの(self):
        result = measure(self.model, self.tokenizer, torch.device("cpu"), "float32", self.directory,
                         chosen=self.definition)
        read = {language: [self.bits(self.model, SENTENCES[key][column]) for key in SENTENCES]
                for column, language in enumerate(flores.LANGUAGES)}
        for language, expected in read.items():
            for found, wanted in zip(result["items"]["sentences"][language], expected, strict=True):
                self.assertAlmostEqual(found, wanted, places=4)
            self.assertAlmostEqual(result["languages"][language]["bits"], sum(expected), places=4)
        self.assertAlmostEqual(result["balance"], sum(read["jpn_Jpan"]) / sum(read["eng_Latn"]), places=5)
        self.assertGreater(result["standard_error"], 0)
        self.assertEqual(result["languages"]["jpn_Jpan"]["bytes"],
                         sum(len(japanese.encode("utf-8")) for _, japanese in SENTENCES.values()))
        self.assertEqual(result["artifacts"], [self.definition["flores"]["text"]])

    def test_二つの結果の釣り合いを同じ文で引き比べ標準誤差を添える(self):
        base = measure(self.model, self.tokenizer, torch.device("cpu"), "float32", self.directory,
                       chosen=self.definition)
        torch.manual_seed(1)
        other_model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=64))
        other = measure(other_model, self.tokenizer, torch.device("cpu"), "float32", self.directory,
                        chosen=self.definition)
        with patch.object(definition, "load", lambda name: self.definition):
            found = compare_results(base, other)
        self.assertEqual((found["base"], found["other"]), (base["balance"], other["balance"]))
        self.assertAlmostEqual(found["difference"], other["balance"] - base["balance"])
        self.assertEqual(found["readable"], abs(found["difference"]) > 2 * found["standard_error"])


if __name__ == "__main__":
    unittest.main()
