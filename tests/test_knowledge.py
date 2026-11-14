import gzip
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers", "language_model_evaluation")):
    raise unittest.SkipTest("Install training dependencies")

import olmes_measure
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.comparison import compare_results
from pretrain.knowledge import measure
from pretrain.model import LanguageModel, ModelConfig

QUESTION, ANSWER, WRONG = "Q: where does rain come from?\nA:", " the sky", " a cup"
DOCUMENT = "Rain falls on the hills."


def byte_tokenizer():
    """A tokenizer whose vocabulary holds only BOS, EOS and the bytes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=258, show_progress=False, special_tokens=["<|bos|>", "<|eos|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


def artifacts(directory):
    """A selection of one question, its answer at the label, and a text of one document, as the definition's
    artifacts are made: (the selection's name, the text's name)."""
    work = Path(tempfile.mkdtemp(dir=directory))
    (work / "requests").mkdir()
    asked = [{"doc_id": 0, "native_id": 0, "idx": idx, "label": 0, "doc": {"query": QUESTION},
              "request": {"context": QUESTION, "continuation": continuation}}
             for idx, continuation in enumerate((ANSWER, WRONG))]
    with gzip.open(work / "requests" / "rain.jsonl.gz", "wt", encoding="utf-8") as lines:
        lines.writelines(json.dumps(request) + "\n" for request in asked)
    (work / "tasks.json").write_text(json.dumps({"rain": {"primary_metric": "bits_per_byte_corr"}}))
    (work / "suites.json").write_text(json.dumps({"weather": ["rain"]}))
    requests = olmes_measure.seal(work, "requests", olmes="test", suites=["weather"])
    selection = olmes_measure.select(requests, directory, ["weather"])
    work = Path(tempfile.mkdtemp(dir=directory))
    (work / "texts" / "hills").mkdir(parents=True)
    with gzip.open(work / "texts" / "hills" / "val.jsonl.gz", "wt", encoding="utf-8") as lines:
        lines.write(json.dumps({"text": DOCUMENT}) + "\n")
    text = olmes_measure.seal(work, "paloma-val", revision="test")
    return selection.name, text.name


class KnowledgeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        torch.manual_seed(0)
        self.tokenizer = byte_tokenizer()
        self.model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=64))
        selection, text = artifacts(self.directory)
        self.definition = {"name": "toy", "olmes": {"suites": ["weather"], "selection": selection},
                           "paloma": {"text": text}, "compare": "mean_rate",
                           "categories": {"answers": {"olmes": ["weather"]}, "reads": {"paloma": ["hills"]}}}

    def bits(self, context, continuation):
        """The bits per byte of a continuation after its context, read by the model directly."""
        bos = self.tokenizer.token_to_id("<|bos|>")
        before = self.tokenizer.encode(context, add_special_tokens=False).ids
        after = self.tokenizer.encode(context + continuation, add_special_tokens=False).ids[len(before):]
        tokens = torch.tensor([[bos, *before, *after]])
        with torch.no_grad():
            logits = self.model(tokens[:, :-1]).log_softmax(-1)[0]
        nats = sum(logits[len(before) + k, token].item() for k, token in enumerate(after))
        return -nats / (math.log(2) * len(continuation.encode("utf-8")))

    def test_測るのは正解の答えと文章のbits_per_byteで区分ごとにまとめる(self):
        result = measure(self.model, self.tokenizer, torch.device("cpu"), "float32", self.directory,
                         chosen=self.definition)
        self.assertEqual(list(result["categories"]), ["answers", "reads"])
        self.assertAlmostEqual(result["tasks"]["rain"], self.bits(QUESTION, ANSWER), places=5)
        self.assertAlmostEqual(result["categories"]["answers"], result["tasks"]["rain"])
        # a document is read by itself after the BOS token
        self.assertAlmostEqual(result["domains"]["hills/hills"], self.bits("", DOCUMENT), places=5)
        self.assertAlmostEqual(result["categories"]["reads"], result["domains"]["hills/hills"])
        self.assertEqual(result["artifacts"], [self.definition["olmes"]["selection"], self.definition["paloma"]["text"]])
        # item by item: the question's bits per byte, and the document's bits and bytes
        self.assertEqual(result["items"]["questions"], {"rain": [result["tasks"]["rain"]]})
        (bits, size), = result["items"]["documents"]["hills/hills"]
        self.assertEqual(size, len(DOCUMENT.encode("utf-8")))
        self.assertAlmostEqual(bits / size, result["domains"]["hills/hills"])
        self.assertEqual(result["groups"], {"weather": ["rain"]})

    def test_定義が名指しする成果物と中身が違えば測らない(self):
        selection = self.directory / self.definition["olmes"]["selection"]
        task = next((selection / "tasks").iterdir())
        task.write_text(task.read_text().replace("sky", "sea"))
        with self.assertRaises(ValueError):
            measure(self.model, self.tokenizer, torch.device("cpu"), "float32", self.directory, chosen=self.definition)

    def test_同じ定義の二つの結果を定義の決まりで比べ区分ごとに標準誤差を添える(self):
        from language_model_evaluation import definition

        base = measure(self.model, self.tokenizer, torch.device("cpu"), "float32", self.directory, chosen=self.definition)
        torch.manual_seed(1)
        other_model = LanguageModel(258, ModelConfig(width=8, heads=2, layers=1, context=64))
        other = measure(other_model, self.tokenizer, torch.device("cpu"), "float32", self.directory,
                        chosen=self.definition)
        with patch.object(definition, "load", lambda name: self.definition):
            found = compare_results(base, other)
        self.assertAlmostEqual(found["improvement"],
                               definition.improvement(self.definition, other["categories"], base["categories"]))
        for name in ("answers", "reads"):
            self.assertEqual(found["categories"][name]["base"], base["categories"][name])
            # one question and one document: the score does not vary with the items drawn
            self.assertIsNone(found["categories"][name]["standard_error"])
            self.assertIsNone(found["categories"][name]["readable"])
        with self.assertRaisesRegex(ValueError, "different definitions"):
            compare_results({**base, "definition": "knowledge-en-1"}, other)


if __name__ == "__main__":
    unittest.main()
