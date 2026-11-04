import csv
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ("torch", "tokenizers", "jinja2", "yaml")):
    raise unittest.SkipTest("Install training and evaluation dependencies")

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.core import (
    answer_spans,
    evaluate_core,
    example_indices,
    fetch_bundle,
    fit_context,
    prompts,
    read_questions,
)


def byte_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=258, show_progress=False, special_tokens=["<|bos|>", "<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["abc"], trainer=trainer)
    return tokenizer


class NextTokenModel(torch.nn.Module):
    """Predict, after each token, the token `following` names for it, and after any other one `otherwise`."""

    def __init__(self, tokenizer, following, otherwise, context=64):
        super().__init__()
        self.config = SimpleNamespace(context=context)
        table = torch.full((tokenizer.get_vocab_size(),), tokenizer.token_to_id(otherwise))
        for before, after in following.items():
            table[tokenizer.token_to_id(before)] = tokenizer.token_to_id(after)
        self.register_buffer("table", table)
        self.size = tokenizer.get_vocab_size()

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, self.size), -10.0)
        return logits.scatter(-1, self.table[tokens][..., None], 10.0)


def write_bundle(directory, tasks):
    """A bundle of nanochat's layout with `tasks`: label to (type, shots, random baseline, questions)."""
    (directory / "eval_data").mkdir(parents=True)
    entries, rows = [], []
    for label, (kind, shots, baseline, questions) in tasks.items():
        (directory / "eval_data" / f"{label}.jsonl").write_text(
            "".join(json.dumps(question) + "\n" for question in questions), encoding="utf-8")
        entries.append({"label": label, "dataset_uri": f"{label}.jsonl", "num_fewshot": [shots], "icl_task_type": kind})
        rows.append({"Eval Task": label, "Random baseline": baseline})
    (directory / "core.yaml").write_text(json.dumps({"icl_tasks": entries}), encoding="utf-8")
    with (directory / "eval_meta_data.csv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=["Eval Task", "Random baseline"])
        writer.writeheader()
        writer.writerows(rows)


class CoreTests(unittest.TestCase):
    def test_renders_each_kind_of_question_as_nanochat_does(self):
        choice = {"query": "Q2", "choices": ["a", "b"], "gold": 0}
        example = {"query": "Q1", "choices": ["x", "y"], "gold": 1}
        self.assertEqual(prompts("multiple_choice", choice, " ", [example]), ["Q1 y\n\nQ2 a", "Q1 y\n\nQ2 b"])
        schema = {"context_options": ["C1", "C2"], "continuation": "end", "gold": 1}
        shown = {"context_options": ["A", "B"], "continuation": "fin", "gold": 0}
        self.assertEqual(prompts("schema", schema, " ", [shown]), ["A fin\n\nC1 end", "A fin\n\nC2 end"])
        modeled = {"context": "ctx  ", "continuation": "ans"}
        self.assertEqual(prompts("language_modeling", modeled, "\nAnswer: ", []), ["ctx\nAnswer:", "ctx\nAnswer: ans"])

    def test_reads_each_answer_after_what_the_choices_share_or_before_what_the_contexts_share(self):
        self.assertEqual(answer_spans("multiple_choice", [[0, 5, 6, 7], [0, 5, 8]]), [([0, 5, 6, 7], 2, 4), ([0, 5, 8], 2, 3)])
        self.assertEqual(answer_spans("schema", [[0, 1, 9, 9], [0, 2, 3, 9, 9]]), [([0, 1, 9, 9], 2, 4), ([0, 2, 3, 9, 9], 3, 5)])
        self.assertEqual(answer_spans("language_modeling", [[0, 1], [0, 1, 4, 5]]), [([0, 1, 4, 5], 2, 4)])
        with self.assertRaises(ValueError):
            answer_spans("language_modeling", [[0, 2], [0, 1, 4]])

    def test_draws_the_examples_nanochat_draws_from_the_other_questions(self):
        for index in (0, 7, 49):
            expected = random.Random(1234 + index).sample([other for other in range(50) if other != index], 10)
            self.assertEqual(example_indices(index, 50, 10), expected)
        self.assertEqual(example_indices(3, 50, 0), [])

    def test_asks_the_questions_in_nanochat_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.jsonl"
            path.write_text("".join(json.dumps({"n": n}) + "\n" for n in range(20)), encoding="utf-8")
            expected = [{"n": n} for n in range(20)]
            random.Random(1337).shuffle(expected)
            self.assertEqual(read_questions(path), expected)
            self.assertEqual(read_questions(path, limit=5), expected[:5])

    def test_keeps_the_last_tokens_of_a_question_longer_than_the_context(self):
        self.assertEqual(fit_context(list(range(10)), 7, 10, 6), (list(range(4, 10)), 3, 6))
        self.assertEqual(fit_context([1, 2, 3], 1, 3, 6), ([1, 2, 3], 1, 3))
        with self.assertRaises(ValueError):
            fit_context(list(range(10)), 3, 10, 6)

    def test_scores_choices_by_mean_loss_and_answers_by_greedy_prediction_and_centers_them(self):
        tokenizer = byte_tokenizer()
        # After "Q" comes a space, and after a space or "a" comes "a".
        model = NextTokenModel(tokenizer, {"Q": "Ġ", "Ġ": "a", "a": "a"}, "a")
        tasks = {
            "choices": ("multiple_choice", 0, 50.0, [
                {"query": "Q", "choices": ["aaa", "bbb"], "gold": 0},
                {"query": "Q", "choices": ["aaa", "bbb"], "gold": 1},
                {"query": "Q", "choices": ["bab", "aab"], "gold": 1},
            ]),
            "answers": ("language_modeling", 0, 0.0, [
                {"context": "Q", "continuation": "aa"},
                {"context": "Q", "continuation": "ab"},
            ]),
        }
        with tempfile.TemporaryDirectory() as directory:
            write_bundle(Path(directory), tasks)
            result = evaluate_core(model, tokenizer, torch.device("cpu"), "float32", directory, batch_tokens=8)
        choices, answers = result["tasks"]["choices"], result["tasks"]["answers"]
        order = list(range(3))
        random.Random(1337).shuffle(order)
        self.assertEqual(choices["correct"], "".join("0" if original == 1 else "1" for original in order))
        self.assertAlmostEqual(choices["accuracy"], 2 / 3)
        self.assertAlmostEqual(choices["centered"], (2 / 3 - 0.5) / 0.5)
        self.assertAlmostEqual(answers["accuracy"], 1 / 2)
        self.assertAlmostEqual(answers["centered"], 1 / 2)
        self.assertAlmostEqual(result["core"], ((2 / 3 - 0.5) / 0.5 + 1 / 2) / 2)

    def test_gives_the_same_answers_whatever_the_batch(self):
        tokenizer = byte_tokenizer()
        model = NextTokenModel(tokenizer, {"Q": "Ġ", "Ġ": "b", "b": "a"}, "a")
        questions = [{"query": "Q" * (index % 5 + 1), "choices": ["ba", "ab", "bb"], "gold": index % 3}
                     for index in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            write_bundle(Path(directory), {"choices": ("multiple_choice", 2, 33.3, questions)})
            alone = evaluate_core(model, tokenizer, torch.device("cpu"), "float32", directory, batch_tokens=1)
            together = evaluate_core(model, tokenizer, torch.device("cpu"), "float32", directory, batch_tokens=4096)
        self.assertEqual(alone["tasks"]["choices"]["correct"], together["tasks"]["choices"]["correct"])

    def test_reads_a_hugging_face_model_s_logits_within_its_context(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        from pretrain.core import HubModel

        torch.manual_seed(0)
        inner = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=258, n_positions=32))
        model = HubModel(inner).eval()
        tokens = torch.tensor([[0, 5, 6, 7]])
        self.assertEqual(model.config.context, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens), inner(tokens).logits))

    def test_uses_no_bundle_but_nanochat_s(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.urlopen", return_value=io.BytesIO(b"another file")
        ):
            with self.assertRaises(ValueError):
                fetch_bundle(Path(directory) / "core")
            self.assertFalse((Path(directory) / "core").exists())


if __name__ == "__main__":
    unittest.main()
