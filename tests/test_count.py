import collections
import contextlib
import dataclasses
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

if any(
    importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")
):
    raise unittest.SkipTest(
        "Install training dependencies with uv sync --extra train-cpu"
    )

import numpy as np
import torch

from pretrain.__main__ import main
from pretrain.count import CountInput, CountTable, build_table, history_keys
from pretrain.data import BOS, Corpus, prepare_data
from pretrain.model import LanguageModel, ModelConfig
from pretrain.training import TrainConfig, load_checkpoint, load_model, train

ORDERS = 4
CANDIDATES = 3


def counted(streams, start, orders):
    """Every history's followers, counted one position at a time."""
    counts = {length: collections.defaultdict(collections.Counter) for length in range(1, orders)}
    unigram = collections.Counter()
    for stream in streams:
        tokens = [int(token) for token in stream]
        unigram.update(tokens[1:])
        for position in range(len(tokens) - 1):
            for length in range(1, orders):
                earliest = position - length + 1
                if earliest < 0 or start in tokens[earliest + 1 : position + 1]:
                    continue
                counts[length][tuple(tokens[earliest : position + 1])][tokens[position + 1]] += 1
    return counts, unigram


def expected_rows(counts, min_history, min_count):
    """Each kept history's candidates, the most frequent first, with their shares."""
    rows = {}
    for length, histories in counts.items():
        for history, followers in histories.items():
            total = sum(followers.values())
            if total < min_history:
                continue
            kept = sorted(
                (token for token, count in followers.items() if count >= min_count),
                key=lambda token: (-followers[token], token),
            )[:CANDIDATES]
            if kept:
                rows[history] = [(token, followers[token] / total) for token in kept]
    return rows


class CountTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        documents = {
            "ja": ["あいう" * 8 + f"え{i % 3}お" * 3 for i in range(12)],
            "en": ["abc " * 8 + f"d{i % 4}e " * 3 for i in range(12)],
        }
        paths = {}
        for language, texts in documents.items():
            paths[language] = cls.root / f"{language}.jsonl"
            paths[language].write_text(
                "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
                encoding="utf-8",
            )
        cls.data = cls.root / "data"
        prepare_data(paths["ja"], paths["en"], cls.data, vocabulary_size=258, validation_fraction=0.25)
        cls.corpus = Corpus(cls.data)
        cls.start = cls.corpus.tokenizer.token_to_id(BOS)
        cls.streams = list(cls.corpus.streams["train"].values())
        cls.counts, cls.unigram = counted(cls.streams, cls.start, ORDERS)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def build(self, name, min_history=2, min_count=1, memory_gb=16):
        return build_table(
            [self.data], self.root / name, ORDERS, CANDIDATES, min_history, min_count, memory_gb, threads=2
        )

    def table_rows(self, table):
        """The table's rows by history, read back through the keys of the counted histories."""
        rows = {}
        for length, histories in self.counts.items():
            keys = table.tensors[f"keys_{length}"].numpy()
            for history in histories:
                key = history_keys(np.array(history, dtype=np.int64), length + 1)[-1][-1]
                place = np.searchsorted(keys, key)
                if place < len(keys) and keys[place] == key:
                    ids = table.tensors[f"candidates_{length}"][place].tolist()
                    shares = table.tensors[f"shares_{length}"][place].float().tolist()
                    rows[history] = [(token, share) for token, share in zip(ids, shares) if share > 0]
        return rows

    def assert_rows(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for history, candidates in expected.items():
            self.assertEqual([token for token, _ in actual[history]], [token for token, _ in candidates])
            for (_, got), (_, share) in zip(actual[history], candidates):
                self.assertAlmostEqual(got, share, places=3)

    def test_numpy_and_pytorch_give_each_history_the_same_key_and_different_histories_different_keys(self):
        tokens = np.random.default_rng(0).integers(0, 258, size=(3, 40))
        for length, (expected, actual) in enumerate(
            zip(history_keys(tokens, 7), history_keys(torch.from_numpy(tokens), 7)), 1
        ):
            np.testing.assert_array_equal(expected, actual.numpy())
            seen = {}
            for row, keys in zip(tokens, expected):
                for position in range(length - 1, len(row)):
                    seen.setdefault(tuple(row[position - length + 1 : position + 1]), set()).add(int(keys[position]))
            self.assertTrue(all(len(keys) == 1 for keys in seen.values()))
            self.assertEqual(len(set().union(*seen.values())), len(seen))

    def test_table_holds_the_most_frequent_followers_of_histories_within_documents(self):
        """表は、文書をまたがない履歴ごとに、多く続いたトークンとその割合を持つ。"""
        self.build("table")
        table = CountTable.load(self.root / "table")
        self.assert_rows(self.table_rows(table), expected_rows(self.counts, 2, 1))
        common = sorted(self.unigram, key=lambda token: (-self.unigram[token], token))[:CANDIDATES]
        self.assertEqual(table.tensors["candidates_0"].tolist(), common)
        total = sum(self.unigram.values())
        for token, share in zip(common, table.tensors["shares_0"].float().tolist()):
            self.assertAlmostEqual(share, self.unigram[token] / total, places=3)

    def test_rare_histories_and_rare_followers_stay_out_of_the_table(self):
        """出てくる回数が下限に届かない履歴と続きは、表に入れない。"""
        self.build("strict", min_history=6, min_count=3)
        table = CountTable.load(self.root / "strict")
        expected = expected_rows(self.counts, 6, 3)
        self.assertLess(len(expected), len(expected_rows(self.counts, 2, 1)))
        self.assert_rows(self.table_rows(table), expected)

    def test_the_command_builds_the_table_that_build_table_builds(self):
        """count-table コマンドは、同じ設定の build_table と同じ表を作る。"""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([
                "count-table", "--data", str(self.data), "--output", str(self.root / "command"),
                "--orders", str(ORDERS), "--candidates", str(CANDIDATES),
                "--min-history", "2", "--min-count", "1", "--threads", "1",
            ])
        self.build("direct")
        self.assertEqual(json.loads(output.getvalue())["digest"], CountTable.load(self.root / "direct").digest)

    def test_counting_in_several_passes_builds_the_same_table(self):
        """メモリに合わせて何回かに分けて数えても、同じ表を作る。"""
        self.build("whole")
        self.build("parts", memory_gb=1e-6)
        whole, parts = CountTable.load(self.root / "whole"), CountTable.load(self.root / "parts")
        self.assertEqual(whole.digest, parts.digest)

    def test_lookup_finds_each_positions_rows_and_leaves_out_histories_across_a_document_start(self):
        """引くときも、窓の中で文書の始まりより前にさかのぼる履歴は使わない。"""
        self.build("lookup")
        table = CountTable.load(self.root / "lookup")
        rows = expected_rows(self.counts, 2, 1)
        stream = self.streams[0]
        starts = np.flatnonzero(np.asarray(stream) == self.start)[1:4]
        windows = np.stack([np.asarray(stream[begin - 5 : begin + 11], dtype=np.int64) for begin in starts])
        ids, shares = CountInput(table).lookup(torch.from_numpy(windows))
        for row, window in enumerate(windows.tolist()):
            for position in range(len(window)):
                for length in range(1, ORDERS):
                    earliest = position - length + 1
                    history = tuple(window[max(earliest, 0) : position + 1])
                    usable = earliest >= 0 and self.start not in window[earliest + 1 : position + 1]
                    expected = rows.get(history, []) if usable else []
                    span = slice(length * CANDIDATES, (length + 1) * CANDIDATES)
                    got = [
                        (token, share * ORDERS)
                        for token, share in zip(ids[row, position, span].tolist(), shares[row, position, span].float().tolist())
                        if share > 0
                    ]
                    self.assertEqual([token for token, _ in got], [token for token, _ in expected])
                    for (_, value), (_, share) in zip(got, expected):
                        self.assertAlmostEqual(value, share, places=3)

    def config(self):
        return ModelConfig(
            width=24, heads=4, layers=1, context=16, architecture="llama",
            document_start=self.start, count_table=True,
        )

    def test_model_adds_the_share_weighted_embeddings_of_the_counted_followers(self):
        """モデルは、各トークンの埋め込みに、数えた続きの埋め込みを割合で重み付けして足す。"""
        self.build("model")
        table = CountTable.load(self.root / "model")
        model = LanguageModel(258, self.config(), count_table=table)
        tokens = torch.from_numpy(np.asarray(self.streams[0][:16], dtype=np.int64))[None]
        ids, shares = model.count.lookup(tokens)
        expected = (model.embedding.weight[ids.long()] * shares.float()[..., None]).sum(-2)
        torch.testing.assert_close(model.count(tokens, model.embedding.weight), expected)
        without = LanguageModel(258, dataclasses.replace(self.config(), count_table=False))
        without.load_state_dict(model.state_dict())
        self.assertFalse(torch.equal(model(tokens), without(tokens)))

    def test_training_keeps_the_table_in_checkpoints_and_a_loaded_model_predicts_the_same(self):
        """学習は表をチェックポイントに入れ、読み込んだモデルは同じ予測をする。"""
        self.build("trained")
        table = CountTable.load(self.root / "trained")
        config = TrainConfig(
            steps=4, batch_size=2, accumulation_steps=1, learning_rate=0.01, min_learning_rate=0.001,
            warmup_steps=1, eval_every=2, save_every=2, eval_batches=1, log_every=2,
        )
        output = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.data, output, self.config(), config, torch.device("cpu"), max_steps=2, count_table=table)
            train(self.data, output, self.config(), config, torch.device("cpu"))
        saved = load_checkpoint(output / "last.pt")
        self.assertEqual(saved["step"], 4)
        model, _, _ = load_model(output / "last.pt", torch.device("cpu"))
        self.assertEqual(model.count_digest, table.digest)
        rebuilt = LanguageModel(258, self.config(), count_table=table)
        rebuilt.load_state_dict(saved["model"])
        tokens = torch.from_numpy(np.asarray(self.streams[1][:16], dtype=np.int64))[None]
        torch.testing.assert_close(model(tokens), rebuilt(tokens))

    def generating_model(self, name):
        """A model trained until it continues the corpus, so its text runs along the table's n-grams."""
        self.build(name)
        config = TrainConfig(
            steps=60, batch_size=4, accumulation_steps=1, learning_rate=0.01, min_learning_rate=0.001,
            warmup_steps=1, eval_every=60, save_every=60, eval_batches=1, log_every=60,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                self.data, self.root / f"{name}-run", self.config(), config, torch.device("cpu"),
                count_table=CountTable.load(self.root / name),
            )
        model, _, _ = load_model(self.root / f"{name}-run" / "last.pt", torch.device("cpu"))
        return model.eval()

    def recomputed(self, model, tokens, steps):
        """The likeliest tokens when every step reads the whole latest window again."""
        tokens, chosen = list(tokens), []
        for _ in range(steps):
            with torch.no_grad():
                logits = model(torch.tensor([tokens[-model.config.context :]]))[0, -1]
            logits[self.start] = -torch.inf
            chosen.append(logits.argmax().item())
            tokens.append(chosen[-1])
        return chosen

    def test_generation_that_reuses_computed_parts_picks_what_reading_everything_again_picks(self):
        """計算済みの部分を使う生成も、文脈を超えた後も、毎回すべて読み直したときと同じトークンを選ぶ。"""
        model = self.generating_model("generation")
        tokenizer = self.corpus.tokenizer
        prompt = "あいう"
        generated = list(model.sample_tokens(tokenizer, prompt, max_new_tokens=20, temperature=0, stops=[]))
        tokens = [self.start, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        self.assertEqual(generated, self.recomputed(model, tokens, 20))

    def recorded_embeddings(self, model):
        """Wrap the model's embedding so that each call's new rows are kept."""
        calls, embed = [], model.embed

        def recording(tokens, history=None):
            embedded = embed(tokens, history)
            calls.append(embedded[:, -1].detach().clone())
            return embedded

        model.embed = recording
        return calls

    def assert_steps_read_their_whole_sequence(self, model, calls, sequences):
        """Each step's new embedding is the one its whole sequence so far gives, and history mattered."""
        mattered = False
        for step, embedded in enumerate(calls[1:], 1):
            for row, sequence in enumerate(sequences):
                read = torch.tensor([sequence[: -len(calls) + step]])
                with torch.no_grad():
                    whole = LanguageModel.embed(model, read)[0, -1]
                    alone = LanguageModel.embed(model, read[:, -1:])[0, -1]
                torch.testing.assert_close(embedded[row], whole)
                mattered |= not torch.allclose(whole, alone)
        self.assertTrue(mattered)

    def allowed_only(self, text):
        """Every token but the ones of text, so that generation stays on the table's n-grams."""
        allowed = set(self.corpus.tokenizer.encode(text, add_special_tokens=False).ids)
        return [token for token in range(258) if token not in allowed]

    def test_each_generation_step_reads_the_tokens_before_it(self):
        """生成の各ステップは、計算済みのトークンを読み直さなくても、その前のトークンで表を引く。"""
        model = self.generating_model("history")
        tokenizer = self.corpus.tokenizer
        banned = self.allowed_only("abc ")
        calls = self.recorded_embeddings(model)
        generated = list(model.sample_tokens(tokenizer, "abc", max_new_tokens=10, temperature=0, stops=[], banned=banned))
        prompt = [self.start, *tokenizer.encode("abc", add_special_tokens=False).ids]
        self.assert_steps_read_their_whole_sequence(model, calls, [prompt + generated])
        sequences = [
            [self.start, *tokenizer.encode("ab", add_special_tokens=False).ids],
            [self.start, *tokenizer.encode("abc ab", add_special_tokens=False).ids],
        ]
        calls.clear()
        continuations = model.continue_sequences(sequences, 6, (), banned=[self.start, *banned])
        whole = [sequence + continuation for sequence, continuation in zip(sequences, continuations)]
        self.assert_steps_read_their_whole_sequence(model, calls, whole)

    def test_a_model_that_reads_a_table_is_refused_without_one_or_with_another_tokenizers(self):
        """表を読むモデルは、表がないときや別のトークナイザーの表では動かない。"""
        self.build("refusals")
        table = CountTable.load(self.root / "refusals")
        config = TrainConfig(steps=2, batch_size=2, accumulation_steps=1, warmup_steps=1)
        with self.assertRaisesRegex(ValueError, "reads a count table"):
            train(self.data, self.root / "no-table", self.config(), config, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "reads no count table"):
            LanguageModel(258, dataclasses.replace(self.config(), count_table=False), count_table=table)
        with self.assertRaisesRegex(RuntimeError, "load the table"):
            LanguageModel(258, self.config())(torch.zeros(1, 4, dtype=torch.long))
        other = CountTable(dict(table.manifest, tokenizer="0" * 64), table.tensors)
        with self.assertRaisesRegex(ValueError, "another tokenizer"):
            train(self.data, self.root / "other", self.config(), config, torch.device("cpu"), count_table=other)


if __name__ == "__main__":
    unittest.main()
