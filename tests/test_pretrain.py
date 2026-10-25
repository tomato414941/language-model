import contextlib
import dataclasses
import importlib.util
import io
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if any(
    importlib.util.find_spec(name) is None for name in ("numpy", "torch", "tokenizers")
):
    raise unittest.SkipTest(
        "Install training dependencies with uv sync --extra train-cpu"
    )

import numpy as np
import torch

from pretrain.data import Corpus, interleaved, prepare_data, token_bytes
from pretrain.growth import grow_checkpoint
from pretrain.model import (
    RESET,
    Experts,
    LanguageModel,
    ModelConfig,
    chunk_gated_delta_rule,
    chunked_delta_rule,
    document_bounds,
    kernel_delta_rule,
    recurrent_delta_rule,
)
from pretrain.training import (
    TrainConfig,
    evaluate,
    evaluate_windows,
    learning_rate,
    load_checkpoint,
    load_model,
    make_optimizer,
    select_device,
    train,
)


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.japanese = self.root / "ja.jsonl"
        self.english = self.root / "en.jsonl"
        self.documents = {
            "ja": ["あいう" * 30 + f" 終わり{i}" for i in range(12)],
            "en": ["abc " * 40 + f" ending{i}" for i in range(12)],
        }
        for language, path in (("ja", self.japanese), ("en", self.english)):
            with path.open("w", encoding="utf-8") as destination:
                for text in self.documents[language] + [self.documents[language][0]]:
                    destination.write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    )
        self.data = self.root / "data"
        prepare_data(
            self.japanese,
            self.english,
            self.data,
            vocabulary_size=258,
            validation_fraction=0.25,
        )
        self.corpus = Corpus(self.data)
        self.model_config = ModelConfig(width=24, heads=4, layers=1, context=16)

    def config(self, steps=8):
        return TrainConfig(
            steps=steps,
            batch_size=4,
            accumulation_steps=2,
            learning_rate=0.01,
            min_learning_rate=0.001,
            warmup_steps=1,
            eval_every=4,
            save_every=4,
            eval_batches=2,
            log_every=4,
        )

    def run_training(self, name, config=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(
                self.data,
                self.root / name,
                self.model_config,
                config or self.config(),
                torch.device("cpu"),
                **kwargs,
            )

    def test_byte_tokenizer_round_trips_unseen_japanese_and_english(self):
        """学習にない漢字・絵文字・英単語も元の文章に復元する。"""
        text = "初めての漢字と🌱。An Unseen English Sentence!\n改行も維持する。"
        encoded = self.corpus.tokenizer.encode(text)
        self.assertEqual(self.corpus.tokenizer.decode(encoded.ids), text)

    def test_preparation_deduplicates_documents_and_holds_out_each_language(self):
        """日英それぞれの文書を重複除去して学習用と検証用に分割する。"""
        self.assertEqual(self.corpus.manifest["duplicates_removed"], 2)
        eos = self.corpus.tokenizer.token_to_id("<|eos|>")
        for language in ("ja", "en"):
            splits = {}
            for split in ("train", "validation"):
                stream = self.corpus.streams[split][language]
                boundaries = np.flatnonzero(stream == eos)
                decoded, start = set(), 0
                for end in boundaries:
                    decoded.add(
                        self.corpus.tokenizer.decode(stream[start : end + 1].tolist())
                    )
                    start = end + 1
                splits[split] = decoded
            self.assertTrue(splits["train"])
            self.assertTrue(splits["validation"])
            self.assertTrue(splits["train"].isdisjoint(splits["validation"]))
            self.assertEqual(
                splits["train"] | splits["validation"], set(self.documents[language])
            )

    def test_prediction_uses_only_the_preceding_tokens(self):
        """未来の入力が変わっても、それより前の次トークン予測を維持する。"""
        model = LanguageModel(258, self.model_config).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 200, 201]])
        torch.testing.assert_close(
            model(first)[:, :3], model(second)[:, :3], rtol=0, atol=1e-6
        )

    def test_training_improves_held_out_predictions_in_both_languages(self):
        """日英の繰り返しパターンを学習して、両言語の検証損失を下げる。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        before = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        after = self.run_training("learn", self.config(steps=24))["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_resume_continues_the_same_training_sequence(self):
        """途中保存から再開しても、連続して学習したときと同じ重み・学習量を得る。"""
        self.run_training("continuous")
        partial = self.run_training("resumed", max_steps=3)
        self.assertEqual(partial["step"], 3)
        self.run_training("resumed", resume=self.root / "resumed" / "last.pt")
        continuous = load_checkpoint(self.root / "continuous" / "last.pt")
        resumed = load_checkpoint(self.root / "resumed" / "last.pt")
        self.assertEqual(resumed["step"], 8)
        self.assertEqual(resumed["tokens_seen"], 8 * 4 * 2 * 16)
        self.assertEqual(continuous["validation"], resumed["validation"])
        for name, parameter in continuous["model"].items():
            torch.testing.assert_close(
                parameter, resumed["model"][name], rtol=0, atol=0
            )

    def test_batched_greedy_continuations_match_generating_each_prompt_alone(self):
        """長さの違う入力をまとめて貪欲生成しても、1件ずつ生成したときと同じトークンを返す。"""
        tokenizer = self.corpus.tokenizer
        bos, eos = tokenizer.token_to_id("<|bos|>"), tokenizer.token_to_id("<|eos|>")
        prompts = ["日本語とEnglish", "abc", "あいう終わり"]
        sequences = [
            [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            for prompt in prompts
        ]
        for config in (
            ModelConfig(width=24, heads=4, layers=2, context=64),
            ModelConfig(
                width=24,
                heads=4,
                kv_heads=2,
                layers=2,
                context=64,
                architecture="llama",
                intermediate_size=64,
            ),
            *(
                ModelConfig(
                    width=24,
                    heads=4,
                    kv_heads=2,
                    layers=2,
                    context=64,
                    architecture="llama",
                    intermediate_size=64,
                    qk_norm=True,
                    attention_gate=gate,
                    document_start=bos,
                )
                for gate in ("headwise", "elementwise")
            ),
            ModelConfig(
                width=24,
                heads=4,
                kv_heads=2,
                layers=4,
                context=64,
                architecture="llama",
                intermediate_size=64,
                qk_norm=True,
                attention_gate="headwise",
                document_start=bos,
                full_attention_every=2,
                experts=4,
                active_experts=2,
                prediction_depth=1,
            ),
        ):
            torch.manual_seed(3)
            model = LanguageModel(258, config).eval()
            # Larger weights make each prediction depend on the earlier positions.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.normal_(std=0.3)
            alone = [
                [
                    token
                    for token in model.sample_tokens(
                        tokenizer, prompt, 20, temperature=0, top_k=0
                    )
                    if token != eos
                ]
                for prompt in prompts
            ]
            self.assertEqual(
                model.continue_greedily(sequences, 20, eos, banned=[bos]), alone
            )

    def test_一つの入力の生成は計算済みの部分を使い文脈を超えても計算し直したときと同じトークンを出す(self):
        """一つの入力から生成するときは計算済みの部分を使い回し、文脈の長さを超えて窓がずれても、毎回計算し直したときと同じトークンを出す。"""
        tokenizer = self.corpus.tokenizer
        bos = tokenizer.token_to_id("<|bos|>")
        prompt = "自然科学の文章"
        for every in (None, 2):
            torch.manual_seed(3)
            config = ModelConfig(
                width=24, heads=4, kv_heads=2, layers=4, context=32, architecture="llama",
                qk_norm=True, attention_gate="headwise", document_start=bos, full_attention_every=every,
            )
            model = LanguageModel(258, config).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.normal_(std=0.3)
            sampled = list(model.sample_tokens(tokenizer, prompt, 60, temperature=0, top_k=0, stops=[]))
            tokens = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            recomputed = []
            with torch.no_grad():
                for _ in range(60):
                    logits = model(torch.tensor([tokens[-32:]]))[0, -1]
                    logits[bos] = -torch.inf
                    recomputed.append(logits.argmax().item())
                    tokens.append(recomputed[-1])
            self.assertEqual(sampled, recomputed)

    def test_終わったと判定された入力だけ生成を止める(self):
        """終わったと判定された入力の生成だけを止め、ほかの入力は同じトークンを生成し続ける。"""
        torch.manual_seed(3)
        model = LanguageModel(
            258, ModelConfig(width=24, heads=4, layers=2, context=64)
        ).eval()
        sequences = [[0, 5, 6], [0, 7]]
        # No token is the end here, so only the limit and the judgment stop them.
        whole = model.continue_greedily(sequences, 6, -1)
        stopped = model.continue_greedily(
            sequences, 6, -1, finished=lambda row, tokens: row == 0 and len(tokens) == 2
        )
        self.assertEqual(stopped, [whole[0][:2], whole[1]])
        self.assertEqual(len(whole[1]), 6)

    def test_まとめた生成は温度に応じて選び同じ乱数なら同じ結果を返す(self):
        """まとめた生成は、上位の確率の範囲が狭いと最も確からしい単語を選び、同じ乱数の種なら同じ結果を返す。"""
        torch.manual_seed(3)
        model = LanguageModel(
            258, ModelConfig(width=24, heads=4, layers=2, context=64)
        ).eval()
        sequences = [[0, 5, 6], [0, 7]]
        greedy = model.continue_greedily(sequences, 6, -1)
        narrow = model.continue_sequences(
            sequences, 6, -1, temperature=1.0, top_p=1e-9, generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(narrow, greedy)
        sampled = [
            model.continue_sequences(sequences, 6, -1, temperature=1.0, generator=torch.Generator().manual_seed(1))
            for _ in range(2)
        ]
        self.assertEqual(sampled[0], sampled[1])
        self.assertEqual([len(row) for row in sampled[0]], [6, 6])
        with self.assertRaisesRegex(ValueError, "top_p"):
            model.continue_sequences(sequences, 6, -1, temperature=1.0, top_p=0)

    def test_終わりの記号のどれかが出た入力の生成を止める(self):
        """終わりの記号を複数渡すと、そのどれかを生成した入力の生成をそこで止める。"""
        torch.manual_seed(3)
        model = LanguageModel(
            258, ModelConfig(width=24, heads=4, layers=2, context=64)
        ).eval()
        sequences = [[0, 5, 6], [0, 7]]
        whole = model.continue_greedily(sequences, 6, -1)
        ends = {whole[0][-1], -1}
        stopped = model.continue_greedily(sequences, 6, ends)
        for row, tokens in zip(stopped, whole):
            cut = next(index for index, token in enumerate([*tokens, -1]) if token in ends)
            self.assertEqual(row, tokens[:cut])
        self.assertLess(len(stopped[0]), len(whole[0]))

    def test_文書を区切ると同じ窓に詰めた前の文書を見ずに予測する(self):
        """文書の始まりの記号を指定すると、同じ窓に詰めた前の文書を見ずに、その文書だけのときと同じ予測をする。"""
        bos = self.corpus.tokenizer.token_to_id("<|bos|>")
        for config in (
            dataclasses.replace(self.llama_config(), document_start=bos),
            dataclasses.replace(
                self.llama_config(),
                layers=4,
                document_start=bos,
                full_attention_every=2,
                experts=4,
                active_experts=2,
            ),
        ):
            torch.manual_seed(5)
            model = LanguageModel(258, config).eval()
            # Larger weights make each prediction depend on the earlier positions.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if not name.endswith(("decay_rate", "decay_step")):
                        parameter.normal_(std=0.3)
            earlier, later = [bos, 5, 6, 7, 8], [bos, 9, 10, 11]
            packed = model(torch.tensor([earlier + later]))[0, len(earlier) :]
            torch.testing.assert_close(
                packed, model(torch.tensor([later]))[0], rtol=1e-4, atol=1e-4
            )
            unseparated = LanguageModel(
                258, dataclasses.replace(config, document_start=None)
            ).eval()
            unseparated.load_state_dict(model.state_dict())
            mixed = unseparated(torch.tensor([earlier + later]))[0, len(earlier) :]
            self.assertFalse(torch.allclose(mixed, packed, atol=1e-3))

    def test_文書の境目を平らにした並びの位置で表す(self):
        """各行の文書を、行の始まりも区切りとして平らにした並びの位置で表し、空の文書で数を2の累乗にそろえる。"""
        starts = torch.tensor([[False, False, True, False], [True, False, False, True]])
        self.assertEqual(document_bounds(starts).tolist(), [0, 2, 4, 7, 8])
        starts = torch.tensor([[False, True, False, True], [False, False, False, False]])
        self.assertEqual(document_bounds(starts).tolist(), [0, 1, 3, 4, 8])
        self.assertEqual(document_bounds(torch.zeros(3, 4, dtype=torch.bool)).tolist(), [0, 4, 8, 12, 12])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_GPUでは文書ごとの注意をマスクと同じ結果で速い計算に任せる(self):
        """GPUの半精度では文書ごとの注意をFlashAttentionで計算し、密なマスクと同じ損失と勾配を返す。"""
        bos = self.corpus.tokenizer.token_to_id("<|bos|>")
        # FlashAttention needs heads at least 8 wide.
        config = dataclasses.replace(
            self.llama_config(), width=64, context=64, document_start=bos, qk_norm=True,
            attention_gate="headwise", prediction_depth=1,
        )
        torch.manual_seed(0)
        model = LanguageModel(258, config).cuda()
        tokens = torch.randint(2, 258, (2, config.context + 1), device="cuda")
        tokens[0, [0, 5, 40]] = bos
        tokens[1, 17] = bos
        results = []
        for flash in (True, False):
            model.zero_grad()
            with (
                patch("pretrain.model.half_precision", return_value=None) if not flash else contextlib.nullcontext(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                loss = model(tokens[:, :-1], tokens[:, 1:])
            loss.backward()
            results.append((loss.item(), [p.grad.float().clone() for p in model.parameters()]))
        self.assertAlmostEqual(results[0][0], results[1][0], places=2)
        for flash, dense in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(flash, dense, rtol=0.05, atol=2e-3)

    def test_文書の始まりの記号がないモデルでは文書を区切った速さの測定を断る(self):
        """文書の始まりの記号を持たないモデルでは、文書の区切りを入れた学習の速さの測定を断る。"""
        from pretrain.throughput import measure_training

        config = {"model": dataclasses.asdict(self.llama_config()), "training": {"steps": 2, "warmup_steps": 1}}
        with self.assertRaisesRegex(ValueError, "document_start"):
            measure_training(config, vocabulary_size=258, document_length=100)

    def test_文書の区切りには文書を始める記号を求める(self):
        """文書の始まりとして、語彙の文書開始記号以外を指定した学習を断る。"""
        bos = self.corpus.tokenizer.token_to_id("<|bos|>")
        self.model_config = dataclasses.replace(self.llama_config(), document_start=bos + 1)
        with self.assertRaisesRegex(ValueError, "document_start"):
            self.run_training("other-start")

    def test_QKの正規化と注意のゲートと文書の区切りを入れて両言語を学ぶ(self):
        """クエリとキーの正規化、注意の出力のゲート、文書の区切りを入れても、両言語の検証損失を下げる。"""
        bos = self.corpus.tokenizer.token_to_id("<|bos|>")
        for gate in ("headwise", "elementwise"):
            self.model_config = dataclasses.replace(
                self.llama_config(), qk_norm=True, attention_gate=gate, document_start=bos
            )
            torch.manual_seed(42)
            before = evaluate(
                LanguageModel(258, self.model_config),
                self.corpus,
                torch.device("cpu"),
                batches=2,
                batch_size=4,
            )
            after = self.run_training(f"gated-{gate}", self.config(steps=24))["validation"]
            for language in ("ja", "en"):
                self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_注意の設定の誤りを学習の前に知らせる(self):
        """未知のゲートの種類や、llama 以外での新しい注意の設定を断る。"""
        with self.assertRaisesRegex(ValueError, "attention_gate"):
            dataclasses.replace(self.llama_config(), attention_gate="scalar")
        with self.assertRaisesRegex(ValueError, "require llama"):
            ModelConfig(width=24, heads=4, layers=1, context=16, qk_norm=True)
        with self.assertRaisesRegex(ValueError, "active_experts"):
            dataclasses.replace(self.llama_config(), experts=4)
        with self.assertRaisesRegex(ValueError, "require experts"):
            dataclasses.replace(self.llama_config(), expert_size=8)
        with self.assertRaisesRegex(ValueError, "prediction_depth"):
            dataclasses.replace(self.llama_config(), prediction_depth=5)

    def test_差分規則をまとめて解いても一つずつ解いたときと同じ出力と状態を得る(self):
        """ゲート付き差分規則をチャンクごとにまとめて解いても、文書の区切りと初めの状態を含めて、一位置ずつ解いたときと同じ出力と最後の状態を得る。"""
        torch.manual_seed(0)
        batch, heads, length, width = 2, 3, 150, 8
        query, key = (
            torch.nn.functional.normalize(torch.randn(batch, heads, length, width), dim=-1)
            for _ in range(2)
        )
        value = torch.randn(batch, heads, length, 5)
        decay = -torch.rand(batch, heads, length) * 0.5
        decay[:, :, 70] = RESET
        beta = torch.rand(batch, heads, length)
        state = torch.randn(batch, heads, width, 5)
        recurrent = recurrent_delta_rule(query, key, value, decay, beta, state)
        chunked = chunked_delta_rule(query, key, value, decay, beta, state)
        for expected, actual in zip(recurrent, chunked, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(
        torch.cuda.is_available() and chunk_gated_delta_rule is not None,
        "CUDA GPU and Gated DeltaNet kernels required",
    )
    def test_GPUでは差分規則を専用の計算で解きまとめて解いたときと同じ出力と状態と勾配を得る(self):
        """GPUでは、ゲート付き差分規則を Gated DeltaNet の専用の計算で解き、文書の区切りと初めの状態を含めて、まとめて解いたときと同じ出力・最後の状態・勾配を得る。"""
        torch.manual_seed(0)
        batch, heads, length, width = 2, 4, 300, 64
        query, key = (
            torch.nn.functional.normalize(torch.randn(batch, length, heads, width, device="cuda"), dim=-1)
            for _ in range(2)
        )
        value = torch.randn(batch, length, heads, width, device="cuda")
        decay = -torch.rand(batch, length, heads, device="cuda") * 0.5
        decay[:, 130] = RESET
        beta = torch.rand(batch, length, heads, device="cuda")
        state = torch.randn(batch, heads, width, width, device="cuda")
        inputs = [query, key, value, decay, beta, state]
        for tensor in inputs:
            tensor.requires_grad_()
        kernel = kernel_delta_rule(query, key, value, decay, beta, state, True)
        outputs, final = chunked_delta_rule(
            *(part.transpose(1, 2) for part in (query, key, value, decay, beta)), state
        )
        chunked = (outputs.transpose(1, 2), final)
        for actual, expected in zip(kernel, chunked, strict=True):
            torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
        gradients = [
            torch.autograd.grad(
                result[0].float().square().mean() + result[1].float().square().mean(), inputs
            )
            for result in (kernel, chunked)
        ]
        for actual, expected in zip(*gradients, strict=True):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-3)

    def test_専門家の混合は各単語を点数の高い専門家に送り点数で混ぜる(self):
        """専門家の混合は、各単語を点数の高い専門家だけに送り、その出力を点数の比で混ぜ、偏りと点数の大きさを抑える損失を返す。"""
        torch.manual_seed(0)
        experts = Experts(width=8, count=4, active=2, size=6)
        hidden = torch.randn(3, 5, 8)
        mixed, loss = experts(hidden)
        tokens = hidden.reshape(-1, 8)
        probabilities = experts.router(tokens).softmax(-1)
        for token, output, scores in zip(tokens, mixed.reshape(-1, 8), probabilities):
            top, chosen = scores.topk(2)
            expected = sum(
                weight / top.sum()
                * (torch.nn.functional.silu(token @ experts.gate[index]) * (token @ experts.up[index]))
                @ experts.down[index]
                for weight, index in zip(top, chosen)
            )
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
        # Even routing makes the balancing loss 1, its least value, weighted by 0.01.
        self.assertGreater(loss.item(), 0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_GPUでは専門家をまとめて計算し一つずつと同じ結果を返す(self):
        """GPUのbfloat16では全専門家の積をまとめて計算し、一つずつ計算したときと同じ出力と勾配を返す。"""
        torch.manual_seed(0)
        experts = Experts(width=64, count=8, active=2, size=32).cuda()
        hidden = torch.randn(4, 50, 64, device="cuda")
        results = []
        for grouped in (True, False):
            experts.zero_grad()
            with (
                patch("pretrain.model.half_precision", return_value=None) if not grouped else contextlib.nullcontext(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                mixed, loss = experts(hidden)
            (mixed.float().square().mean() + loss).backward()
            results.append((mixed.float(), [p.grad.float().clone() for p in experts.parameters()]))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0.05, atol=2e-3)
        for grouped, separate in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(grouped, separate, rtol=0.05, atol=2e-3)

    def test_さらに先の予測は学習の目的にだけ加わる(self):
        """さらに先の単語の予測は学習中の目的にだけ加わり、評価の損失は次の単語の予測だけで測る。"""
        torch.manual_seed(0)
        model = LanguageModel(258, dataclasses.replace(self.llama_config(), prediction_depth=2))
        tokens = torch.randint(2, 258, (2, 17))
        x, y = tokens[:, :-1], tokens[:, 1:]
        model.eval()
        evaluated = model(x, y)
        expected = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        torch.testing.assert_close(evaluated, expected)
        model.train()
        trained = model(x, y)
        self.assertGreater(trained.item(), evaluated.item())
        trained.backward()
        for name, parameter in model.predictors.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_線形注意と専門家とさらに先の予測を入れて両言語を学ぶ(self):
        """Gated DeltaNet の層、専門家の混合、さらに先の予測を入れても、両言語の検証損失を下げる。"""
        bos = self.corpus.tokenizer.token_to_id("<|bos|>")
        self.model_config = dataclasses.replace(
            self.llama_config(),
            layers=4,
            qk_norm=True,
            attention_gate="headwise",
            document_start=bos,
            full_attention_every=2,
            experts=4,
            active_experts=2,
            prediction_depth=1,
        )
        torch.manual_seed(42)
        before = evaluate(
            LanguageModel(258, self.model_config),
            self.corpus,
            torch.device("cpu"),
            batches=2,
            batch_size=4,
        )
        after = self.run_training("modern-2026", self.config(steps=24))["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_checkpoint_restores_predictions_and_seeded_long_prompt_generation(self):
        """保存した重みと語彙を復元し、長い日英の接頭辞を保って再現可能に生成する。"""
        self.run_training("generation", max_steps=2)
        path = self.root / "generation" / "last.pt"
        first, tokenizer, _ = load_model(path, torch.device("cpu"))
        second, restored_tokenizer, _ = load_model(path, torch.device("cpu"))
        inputs = torch.tensor([[10, 20, 30]])
        torch.testing.assert_close(first(inputs), second(inputs), rtol=0, atol=0)
        prompt = "日本語とEnglishの長い文章です。" * 5
        output = first.generate(tokenizer, prompt, max_new_tokens=12, seed=9)
        self.assertTrue(output.startswith(prompt))
        self.assertEqual(
            output,
            second.generate(restored_tokenizer, prompt, max_new_tokens=12, seed=9),
        )

    def test_rotary_grouped_attention_learns_both_languages_causally(self):
        self.model_config = ModelConfig(
            width=24,
            heads=4,
            kv_heads=2,
            layers=2,
            context=16,
            architecture="llama",
            intermediate_size=64,
        )
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 200, 201]])
        torch.testing.assert_close(
            model(first)[:, :3], model(second)[:, :3], rtol=0, atol=1e-6
        )
        before = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        after = self.run_training("modern", self.config(steps=24))["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)

    def test_rotary_model_resumes_the_same_updates_and_restores_generation(self):
        self.model_config = ModelConfig(
            width=24,
            heads=4,
            kv_heads=2,
            layers=2,
            context=16,
            architecture="llama",
            intermediate_size=64,
        )
        self.run_training("modern-continuous")
        self.run_training("modern-resumed", max_steps=3)
        self.run_training("modern-resumed", resume=self.root / "modern-resumed/last.pt")
        continuous, tokenizer, first = load_model(
            self.root / "modern-continuous/last.pt", torch.device("cpu")
        )
        resumed, _, second = load_model(
            self.root / "modern-resumed/last.pt", torch.device("cpu")
        )
        self.assertEqual(first["tokens_seen"], second["tokens_seen"])
        for name, parameter in continuous.state_dict().items():
            torch.testing.assert_close(
                parameter, resumed.state_dict()[name], rtol=0, atol=0
            )
        prompt = "日本語とEnglish"
        self.assertEqual(
            continuous.generate(tokenizer, prompt, max_new_tokens=8, seed=7),
            resumed.generate(tokenizer, prompt, max_new_tokens=8, seed=7),
        )

    def test_new_training_phase_keeps_learned_predictions_on_additional_data(self):
        self.run_training("phase-one", max_steps=3)
        source = self.root / "phase-one/last.pt"
        old = load_checkpoint(source)
        new_data = self.root / "additional-data"
        prepare_data(
            self.japanese,
            self.english,
            new_data,
            vocabulary_size=258,
            seed=99,
            tokenizer_path=self.data / "tokenizer.json",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                new_data,
                self.root / "phase-two",
                self.model_config,
                self.config(steps=12),
                torch.device("cpu"),
                initialize_from=source,
                stop_requested=lambda: True,
            )
        initialized = load_checkpoint(self.root / "phase-two/last.pt")
        self.assertEqual(initialized["step"], 0)
        self.assertEqual(
            initialized["initialization"]["source_tokens_seen"], old["tokens_seen"]
        )
        self.assertEqual(
            initialized["dataset_fingerprint"], Corpus(new_data).fingerprint
        )
        for name, parameter in old["model"].items():
            torch.testing.assert_close(
                parameter, initialized["model"][name], rtol=0, atol=0
            )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                new_data,
                self.root / "phase-two",
                self.model_config,
                self.config(steps=12),
                torch.device("cpu"),
                resume=self.root / "phase-two/last.pt",
                max_steps=2,
            )
        self.assertEqual(result["step"], 2)
        self.assertEqual(
            Corpus(new_data).tokenizer.to_str(), self.corpus.tokenizer.to_str()
        )

    def wsd_config(self, steps=8, **changes):
        return TrainConfig(
            **{
                **self.config(steps).__dict__,
                "schedule": "wsd",
                "decay_steps": 2,
                "sampling": "sequential",
                **changes,
            }
        )

    def test_文脈を伸ばす段階は文脈長と回転の基準だけを変えて続ける(self):
        """保存した学習を、文脈長と回転位置埋め込みの基準だけを変えて続け、ほかの設定の違いは断る。"""
        self.model_config = self.llama_config()
        self.run_training("short", self.wsd_config(), max_steps=6)
        longer = dataclasses.replace(self.model_config, context=32, rope_theta=40000.0)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                self.data,
                self.root / "long",
                longer,
                self.wsd_config(),
                torch.device("cpu"),
                continue_from=self.root / "short/last.pt",
            )
        self.assertEqual(result["step"], 8)
        model, _, saved = load_model(self.root / "long/last.pt", torch.device("cpu"))
        self.assertEqual((model.config.context, model.config.rope_theta), (32, 40000.0))
        self.assertEqual(saved["tokens_seen"], load_checkpoint(self.root / "short/last.pt")["tokens_seen"] + 2 * 4 * 2 * 32)
        with self.assertRaisesRegex(ValueError, "same model configuration"):
            train(
                self.data,
                self.root / "wider",
                dataclasses.replace(longer, intermediate_size=32),
                self.wsd_config(),
                torch.device("cpu"),
                continue_from=self.root / "short/last.pt",
            )

    def test_wsd_schedule_holds_the_peak_and_decays_over_the_last_steps(self):
        config = self.wsd_config(steps=10, decay_steps=4)
        rates = [learning_rate(config, step) for step in range(10)]
        self.assertEqual(rates[1:6], [0.01] * 5)
        self.assertTrue(all(a > b for a, b in itertools.pairwise(rates[5:])))
        self.assertAlmostEqual(rates[-1], 0.001)
        longer = self.wsd_config(steps=20, decay_steps=4)
        self.assertEqual(learning_rate(longer, 5), learning_rate(config, 5))

    def test_square_root_decay_falls_faster_at_first_and_reaches_zero(self):
        """1-sqrtの減衰は直線より先に大きく下がり、最後のステップで0にする。"""
        linear = self.wsd_config(steps=10, decay_steps=4, min_learning_rate=0.0)
        shaped = self.wsd_config(
            steps=10, decay_steps=4, min_learning_rate=0.0, decay_shape="1-sqrt"
        )
        straight = [learning_rate(linear, step) for step in range(10)]
        rates = [learning_rate(shaped, step) for step in range(10)]
        self.assertEqual(rates[:6], straight[:6])
        self.assertAlmostEqual(rates[6], 0.005)
        self.assertTrue(all(a < b for a, b in zip(rates[6:9], straight[6:9])))
        self.assertTrue(all(a > b for a, b in itertools.pairwise(rates[5:])))
        self.assertEqual(rates[-1], 0.0)

    def test_every_window_evaluation_predicts_each_held_out_token_once(self):
        """学習データなしで検証データを窓に区切り、各トークンを一度ずつ予測した損失を返す。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        context = self.model_config.context
        for path in self.data.glob("train_*.bin"):
            path.unlink()
        corpus = Corpus(self.data, ("validation",))
        result = evaluate_windows(model, corpus, torch.device("cpu"), batch_size=3)
        for language in ("ja", "en"):
            stream = corpus.streams["validation"][language]
            count = (len(stream) - 1) // context
            tokens = torch.from_numpy(
                np.asarray(stream[: count * context + 1], dtype=np.int64)
            )
            with torch.no_grad():
                expected = model(
                    tokens[:-1].view(count, context), tokens[1:].view(count, context)
                ).item()
            self.assertAlmostEqual(result[language]["loss"], expected, places=5)
            self.assertEqual(result[language]["evaluated_tokens"], count * context)
        first = evaluate_windows(
            model, corpus, torch.device("cpu"), batch_size=3, limit=1
        )
        self.assertEqual(first["ja"]["evaluated_tokens"], context)

    def test_token_bytes_count_the_text_each_token_stands_for(self):
        """各トークンが表す文章のバイト数を数え、特別なトークンは0バイトとする。"""
        data = self.root / "merged"
        prepare_data(
            self.japanese,
            self.english,
            data,
            vocabulary_size=300,
            validation_fraction=0.25,
        )
        tokenizer = Corpus(data).tokenizer
        sizes = token_bytes(tokenizer)
        text = "あいうあいう 終わり1 abc abc ending1"
        ids = tokenizer.encode(text).ids
        self.assertGreater(sizes[ids].max(), 1)
        self.assertEqual(sizes[ids].sum(), len(text.encode()))
        for special in ("<|bos|>", "<|eos|>"):
            self.assertEqual(sizes[tokenizer.token_to_id(special)], 0)

    def test_validation_reports_bits_per_byte_of_the_predicted_text(self):
        """検証で、予測したトークンのビット数を、それらが表す文章のバイト数で割って返す。"""
        torch.manual_seed(42)
        model = LanguageModel(258, self.model_config)
        corpus = Corpus(self.data, ("validation",))
        sizes = token_bytes(corpus.tokenizer)
        every = evaluate_windows(model, corpus, torch.device("cpu"), batch_size=3)
        sampled = evaluate(model, corpus, torch.device("cpu"), batches=2, batch_size=2)
        for language in ("ja", "en"):
            stream = corpus.streams["validation"][language]
            tokens = every[language]["evaluated_tokens"]
            targets = np.asarray(stream[1 : tokens + 1], dtype=np.int64)
            self.assertEqual(every[language]["evaluated_bytes"], sizes[targets].sum())
            for result in (every, sampled):
                measured = result[language]
                self.assertAlmostEqual(
                    measured["bits_per_byte"],
                    measured["loss"]
                    * measured["evaluated_tokens"]
                    / math.log(2)
                    / measured["evaluated_bytes"],
                )
        for result in (every, sampled):
            self.assertAlmostEqual(
                result["bits_per_byte"],
                (result["ja"]["bits_per_byte"] + result["en"]["bits_per_byte"]) / 2,
            )

    def test_compiled_training_saves_checkpoints_that_load_without_compilation(self):
        """torch.compileで学習しても、通常のモデルで同じ予測をするチェックポイントを保存する。"""
        result = self.run_training("compiled", self.config(steps=4), compiled=True)
        model, _, payload = load_model(
            self.root / "compiled/last.pt", torch.device("cpu")
        )
        self.assertEqual(payload["step"], 4)
        again = evaluate(
            model, self.corpus, torch.device("cpu"), batches=2, batch_size=4
        )
        for language in ("ja", "en"):
            self.assertAlmostEqual(
                again[language]["loss"],
                result["validation"][language]["loss"],
                places=4,
            )

    def test_sequential_reading_gives_languages_rows_by_their_windows_once_per_epoch(self):
        context = 16
        streams = self.corpus.streams["train"]
        # Distinct tokens make every window recognizable, and English starts at 1000.
        streams["ja"], streams["en"] = (
            np.arange(10 * context + 1),
            np.arange(7 * context + 5) + 1000,
        )
        counts = {"ja": 10, "en": 7}
        x, _ = self.corpus.sequential_batch(0, 34, context, 42, torch.device("cpu"))
        # Both languages finish their epoch at the same row.
        for epoch in (x[:17], x[17:]):
            for language, offset in (("ja", 0), ("en", 1000)):
                seen = [tuple(row.tolist()) for row in epoch if (row[0] >= 1000) == bool(offset)]
                windows = {
                    tuple(range(offset + i * context, offset + (i + 1) * context))
                    for i in range(counts[language])
                }
                self.assertEqual(len(seen), counts[language])
                self.assertEqual(set(seen), windows)
        again, _ = self.corpus.sequential_batch(3, 4, context, 42, torch.device("cpu"))
        torch.testing.assert_close(again, x[3:7], rtol=0, atol=0)
        self.assertEqual(
            [interleaved(row, {"ja": 5, "en": 5}) for row in range(4)],
            [("ja", 0), ("en", 0), ("ja", 1), ("en", 1)],
        )
        self.assertEqual(
            [interleaved(row, {"ja": 3, "en": 1})[0] for row in range(8)],
            ["ja", "ja", "ja", "en"] * 2,
        )

    def test_sequential_training_resumes_the_same_updates(self):
        self.run_training("sequential-continuous", self.wsd_config())
        self.run_training("sequential-resumed", self.wsd_config(), max_steps=3)
        self.run_training(
            "sequential-resumed",
            self.wsd_config(),
            resume=self.root / "sequential-resumed/last.pt",
        )
        first, second = (
            load_checkpoint(self.root / name / "last.pt")
            for name in ("sequential-continuous", "sequential-resumed")
        )
        for name, parameter in first["model"].items():
            torch.testing.assert_close(parameter, second["model"][name], rtol=0, atol=0)

    def test_同じデータで学習を続けると前の学習が読み終えたところから読む(self):
        """Continuing on the saved run's data reads on from where it stopped, as one run would."""
        self.run_training("continuous", self.wsd_config())
        self.run_training("first", self.wsd_config(), max_steps=3)
        self.run_training(
            "second", self.wsd_config(), continue_from=self.root / "first/last.pt"
        )
        continuous, continued = (
            load_checkpoint(self.root / name / "last.pt")
            for name in ("continuous", "second")
        )
        self.assertEqual(continued["step"], continuous["step"])
        for name, parameter in continuous["model"].items():
            torch.testing.assert_close(
                parameter, continued["model"][name], rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "keeps the seed"):
            self.run_training(
                "reordered",
                self.wsd_config(seed=7),
                continue_from=self.root / "first/last.pt",
            )

    def test_continuation_carries_the_run_onto_new_data_without_a_rate_jump(self):
        self.run_training("stable", self.wsd_config(), max_steps=6)
        stable = load_checkpoint(self.root / "stable/last.pt")
        new_data = self.root / "decay-data"
        prepare_data(
            self.japanese,
            self.english,
            new_data,
            vocabulary_size=258,
            seed=99,
            tokenizer_path=self.data / "tokenizer.json",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                new_data,
                self.root / "decay",
                self.model_config,
                self.wsd_config(),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
            )
        decayed = load_checkpoint(self.root / "decay/last.pt")
        self.assertEqual(result["step"], 8)
        self.assertEqual(decayed["phase_start_step"], 6)
        self.assertEqual(decayed["tokens_seen"], stable["tokens_seen"] + 2 * 4 * 2 * 16)
        self.assertEqual(decayed["dataset_fingerprint"], Corpus(new_data).fingerprint)
        self.assertEqual(decayed["initialization"]["continued_from"]["step"], 6)
        with contextlib.redirect_stdout(io.StringIO()):
            extended = train(
                self.data,
                self.root / "extended",
                self.model_config,
                self.wsd_config(steps=12),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
                max_steps=7,
            )
        self.assertEqual(extended["step"], 7)
        with self.assertRaisesRegex(ValueError, "keep the learning rate"):
            train(
                new_data,
                self.root / "restarted",
                self.model_config,
                self.config(),
                torch.device("cpu"),
                continue_from=self.root / "stable/last.pt",
            )

    def llama_config(self):
        return ModelConfig(
            width=24,
            heads=4,
            kv_heads=2,
            layers=2,
            context=16,
            architecture="llama",
            intermediate_size=64,
        )

    def test_層を増やしたモデルは元と同じ予測から学習を始め増やした層も学ぶ(self):
        """Grows a model to twice the layers with the same predictions, then trains every layer."""
        self.model_config = self.llama_config()
        self.run_training("small", max_steps=3)
        grown_path = self.root / "grown.pt"
        grow_checkpoint(self.root / "small/last.pt", grown_path)
        small, _, _ = load_model(self.root / "small/last.pt", torch.device("cpu"))
        grown, _, _ = load_model(grown_path, torch.device("cpu"))
        self.assertEqual(grown.config.layers, 2 * small.config.layers)
        tokens = torch.randint(
            small.vocabulary_size, (2, 16), generator=torch.Generator().manual_seed(0)
        )
        with torch.no_grad():
            torch.testing.assert_close(grown(tokens), small(tokens), rtol=0, atol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            train(
                self.data,
                self.root / "large",
                grown.config,
                self.config(steps=2),
                torch.device("cpu"),
                initialize_from=grown_path,
            )
        trained = load_checkpoint(self.root / "large/last.pt")["model"]
        for name in ("blocks.1.attention_output.weight", "blocks.3.down.weight"):
            self.assertGreater(trained[name].abs().sum().item(), 0)

    def test_muon_updates_block_matrices_and_adamw_the_embedding_and_norms(self):
        """Muon はブロック内の行列を、AdamW は埋め込みと正規化の重みを受け持ち、どの重みも一度だけ更新する。"""
        config = TrainConfig(**{**self.config().__dict__, "optimizer": "muon"})
        for model_config in (
            self.llama_config(),
            dataclasses.replace(self.llama_config(), qk_norm=True, attention_gate="elementwise"),
            dataclasses.replace(
                self.llama_config(),
                full_attention_every=2,
                experts=4,
                active_experts=2,
                prediction_depth=1,
            ),
        ):
            model = LanguageModel(258, model_config)
            muon, adamw = make_optimizer(model, config, torch.device("cpu")).optimizers
            owners = {}
            for name, optimizer in (("muon", muon), ("adamw", adamw)):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        owners.setdefault(id(parameter), []).append(name)
            for name, parameter in model.named_parameters():
                matrix = name.startswith("blocks.") and parameter.ndim == 2
                self.assertEqual(
                    owners[id(parameter)], ["muon" if matrix else "adamw"], name
                )

    def test_muon_training_learns_and_resumes_the_same_updates(self):
        """Muon で両言語の検証損失を下げ、途中保存から再開しても連続学習と同じ重みを得る。"""
        self.model_config = self.llama_config()
        config = TrainConfig(**{**self.config(steps=24).__dict__, "optimizer": "muon"})
        torch.manual_seed(42)
        before = evaluate(
            LanguageModel(258, self.model_config),
            self.corpus,
            torch.device("cpu"),
            batches=2,
            batch_size=4,
        )
        after = self.run_training("muon-continuous", config)["validation"]
        for language in ("ja", "en"):
            self.assertLess(after[language]["loss"], before[language]["loss"] * 0.8)
        self.run_training("muon-resumed", config, max_steps=10)
        self.run_training(
            "muon-resumed", config, resume=self.root / "muon-resumed/last.pt"
        )
        first, second = (
            load_checkpoint(self.root / name / "last.pt")
            for name in ("muon-continuous", "muon-resumed")
        )
        for name, parameter in first["model"].items():
            torch.testing.assert_close(parameter, second["model"][name], rtol=0, atol=0)

    def test_continuation_keeps_the_optimizer_that_saved_its_state(self):
        """継続では保存時の最適化手法を引き継ぎ、手法を変えるには重みだけを引き継ぐよう求める。"""
        self.run_training("adamw-stable", self.wsd_config(), max_steps=6)
        with self.assertRaisesRegex(ValueError, "change the optimizer"):
            self.run_training(
                "muon-decay",
                self.wsd_config(optimizer="muon"),
                continue_from=self.root / "adamw-stable/last.pt",
            )

    def test_prepared_data_reports_corruption_before_training(self):
        """改変されたトークンファイルを検出して、データの破損を通知する。"""
        path = self.data / "train_ja.bin"
        with path.open("r+b") as source:
            source.write(b"\xff\xff\xff\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Corpus(self.data)

    def test_resume_validates_the_original_dataset(self):
        """別のデータを指定した再開には、元のデータが必要なことを通知する。"""
        self.run_training("original", max_steps=2)
        other_data = self.root / "other"
        prepare_data(
            self.japanese, self.english, other_data, vocabulary_size=258, seed=99
        )
        with self.assertRaisesRegex(ValueError, "same prepared dataset"):
            train(
                other_data,
                self.root / "original",
                self.model_config,
                self.config(),
                torch.device("cpu"),
                resume=self.root / "original" / "last.pt",
            )

    def test_stop_request_saves_the_last_completed_update(self):
        """停止要求で完了済みの学習を保存し、そこから残りの学習を再開する。"""
        config = TrainConfig(**{**self.config().__dict__, "log_every": 1})
        metrics = self.root / "stopped" / "metrics.jsonl"

        def stop_requested():
            return metrics.exists() and metrics.stat().st_size > 0

        stopped = self.run_training("stopped", config, stop_requested=stop_requested)
        saved = load_checkpoint(self.root / "stopped" / "last.pt")
        self.assertEqual(stopped["step"], 1)
        self.assertEqual(saved["step"], stopped["step"])
        self.assertEqual(saved["validation_step"], stopped["step"])
        self.run_training("stopped", config, resume=self.root / "stopped" / "last.pt")
        self.run_training("uninterrupted", config)
        resumed = load_checkpoint(self.root / "stopped" / "last.pt")
        continuous = load_checkpoint(self.root / "uninterrupted" / "last.pt")
        for name, parameter in continuous["model"].items():
            torch.testing.assert_close(
                parameter, resumed["model"][name], rtol=0, atol=0
            )

    def test_explicit_gpu_training_reports_unavailable_hardware(self):
        """GPU指定時にCUDAが使えなければ、環境の確認方法を通知する。"""
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "CUDA GPU is unavailable"),
        ):
            select_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_trains_and_evaluates_on_cuda(self):
        """CUDA上で学習し、日英の有限な検証損失を返す。"""
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                self.data,
                self.root / "cuda",
                self.model_config,
                self.config(),
                torch.device("cuda"),
                max_steps=2,
            )
        for language in ("ja", "en"):
            self.assertTrue(np.isfinite(result["validation"][language]["loss"]))


if __name__ == "__main__":
    unittest.main()
