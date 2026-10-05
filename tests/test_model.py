import math
import tempfile
import unittest
from pathlib import Path

from model import (
    GPT,
    Config,
    Scalar,
    Tokenizer,
    evaluate,
    load_model,
    save_model,
    train_model,
)


class LanguageModelTests(unittest.TestCase):
    def test_tokenizer_round_trips_japanese(self):
        """日本語を文字単位で符号化し、境界トークンを除いて復元する。"""
        tokenizer = Tokenizer("あいうえお")
        self.assertEqual(
            tokenizer.decode([tokenizer.boundary, *tokenizer.encode("あおい")]),
            "あおい",
        )
        with self.assertRaisesRegex(ValueError, "outside the training vocabulary"):
            tokenizer.encode("か")

    def test_backward_accumulates_derivatives_along_shared_paths(self):
        """同じ値を複数回使った式の勾配を合計する。"""
        x, y = Scalar(2), Scalar(3)
        result = ((x * y + x).exp()).log() + x**2
        result.backward()
        self.assertAlmostEqual(result.value, 12)
        self.assertAlmostEqual(x.grad, 8)
        self.assertAlmostEqual(y.grad, 2)

    def test_attention_gradient_matches_finite_difference(self):
        """文字列の損失から求めた勾配が、重みを微小変更した損失の差に一致する。"""
        model = GPT(Tokenizer("ab"), Config(width=4, heads=2, context=4))
        model.loss("aba").backward()
        for name, row, column in [
            ("0.query", 1, 2),
            ("token", 0, 1),
            ("0.expand", 3, 2),
        ]:
            parameter = model.weights[name][row][column]
            expected = parameter.grad
            original, epsilon = parameter.value, 1e-5
            parameter.value = original + epsilon
            upper = model.loss("aba").value
            parameter.value = original - epsilon
            lower = model.loss("aba").value
            parameter.value = original
            self.assertAlmostEqual(expected, (upper - lower) / (2 * epsilon), places=5)

    def test_training_learns_a_repeated_pattern(self):
        """学習によって文字パターンの予測誤差を下げる。"""
        model = GPT(Tokenizer("ab"), Config(width=4, heads=2, context=5))
        documents = ["abab", "abab", "abab"]
        before = evaluate(model, documents)
        losses = train_model(model, documents, steps=60, learning_rate=0.03)
        after = evaluate(model, documents)
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertLess(after, before * 0.5)
        self.assertEqual(model.generate(temperature=0), "abab")

    def test_checkpoint_preserves_predictions_and_seeded_samples(self):
        """保存して読み込んだモデルが同じ予測と乱数シードに対する生成を行う。"""
        model = GPT(Tokenizer("abc"), Config(width=4, heads=2, layers=2, context=5))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.json"
            save_model(model, checkpoint)
            restored = load_model(checkpoint)
        self.assertEqual(restored.loss("abc").value, model.loss("abc").value)
        self.assertEqual(restored.generate("a", seed=9), model.generate("a", seed=9))

    def test_generation_keeps_prefix_and_obeys_limits(self):
        """接頭辞を維持して、指定した文字数と文脈長の範囲で生成する。"""
        model = GPT(Tokenizer("abc"), Config(width=4, heads=2, context=5))
        output = model.generate("ab", max_new_tokens=2)
        self.assertTrue(output.startswith("ab"))
        self.assertLessEqual(len(output), 4)
        self.assertLessEqual(len(model.generate("abc", max_new_tokens=20)), 5)
        with self.assertRaisesRegex(ValueError, "prefix must be shorter"):
            model.generate("abcab")


if __name__ == "__main__":
    unittest.main()
