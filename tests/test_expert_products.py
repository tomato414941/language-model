import importlib.util
import os
import unittest

import torch

# Triton runs the kernels on the CPU through its interpreter, which it picks
# when the module that defines them is imported.
os.environ["TRITON_INTERPRET"] = "1"
TRITON = importlib.util.find_spec("triton") is not None
if TRITON:
    from pretrain.expert_products import expert_outer_products, expert_products
from pretrain.model import grouped, grouped_outer


@unittest.skipUnless(TRITON, "Triton is not installed")
class ExpertProductsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # The first expert takes no row; the others' rows end inside the
        # kernels' blocks of 16 and span several of them.
        self.ends = torch.tensor([0, 21, 26, 44], dtype=torch.int32)
        self.rows = torch.randint(0, 13, (44,))
        self.scores = torch.rand(44)

    def test_専門家の積は全専門家を一度に掛けて一つずつ掛けたときと同じ値を返す(self):
        """専門家の積は、全専門家の行を一度に掛け、単語の行を集める形、隠れ層の値に変えて点数を掛ける形、
        転置した行列を読む形のどれでも、一つずつ掛けたときと同じ値を返す。"""
        tokens, raised, gated = torch.randn(13, 40), torch.randn(44, 40), torch.randn(44, 40)
        matrices = torch.randn(4, 40, 24)
        cases = {
            "sorted rows": (raised, matrices, {}),
            "gathered rows": (tokens, matrices, {"rows": self.rows}),
            "squared ReLU": (raised, matrices, {"scores": self.scores}),
            "SwiGLU": (raised, matrices, {"gates": gated, "scores": self.scores}),
            "transposed matrices": (raised, torch.randn(4, 24, 40).transpose(-2, -1), {}),
        }
        for name, (inputs, factors, options) in cases.items():
            with self.subTest(name):
                torch.testing.assert_close(
                    expert_products(inputs, factors, self.ends, **options),
                    grouped(inputs, factors, self.ends, **options),
                )

    def test_専門家の外積は各専門家の行の転置と勾配の行の積を足し合わせる(self):
        """専門家の外積は、各専門家の行の転置と勾配の行の積を足し合わせ、行のない専門家には0を返す。
        入力や勾配の行を集める形、入力を隠れ層の値に変えて点数を掛ける形でも、一つずつ掛けたときと同じ値を返す。"""
        tokens, raised, gated = torch.randn(13, 40), torch.randn(44, 40), torch.randn(44, 40)
        gradients, received = torch.randn(44, 24), torch.randn(13, 24)
        cases = {
            "sorted rows": (raised, gradients, {}),
            "gathered inputs": (tokens, gradients, {"input_rows": self.rows}),
            "gathered gradients": (raised, received, {"gradient_rows": self.rows}),
            "squared ReLU": (raised, received, {"gradient_rows": self.rows, "scores": self.scores}),
            "SwiGLU": (raised, gradients, {"gates": gated, "scores": self.scores}),
        }
        for name, (inputs, factors, options) in cases.items():
            with self.subTest(name):
                got = expert_outer_products(inputs, factors, self.ends, **options)
                torch.testing.assert_close(got, grouped_outer(inputs, factors, self.ends, **options))
                self.assertEqual(got[0].abs().max().item(), 0.0)

    def test_専門家の積はコンパイルした計算の中でも同じ値と形を返す(self):
        """専門家の積と外積は、コンパイルした計算の中でも、そのまま呼んだときと同じ値と形を返す。"""
        tokens, matrices, gradients = torch.randn(13, 40), torch.randn(4, 40, 24), torch.randn(13, 24)

        def products(tokens, matrices, gradients):
            raised = expert_products(tokens, matrices, self.ends, self.rows)
            return raised, expert_outer_products(tokens, gradients, self.ends, self.rows, self.rows)

        compiled = torch.compile(products, backend="aot_eager", fullgraph=True)
        for got, expected in zip(compiled(tokens, matrices, gradients), products(tokens, matrices, gradients)):
            torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
