# MUDDFormer

MUDD（Multiway Dynamic Dense Connections）。Xiao ら（arXiv 2502.12170）。

残差の接続の代わりに、それまでのすべての層の出力を混ぜて次の層の入力を作る。混ぜる重みは、位置ごとに隠れ状態から動的に作り、query、key、value、残差の入力ごとに分ける。

Transformer を1.8〜2.4倍の計算量で学習したときと同じ性能になるとしている。MUDDPythia-2.8B は、パラメータを0.23%足すだけで、Pythia-6.9B の perplexity と下流の課題に並ぶ。

読んだのは要旨、導入、結論まで。
