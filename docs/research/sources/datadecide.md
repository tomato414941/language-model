# DataDecide

How to Predict Best Pretraining Data with Small Experiments（arXiv 2504.11393）。要旨を読んだ。

25のコーパスを、最大 100B トークン、最大 1B のモデル、3つのシードで比べた。150M のような小さな1つの大きさでの順位が、1B での最良を予測する良い基準で、比較の約80%が当たった。8つのスケーリング則の方法は、これを超えなかった。小さな実験で連続の尤度の指標を使うと、MMLU、ARC、HellaSwag、MBPP、HumanEval は、0.01%の計算で 1B での結果を80%以上予測できた。
