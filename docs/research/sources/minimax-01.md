# MiniMax-01

MiniMax-01: Scaling Foundation Models with Lightning Attention（arXiv 2501.08313）。456B（活性45.9B）の MoE に lightning attention を組み合わせ、学習で100万トークンの文脈まで扱う。読んだのは要旨とデータの繰り返しの節だけ。

全体の学習の結果と合うように、繰り返しを考慮した小さな実験の枠組みを作って調べ、質の低いデータは2回を超えると大きく悪くなり、質の高いデータは4回まで効いたとしている。
