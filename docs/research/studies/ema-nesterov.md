# EMA-Nesterov

EMA-Nesterov: Stabilizing Nesterov's Lookahead for Accelerated Deep Learning Optimization（arXiv 2605.25395、2026-05）。

どのオプティマイザにもかぶせられる包み。パラメータの更新量の EMA を持ち、その方向に先読みした位置で勾配を計算し、元のオプティマイザの更新をその位置から行う。一歩ごとの差分ではなく EMA を使うので、先読みがノイズで不安定になりにくいとしている。

- modded-nanogpt の最適化の部門（162M）で、調整済みの Muon が 3.28 に届くまでのステップを6%減らした。部門の表では #12 の 3325 が #39 の 3125 になっている（n=20）。
- 124M では、Adam、SOAP、Muon、NorMuon のどれでも perplexity が下がり、下がり幅は Muon の 0.208 から Adam の 1.305 まで。
- Llama の形で 60M、130M、350M を調べ、大きいモデルほど最適な γ は上がり、最大の β は下がった。既定の値は β=0.5、γ は 350M 以下で 0.99、それより大きければ 0.995。
- 1B（Muon、5000反復）で perplexity が 12.22 → 12.12。
- β は学習率に沿わせ、初めの30%は0、最後の20%も切る。Llama の実験は C4、文脈256。

比べた先読みの手法（Lookahead、SNOO、GPA）の中では、Muon の上で一番改善が大きかった。
