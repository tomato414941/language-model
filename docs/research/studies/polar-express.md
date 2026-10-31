# Polar Express

The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm（arXiv 2505.16932）。実装は GitHub の NoahAmsel/PolarExpress。

Muon の中の極分解を、行列の掛け算だけで計算する多項式の方法。反復ごとの更新式を minimax 問題を解いて変える。bfloat16 で使うときの有限精度の問題にも対処している。

GPT-2 の形（7.74億パラメータ）を FineWeb の 10億トークンで学習した比較の検証損失は、どれも学習率 0.02 で次のとおり。

| 極分解の計算 | 検証損失 |
|---|---|
| muon-You | 3.399 |
| muon-Jordan | 3.398 |
| Polar Express | 3.340 |

読んだのは要旨と最初の図の説明まで。
