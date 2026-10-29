# LayerNorm scaling

各層の正規化した入力に 1/√(層の番号+1) を掛ける。

- IMU-1（7000万パラメータ）: 学習損失 −0.36%。
- nanochat の dev/LOG.md（commit 92d63d4e）: 「Did not help」。

2026-10-29 不採用（結果が割れている）。

見直すとき：大きな規模での比較が出たとき。
