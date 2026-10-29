# LayerNorm scaling

各層の正規化した入力に 1/√(層の番号+1) を掛ける。IMU-1 では学習損失 −0.36%、nanochat の LOG（2026-03-24）では「Did not help」。
