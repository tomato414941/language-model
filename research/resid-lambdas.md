# resid lambdas

各層の前で、流れに層ごとに学習する数を掛ける。

- Marin の nano_arch_ablations：1.3156 → 1.3123。
- nanochat の LOG（2026-01-11）：掛け算は層を重ねるごとに効くので、x0 lambdas の100分の1の学習率でないとうまくいかない。x0 lambdas だけのときより少し良くなった。
