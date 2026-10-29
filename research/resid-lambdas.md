# resid lambdas

v6 は各層の前で、流れに層ごとに学習する数を掛ける（5541818）。初期値と学習率は nanochat の今のコードに合わせた。

- Marin の nano_arch_ablations：1.3156 → 1.3123。
- nanochat の LOG（2026-01-11）：掛け算は層を重ねるごとに効くので、x0 lambdas の100分の1の学習率でないとうまくいかない。x0 lambdas だけのときより少し良くなった。
