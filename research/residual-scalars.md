# x0 / resid lambdas

各層の前で `x = resid[i]·x + x0[i]·x0` とする。初期値は nanochat と同じ（resid は 1.15 → 1.05、x0 は 0.20 → 0.05）。学習率は nanochat の 0.005 と 0.5 をバッチの平方根で換算し、x0 の beta1 は 0.96。

- Marin speedrun の nano_arch_ablations（約1.5億〜2.8億パラメータ、Paloma C4 の BPB。前の行に1つずつ足していく比較）: resid lambdas で −0.0033、x0 で −0.0011。
- nanochat の dev/LOG.md（commit 92d63d4e） 2026-01-11: x0 はどの深さでも BPB が下がった（d8 で −0.0103 〜 d20 で −0.0039）。resid は学習率を100分の1にしないといけない。
- nanochat の dev/LOG.md（commit 92d63d4e） 2026-01-19〜22: x0 の beta1 を 0.96 にすると、d20 で −0.0007。

2026-10-29 採用（5541818）。
