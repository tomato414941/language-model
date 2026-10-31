# FP8 で学習する

行列の計算を、8ビットの浮動小数点で行う。compute capability 8.9 以上の GPU で動き、Ada（RTX 4090 など）から使える。A100 と RTX 3090 の Ampere では動かない。NVIDIA の NeMo AutoModel の FP8 の入口も、この 8.9 を条件にしている。

- nanochat の LOG（2026-02-02）：torchao の tensorwise で、d26 の速さが1.17倍、メモリが約9GB減った。同じ損失で比べると、速くなるのは約5%。rowwise は0.90倍で遅くなる。torch.compile がないと4倍遅い。辺が16の倍数でない行列は変えられず、d12 ではかえって遅い。
- modded-nanogpt の[記録 #92](studies/modded-nanogpt-anvil2.md)：FP8 の拡張を外すと、39.9秒のうち時間が5.02秒増え、損失は 0.4 ミリナット下がる。独自のカーネルを使っていて、torchao ではない。
- どちらも H100 での結果で、Ada で測った結果は見つけていない。
