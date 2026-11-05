# 学習の数値の精度

v6 は重みを fp32 で持ち、計算を `torch.autocast` の bf16 で行う（`pretrain/training.py`）。

- [レバレッジの研究](../sources/weight-update-leverage.md)：bf16 の重みを直接更新すると、更新が丸めで消えて36〜99%の成分が動かなくなる。fp32 の重みを持つ v6 には当たらない。bf16 の重みを直接更新する世代では当たる。見直し期限：2026-11-15。
- [損失の発散の研究](../sources/quantization-loss-spikes.md)：頭のロジットが大きくなると、学習の後半に丸めの誤差で発散することがある。頭の最適化をモメンタムつき SGD にすると防げる。再現の条件は 404M を 96B トークン、1更新 131,072 トークン、weight decay なしで、v6 の条件（8.4B トークン、1,048,576 トークン、weight decay あり）とは離れている。読む量を大きく増やすか weight decay を外す世代、または v6 で損失が戻らなくなったときに読み直す。見直し期限：2026-11-15。

FP4：[FP4 All the Way](../sources/fp4-all-the-way.md) は NVFP4 で 7B・200B トークンを bf16 並みに学習した。勾配が量子化の雑音に埋もれる終盤は、精度を上げる必要がある。[Quartet](../sources/quartet.md) は、RTX 5090 で線形層の計算が FP8 の約2倍速いとした。5090 で FP4 の学習を動かす道具は PyTorch や torchao にまだない（[5090 の行列計算の記事](../sources/rtx-5090-matmul.md)）。v7 の FP8 の測定では行列の計算が GPU の時間の約58%で、FP4 にしても全体の短縮は1〜2割にとどまる見込み（概算）。v7 では使わない。
