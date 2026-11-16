# Partial RoPE

ヘッドの次元の一部だけに、回転の位置符号をかける。[Marin](../sources/marin-nano-arch-ablations.md) では半分にかけて 1.2838 → 1.2827、[nanochat の LOG](../sources/nanochat-log.md)（2026-03-24）では4分の1にかけて「Slightly worse」。

[層ごとの割り当ての研究](../sources/layerwise-rope.md)は、短い実行で層ごとの効きを測って半分の層に割り当てると、1B で検証損失が 2.785 から 2.758 に下がるとした（350M では 3.027 から 3.024）。一様に半分にすると悪くなる。下流の点は読んだ範囲にない。v7 まですべての層に使う。形を見直す世代で検討する。見直し期限：2026-12-15。
