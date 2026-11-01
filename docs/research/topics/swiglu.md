# SwiGLU

MLP の形の1つ。ゲートの行列の出力を SiLU に通して、もう1つの行列の出力と掛け合わせ、3つめの行列で戻す。

[nanochat の LOG](../sources/nanochat-log.md)（2026-02-05）では、パラメータと計算量をそろえた ReLU² と比べて、d12 と d24 のどちらでも、ステップ、時間、計算量のすべてで悪かった。Marin の [nano_arch_ablations](../sources/marin-nano-arch-ablations.md) では、ReLU² にしても 1.3156 のまま変わらなかった。

[Closed Gates Leak](../sources/closed-gates-leak.md) は、SiLU や GELU のゲートは閉じても少し通すので、隣の層の打ち消しが起きるとした。性能の良し悪しの結果ではなく、ReLU² を使う v6 には当たらない。ゲート付きの MLP を使う世代で読み直す。見直し期限：2026-11-15。
