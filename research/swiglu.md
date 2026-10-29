# SwiGLU

v6 は v5 から SwiGLU の MLP を引き継いでいて、見直していない。

nanochat の LOG（2026-02-05）では、パラメータと計算量をそろえた ReLU² と比べて、d12 と d24 のどちらでも、ステップ、時間、計算量のすべてで悪かった。Marin の nano_arch_ablations では、ReLU² にしても 1.3156 のまま変わらなかった。

v6 で替えるかは決めていない。
