# Muon+

Muon+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training（arXiv 2602.21545）。

実際に使う回数の少ない直交化では、行や列の大きさの偏りがかえって大きくなることがあり、それが安定して使える学習率を下げるとしている。直交化のあとに正規化を1回足すだけで直す。状態は増えず、1ステップの時間もメモリも Muon とほぼ同じ。

GPT と LLaMA の 60M〜7B で、1パラメータ約200トークンまでの範囲で、学習と検証の perplexity が Muon より良かったとしている。LLaMA 350M では、学習率を上げても Muon より安定していた。

読んだのは要旨、主な主張、結論まで。表の数字は確かめていない。
