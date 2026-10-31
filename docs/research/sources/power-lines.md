# Power Lines

Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training（arXiv 2505.13738）。

モデルの大きさ N、データの量 D、バッチ B を変えるときに、学習率 η と weight decay λ をどう変えるかの法則を調べた。AdamW の時間スケール τ=B/(ηλD) を一定に保つという先行研究の考えを確かめたうえで、最適な τ は1パラメータあたりのトークン数のべき乗に従うとしている。

最適なバッチ Bopt と臨界のバッチ Bcrit は、どちらも D のべき乗で、モデルの大きさ N によらない。nanochat の LOG（2026-02-05）は、この指数を 0.383 として引いている。

読んだのは冒頭の主張と結論まで。指数は本文で確かめていない。
