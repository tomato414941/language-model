# CoLoR-Filter

Brandfonbrener ら、CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training（arXiv 2406.10670）。要旨と結論を読んだ。

2つの補助モデルの損失の差で文書を選ぶ。C4 から Books 向けに選ぶ課題と、選択式の課題向けに選ぶ課題で評価し、1.2B のモデルまで良い傾向が続き、64個に1個まで絞っても改善が続いたとしている。データの多様さを考えていないことが妨げになるかは、未解決としている。
