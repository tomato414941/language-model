# LAWA

Early Weight Averaging meets High Learning Rates for LLM Pre-training（arXiv 2306.03241）。実装は GitHub の sanyalsunny111/Early_Weight_Avg。

学習の途中の重みを平均する。学習率が高いほど得が大きく、平均する重みの間隔が広いほど得が大きいとしている。

OpenWebText（90億トークン）、7万ステップの検証損失。

| モデル | 元の学習 | EMA | SWA | LAWA |
|---|---|---|---|---|
| 125M | 2.963 | 2.949 | 2.952 | 2.917 |
| 355M | 2.855 | 2.845 | 2.837 | 2.819 |

770M でも同じ順で LAWA が一番低い。Pythia の 1B〜12B（207B トークン）でも調べている。

読んだのは要旨、結論、最初の図の説明まで。要旨では中くらいのモデルを 335M、本文では 355M と書いている。
