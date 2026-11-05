# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

arXiv 2508.18672。要旨を読んだ。

計算をそろえて、全体と使う大きさと選ぶ数を変えた MoE を比べた。記憶は全体のパラメータとともに上がる。推論は、同じ損失でも使う計算が多いほど良く、1パラメータあたりのトークンに最適な値がある。GRPO や推論の計算を増やしても傾向は変わらない。
