# Accelerate Scaling of LLM Finetuning via Quantifying the Coverage and Depth of Instruction Set

arXiv 2509.06463 v2。SFT のデータの量を増やしても点が比例して上がるとは限らないとして、意味の広さ（coverage）と1件の情報の深さ（depth）で説明する。要旨と実験の節を読んだ。

- 200万件の指示から 1万、2万、10万、20万、50万件を選び、Qwen2-7B と Llama3-8B を学習して AlpacaEval 2.0 と ArenaHard で比べた。
- ランダムに選ぶと、50万件まで点は上がり続けた。増えたのは広さで、深さはほとんど変わらなかった。
- 広さと深さの2つで、開発データの損失の差の7割以上を説明できた。
- 複雑さの指標で選ぶ既存の方法（Deita）は、10万件を超えると効果が落ちた。
