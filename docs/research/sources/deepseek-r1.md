# DeepSeek-R1

DeepSeek-R1（arXiv 2501.12948）。評価の設定と、使い方の注意の部分を読んだ。

V3 の設定に従い、MMLU、DROP、GPQA Diamond、SimpleQA は simple-evals の問いで評価し、MMLU-Redux は Zero-Eval の形で例なしで評価した。R1 は問いの形に敏感で、例を付けると成績が下がるので、問題をそのまま書いて出力の形を指定するよう勧めている。
