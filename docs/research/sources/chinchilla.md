# Chinchilla

Training Compute-Optimal Large Language Models（arXiv 2203.15556）。Hoffmann ら。

7000万〜160億パラメータの400以上のモデルを、50億〜5000億トークンで学習して、計算量が決まっているときのパラメータ数とトークン数の配分を調べた。パラメータ数とトークン数は同じ割合で増やすのが最適で、当時の大きなモデルは学習が足りないとしている。

確かめに、Gopher（280B）と同じ計算量で、70B パラメータと4倍のデータで学習した Chinchilla が、Gopher、GPT-3、Jurassic-1、Megatron-Turing NLG を広い課題で上回った。MMLU の平均は 67.5% で、Gopher より7%以上良い。

読んだのは要旨、導入、結論まで。
