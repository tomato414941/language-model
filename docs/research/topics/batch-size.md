# バッチの大きさ

[nanochat の LOG](../sources/nanochat-log.md)（2026-02-05）は、[Power Lines](../sources/power-lines.md)（arXiv 2505.13738）の「最適なバッチは学習トークン数の 0.383 乗に比例する」を使い、d12 で測った 2^19 から外挿している。その表では、34億〜96億トークンを読むモデルが 2^20 になる。

バッチを変えたときの換算も、nanochat の scripts/base_train.py にある。

- 学習率はバッチの平方根に比例させる。コメントには、Muon にこれを使うのは「not studied carefully, assumption!」とある。
- weight decay は λ·√(B/B_ref)·(D_ref/D) で換算する。学習を通して重みが忘れていく速さを保つための式。

[最適な手法とバッチの研究](../sources/best-optimizer-batch-size.md)は、最適化の部門の設定（1.7B トークン）で、128K から移すときの最良の規則が 512K までは学習率の平方根で、1M と 2M では学習率の据え置きと weight decay の比例になるとした。v6 は v5 の4倍の 1M に平方根の規則で移した。NorMuon では比べられていない。
