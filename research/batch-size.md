# バッチの大きさ（約100万トークン）と学習率・weight decay の換算

1回の更新を 262,144 トークンから 1,048,576 トークンにする。学習率はバッチの平方根に比例させて2倍（0.0032）、weight decay は 0.1 × 2 × 117.4億/83.9億 ≈ 0.28。

- nanochat の dev/LOG.md（commit 92d63d4e） 2026-02-05: Power Lines（arXiv 2505.13738）の Bopt ∝ D^0.383 で、34〜96億トークンを読むモデルは 2^20。
- nanochat の base_train.py: 学習率を √(B/B_ref) で換算。Muon にも同じ換算を使うのは「not studied carefully, assumption」と書かれている。weight decay は λ = λ_ref·√(B/B_ref)·(D_ref/D)。

2026-10-29 採用（8d55a8b）。
