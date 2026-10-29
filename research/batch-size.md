# バッチの大きさ

v6 は1回の更新を 1,048,576 トークンにした（v5 の4倍、8d55a8b）。

- nanochat LOG 2026-02-05：Power Lines（arXiv 2505.13738）の Bopt ∝ D^0.383 から、34〜96億トークンを読むモデルは 2^20。v6 は84億。
- 学習率は nanochat に従ってバッチの平方根で換算し、v5 の2倍。nanochat 自身、Muon に同じ換算を使うのは「not studied carefully, assumption」と書いている。
- weight decay は nanochat の式 λ·√(B/B_ref)·(D_ref/D) で 0.1 × 2 × 117.4億/83.9億 ≈ 0.28。
