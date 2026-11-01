# Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization

arXiv 2606.16899（2026-06）。方法と実験の節（2、3）を読んだ。理論の節は読んでいない。

注意と MLP の重み行列について、更新の大きさと重みの大きさ（Frobenius ノルム）を固定する。初期値のノルムを半径 R とし、`W ← R · normalize(W − η R · normalize(u))` とする。埋め込み、正規化のゲイン、スカラーは対象にしない。計算は行列ごとに O(N²)。

- 1.2B の Qwen3 形、1×〜8× の Chinchilla で、MuonH は AdamW のスケーリング則に対してトークン換算で20〜30%速く、weight decay の Muon は約10%だった。差は学習を長くするほど広がる。学習の初めは少し悪く、学習率が下がるにつれて追い越す。
- Marin の speedrun（FineWeb-Edu、1×）では、AdamH と MuonH は約10%大きい weight decay の基準と同じ損失だった。
- modded-nanogpt の Track 3 では、差がない組み合わせもある。Muon（3325 ステップ）と MuonH（3325）、NorMuon（3250）と NorMuonH（3250）。
- 深さ 4〜512 と幅 128〜2048 で、最適な学習率の振れが約1.4倍で済んだ。AdamW は約3倍、Muon は約4倍。
- 130M を 1B〜128B トークンで学習した範囲で、MuonH の最良の損失は全域で MuonW より低かった。
