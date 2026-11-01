# MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control

arXiv 2610.37647（2026-10）。要旨、はじめに、実験の節、付録の比較を読んだ。証明は読んでいない。

Muon の更新から、重みに平行な成分を Gram–Schmidt で取り除く。Hyperball と違って重みのノルムは固定しない。一定の学習率なら、ノルムの増え方は √t 以下になる。

- FineWeb、modded-nanogpt の形の 160M と 400M、1×〜8× の Chinchilla で MuonH と比べた。長い学習では検証の perplexity が MuonH より低く、トークン換算で最大7〜15%速い。
- 160M と 400M、20k ステップ（約10.5B トークン）の比較では、良い順に MuonGS、MuonH、weight decay の Muon で、何もしない Muon は明らかに悪い。
- MoE（1B、活性 135M）でも MuonH より低かった。予備的な結果としている。

MuonH の損失をトークン数の関数として当てはめて換算している。NorMuon とは比べていない。
