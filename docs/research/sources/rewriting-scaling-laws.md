# Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints

arXiv 2610.31867（2026-10）。要旨と、計算量で制限する実験の節を読んだ。データ量で制限する実験は読んでいない。

FineWeb-Edu に FinePhrase（SmolLM2-1.7B の書き直し）を混ぜる割合 R を 0 から 1 まで 0.1 刻みで変え、3×10¹⁸〜9×10¹⁹ FLOPs のモデルの列でスケーリング則を当てはめた。FLOPs は 6ND で数え、書き直しを作る費用も1トークンあたり 4Nr として足す。

- 最適な R は計算量とともに上がる。Paloma では 3×10¹⁸ FLOPs で10%未満、10²² FLOPs で32%。
- 評価で違う。コードと数学は計算量の範囲内で R=1 まで上がる。QA と Paloma は漸近値が 0.77 と 0.32。
- 10²² FLOPs で、書き直しだけ（R=1）は自然なデータだけ（R=0）に比べて、数学の bits per byte が 1.33 から 0.80、コードが 1.04 から 0.53 に下がる。
- 10²² FLOPs の R=0.32 は、Paloma の16のうち10で書き手より低い bits per byte になった。

モデルの列の学習データは FineWeb-Edu で、日本語のデータはない。
