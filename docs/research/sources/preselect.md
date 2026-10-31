# PreSelect

Shum ら、Predictive Data Selection: The Data That Predicts Is the Data That Teaches（arXiv 2503.00808）。要旨と結論を読んだ。

モデルの損失が下流の能力をよく予測する文書を、学習にも効く文書だと仮定し、fastText の判定器で選ぶ。1B と 3B のモデルで、選んだ 30B トークンの学習が、何も選ばない 300B トークンの学習を上回った。3B を 100B トークン学習した比較では、DCLM と FineWeb-Edu より良かったとしている。評価は著者ら自身のもの。
