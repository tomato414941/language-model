# Improving Data-Constrained Scaling with Hyperparameter Transfer and Ensembling

arXiv 2610.37842（2026-10）。要旨と導入を読んだ。

データが足りず繰り返す学習で、weight decay などをモデルの大きさとエポック数に合わせて変えるべき量を、べき乗則で当てはめた。複数のモデルの出力の平均も組み合わせる。FineWeb の 1億トークンの学習で、従来の方法より検証損失が 0.15 ナット以上下がった。小さな実行で当てはめ、12倍の計算まで確かめた。
