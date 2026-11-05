# Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization

arXiv 2502.19261（ICLR 2025）。要旨を読んだ。

学習済みの密なモデルを複製して MoE にするとき、重みの一部を元の分布で初期化し直して、専門家が分かれやすくする。そのまま複製する方法は、初めは速いが長い学習では一から学習した MoE に負ける。この方法は数千億トークン以上で上回り、5.9B を使う MoE が、同じ系列の密な 13B に約1/4の学習の計算で並んだ。コード、データ、重みを公開している。
