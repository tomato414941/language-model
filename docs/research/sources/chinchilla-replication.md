# Chinchilla Scaling: A replication attempt

Besiroglu ら（arXiv 2404.10102）。損失の式の当てはめの部分だけを読んだ。

Chinchilla の図から読み取った点で式を当てはめ直した。

- 当てはめ直し：L = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658
- Chinchilla の報告：L = 1.6934 + 406.4/N^0.3392 + 410.7/D^0.2849

N はパラメータ数、D はトークン数。英語のデータで、損失の式。
