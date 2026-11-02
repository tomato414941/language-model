# パラメータ数と学習トークン数の配分

決まった計算量を、パラメータ数と学習トークン数にどう分けるか。[Chinchilla](../sources/chinchilla.md) の目安は、1パラメータ約20トークン。

[nanochat の LOG](../sources/nanochat-log.md)（2026-01-17）は、value embeddings でパラメータが増えたので、最適なトークン数とパラメータの比が8から4に下がったとしている（Chinchilla の20よりずっと低い）。

[Kaplan ら](../sources/scaling-laws.md)は、形の影響はとても小さいとしている。

[Compute Optimal Tokenization](../sources/compute-optimal-tokenization.md) は、20 はトークナイザの圧縮率に依存し、安定するのはバイトだとした。英語で1パラメータ約60バイト、6言語を一緒に学習して約70バイト。BPE の当てはめの係数からは約45バイト（こちらの計算）。

[Chinchilla の当てはめ直し](../sources/chinchilla-replication.md)と [ATLAS](../sources/atlas.md) の英語の式では、配分より計算の総量が損失に効く。パラメータ3.86億、1パラメータ11トークンを起点に、読む量を22トークンにすると −.11〜−.14、計算量を固定して大きさを2.4億にすると −.01〜−.05（こちらの計算）。

言語を増やしても、計算をそろえれば最適な比はほとんど変わらない（[複数の言語で学習する](multilingual-training.md)）。
