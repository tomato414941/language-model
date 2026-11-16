# Bigram hash embedding

直前のトークンとの組をハッシュして表を引き、流れに足す。Marin の [nano_arch_ablations](../sources/marin-nano-arch-ablations.md) では 1.2787 → 1.2757。nanochat は d25 で改善がごく小さく、時間で比べると消え、メモリも増えたとして取り除いた（LOG 2026-01-28）。

[低ビットの研究](../sources/low-bit-ngram.md)では、学習する n-gram の表を4ビットにするとメモリが約75%減り、効果の99%以上が残った。表を学習しない [COUNT](count-ngram-embedding.md) のほうが GPU のメモリを使わない。v7 まで n-gram の表を使っていないので当たらない。学習する n-gram の表を入れる世代で読み直す。見直し期限：2026-12-15。
