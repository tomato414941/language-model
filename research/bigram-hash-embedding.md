# Bigram hash embedding

直前のトークンとの組をハッシュして表を引き、流れに足す。Marin の nano_arch_ablations では 1.2787 → 1.2757。nanochat は d25 で改善がごく小さく、時間で比べると消え、メモリも増えたとして取り除いた（LOG 2026-01-28）。
