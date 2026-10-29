# Bigram hash embedding

直前のトークンとの組をハッシュして表を引き、流れに足す。

- Marin speedrun の nano_arch_ablations（約1.5億〜2.8億パラメータ、Paloma C4 の BPB。前の行に1つずつ足していく比較）: −0.0030（約2.8億パラメータ）。
- nanochat の dev/LOG.md（commit 92d63d4e） 2026-01-28: d25 では改善がごく小さく、時間で比べると消えた。メモリも増えるので取り除いた。

2026-10-29 不採用。
