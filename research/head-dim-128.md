# ヘッド幅128

注意のヘッド1つあたりの幅を64から128にする。v6 は query 8、key-value 2 で、幅と key-value の幅は v5 と同じ。

- Marin speedrun の nano_arch_ablations（約1.5億〜2.8億パラメータ、Paloma C4 の BPB。前の行に1つずつ足していく比較）: head_dim=128 で 1.2962 → 1.2838（−0.0124）。比較の中で最大の改善。
- nanochat の dev/LOG.md（commit 92d63d4e） 2026-01-17: 「Head dim definitely prefers to be 128」。

2026-10-29 採用（326f637）。
