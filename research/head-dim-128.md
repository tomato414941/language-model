# ヘッド幅128

v6 はヘッド幅を64から128にした。query 8、key-value 2 で、幅と key-value の幅は v5 と同じ（326f637）。

Marin の nano_arch_ablations（約1.5億〜2.8億パラメータ、Paloma C4 の BPB）では、12ヘッドを6ヘッドにして 1.2962 → 1.2838 と、並んだ変更の中で一番大きく下がった。nanochat の LOG（2026-01-17）にも「Head dim definitely prefers to be 128 instead of 64, i.e. fewer bigger heads」とある。
