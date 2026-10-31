# ヘッド幅128

注意のヘッド1つあたりの幅を、64ではなく128にする。

Marin の nano_arch_ablations（約1.5億〜2.8億パラメータ、Paloma C4 の BPB）では、12ヘッドを6ヘッドにして 1.2962 → 1.2838 と、並んだ変更の中で一番大きく下がった。nanochat の LOG（2026-01-17）にも「Head dim definitely prefers to be 128 instead of 64, i.e. fewer bigger heads」とある。
