# GPU

学習に借りる GPU は、業者を問わず Ada 以降にする。

FP8 が使えるのは Ada 以降だから。A100 や RTX 3090 など Ampere の GPU は使わない。

v6 は RTX 4090 で学習し、FP8 は使わない。RTX 4090 で試すと勾配が非有限になって止まった（`research/experiments/v6-throughput.md`）。

API を動かす GPU はこの決まりの外で、`deploy_modal.py` で決める。
