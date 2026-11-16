# GPU

学習に借りる GPU は、業者を問わず Ada 以降にする。

FP8 が使えるのは Ada 以降だから。A100 や RTX 3090 など Ampere の GPU は使わない。

v7 は RTX 5090 で学習し、注意の query、key、value と MoE の専門家を除く行列を FP8 で掛けた。bfloat16 より22%速かった（`research/experiments/v7-throughput.md`）。v6 は RTX 4090 で、FP8 を使わずに学習した。RTX 4090 で FP8 を試すと、勾配が非有限になって止まった（`research/experiments/v6-throughput.md`）。

API を動かす GPU はこの決まりの外で、`deploy_modal.py` で決める。
