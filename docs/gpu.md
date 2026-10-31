# GPU

学習に借りる GPU は、業者を問わず Ada 以降にする。

FP8 が使えるのは Ada 以降だから。A100 や RTX 3090 など Ampere の GPU は使わない。

API を動かす GPU はこの決まりの外で、`deploy_modal.py` で決める。
