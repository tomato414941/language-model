# GPU

学習に借りる GPU は、業者を問わず Ampere 以降にする。

V100 のような古い世代では、PyTorch の CUDA 13 版が動かず、v5 の環境も動かなかった。

API を動かす GPU はこの決まりの外で、`deploy_modal.py` で決める。
