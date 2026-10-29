# 注意の出力のヘッドごとのゲート

注意の出力に、ヘッドごとに学習するゲートを掛ける。

IMU-1（arXiv 2602.02522、7000万パラメータ）では、単独で学習損失 −0.05% だった。ただし、QK-norm、value residual、LayerNorm scaling と合わせると −1.64% で、それぞれの和より大きかった。Marin の nano_arch_ablations では 1.2827 → 1.2787。
