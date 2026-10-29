# Cautious weight decay と weight decay の cos 形の下げ方

更新と重みの符号が同じ（更新がその重みを0に近づける）ときだけ減衰をかける（arXiv 2510.12402）。強さは 0.28 から cos 形で0まで下げ、ブロック内の行列だけに使う（nanochat と同じ）。

- IMU-1: NorMuon に足して、学習損失がさらに約1ポイント下がった（−2.88% → −3.85%）。
- nanochat の dev/LOG.md（commit 92d63d4e） の base_train.py: cos 形で0まで下げる形を使っている。

2026-10-29 採用（f994a12）。0.28 が cos 形の下げ方で最適かは確かめていない。偶然 nanochat の既定値も 0.28。
