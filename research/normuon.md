# NorMuon と cautious weight decay

v6 の行列は NorMuon（arXiv 2510.05491）で更新し、cautious weight decay（arXiv 2510.12402）を 0.28 から cos 形で0まで下げる。nanochat の実装に合わせた（f994a12）。

- IMU-1（arXiv 2602.02522、7000万パラメータ）：NorMuon で学習損失 −2.88%、cautious weight decay を足して −3.85%。
- 0.28 が cos 形の下げ方で最適かは確かめていない。
