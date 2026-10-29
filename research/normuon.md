# NorMuon と cautious weight decay

v6 のブロック内の行列は、NorMuon と cautious weight decay で更新する（f994a12）。torch の Muon を継承し、手順は nanochat の nanochat/optim.py に合わせた。

- NorMuon（arXiv 2510.05491）は、直交化した更新の2乗平均をニューロン（行か列）ごとに移動平均で持ち（beta2 0.9）、その平方根で割る。そのあと、全体の大きさを直交化した直後のものに戻す。更新の大きさが Muon と同じなので、match_rms_adamw の学習率がそのまま使える。
- cautious weight decay（arXiv 2510.12402）は、更新がその重みを0に近づける向きのときだけ減衰をかける。
- weight decay の強さは、学習を通して 0.28 から cos 形で0まで下げる。nanochat と同じく行列だけにかける。

IMU-1（arXiv 2602.02522）の 7000万パラメータの比較では、AdamW の 3.250 から、NorMuon で 3.156（−2.88%）、cautious weight decay を足して 3.125（−3.85%）になった（学習損失）。

nanochat の既定の weight decay もたまたま 0.28 だが、学習率の基準が違うので、同じ値だから正しいとは言えない。cos 形で下げるときに 0.28 が良いかは確かめていない。

nanochat はほかに Polar Express、MuonEq、Muon+ も使っている。MuonEq と Muon+ は nanochat 自身が「(very) slight and optional」としているので、v6 では入れていない。
