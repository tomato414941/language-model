# Polar Express

Muon の直交化の多項式の係数を、反復ごとに変える（arXiv 2505.16932）。nanochat は使っている。[論文](../sources/polar-express.md)では、7.74億パラメータ、10億トークンの検証損失が、Muon の元の方法の 3.398 に対して 3.340 だった。

[Polar Switch](../sources/polar-switch.md) は、同じ精度で直交化の時間を Polar Express より 15.5〜18.7% 減らすとした。v7 まで torch の Muon の直交化を使い、学習の時間のうち直交化の割合は測っていない。割合を測ってから、入れるかを決める。見直し期限：2026-12-15。
