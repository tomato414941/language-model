# SOAP-Muon

Muon の直交化の前に、行列の小さいほうの辺に SOAP の前処理をかける（[Vyas ら](studies/soap-muon.md)）。

[modded-nanogpt の最適化の部門](studies/modded-nanogpt-optimization-track.md)では、MLP だけに入れた #14 が 3150 ステップ、すべての隠れ層の行列に毎ステップ入れた #44 が 2750 ステップ（n=20）で、3.28 に届いた。この部門は時間を問わないので、実時間で得かどうかは別に測る必要がある。
