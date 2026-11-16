# 重みの大きさの制御（Hyperball、MuonGS）

v6 は NorMuon に、更新が0へ動かしている重みだけを減衰する [cautious weight decay](cautious-weight-decay.md) を使い、重みの大きさそのものは固定しなかった。v7 は NorMuon の行列を Hyperball で初めのノルムに保ち、weight decay を外した（`configs/model-v7.toml`）。

- [Hyperball](../sources/hyperball.md)：重みのノルムと更新のノルムを固定する。1.2B、1×〜8× で Muon の weight decay より速く、学習を長くするほど差が広がる。学習率の振れも小さい。
- [Puro-2B](../sources/puro-2b.md)：パラメータあたり20トークンのモデルの列で、Muon より同じ損失に要る計算が16.1%少ない。
- [MuonGS](../sources/muongs.md)：更新から重みに平行な成分を除くだけで、ノルムは固定しない。160M と 400M の長い学習で MuonH より最大7〜15%速い。
- modded-nanogpt の Track 3（約 160M を約 1.6B トークン、パラメータあたり約10トークン）では、MuonH と NorMuonH は元の手法と同じステップ数だった。
- Track 3 の記録 #29 以降の radial brake は、外へ向かう成分だけを半分にする近い手法。

v7 は1トークンが使うパラメータあたり18トークンで、効果が小さい範囲にいる可能性がある。v7 は Hyperball だけの効果を測っていない。パラメータあたりのトークンを増やす世代で読み直す。見直し期限：2026-12-15。NorMuon と cautious weight decay の組み合わせと比べた研究は見つけていない。
