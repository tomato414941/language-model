# u/w floor

更新の大きさと重みの大きさの比 ‖u‖/‖w‖ に下限を設け、小さすぎる更新を持ち上げる。weight decay を使わずに済む形もある。

[modded-nanogpt の最適化の部門](studies/modded-nanogpt-optimization-track.md)の #9 は、NorMuon に足して下限 0.35、weight decay なしで 3250 ステップ（n=8）。#10 の NorMuon（weight decay 0.025）も同じ 3250 ステップ。#46 は行ごとに下限 0.3825 をかけている。この部門の 1.62 億パラメータ、約3000ステップより大きな学習で試した結果は見つけていない。
