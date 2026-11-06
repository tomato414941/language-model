# Mellum2

Mellum2 Technical Report（arXiv 2605.31268）。MoE の構成と学習の設定を読んだ。

専門家 64（大きさ 896）から8つを選び、トークンを捨てない。router は float32 で計算する。負荷は更新全体で数える均しの損失（重み 1e-3）で均し、router の z-loss（1e-3）を足す。Muon で問題なく学習できた。
