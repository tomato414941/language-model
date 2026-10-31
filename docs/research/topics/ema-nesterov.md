# EMA-Nesterov

パラメータの更新量の EMA の方向に先読みした位置で勾配を計算する、オプティマイザの包み。重みと同じ大きさの状態を1つ持つ。

[論文](studies/ema-nesterov.md)では、調整済みの Muon が 3.28 に届くまでのステップが6%減り、1B の Llama の形でも perplexity が下がった。NorMuon でも効いたのは 124M での結果。
