# チェックポイントの平均

学習の途中で残した重みを、学習のあとで平均する。学習の費用は増えない。

- [IMU-1](../sources/imu-1.md)（arXiv 2602.02522）は、減衰期に5000ステップおきに残した最後の10個に、あとから EMA（β=0.8）をかけ、ベンチマークの平均が +0.014 になった。
- modded-nanogpt の[最適化の部門](../sources/modded-nanogpt-optimization-track.md)の #45 は、学習中に重みの EMA を持ち、最後に混ぜて、2750 ステップを 2720 ステップにした（Tail-EMA）。[記録 #92](../sources/modded-nanogpt-anvil2.md)では、最後の重みの作り方を外すと損失が 20.2 ミリナット悪くなる。
- [nanochat の LOG](../sources/nanochat-log.md)（2026-03-24）には、EMA と SWA は試して効かなかったとある。
- [LAWA](../sources/lawa.md)（arXiv 2306.03241）は、OpenWebText で 125M の検証損失が 2.963 → 2.917（EMA 2.949、SWA 2.952）、355M で 2.855 → 2.819 になった。学習率が高いほど、残す間隔が広いほど効く、としている。
- [Signal and Noise](../sources/signal-and-noise.md) は、途中のチェックポイントの出力を平均すると評価の雑音が減り、判断の信頼性が上がるとしている。重みの平均ではなく出力の平均の話。
