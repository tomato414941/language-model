# Attention Residuals

Kimi Team、Attention Residuals（arXiv 2603.15031、2026-03）。実装は GitHub の MoonshotAI/Attention-Residuals。

各層の出力を同じ重みで足し続ける残差の接続を、それまでの層の出力に対する softmax の注意に置き換える。Block AttnRes は層をブロックに分け、ブロックの表現だけを見る。

- スケーリング則の実験で、Block AttnRes は基準を1.25倍の計算量で学習したときと同じ損失になったとしている。
- 普通の学習では、Full AttnRes に要る層の出力は誤差逆伝播のために既に持っているので、メモリはほとんど増えない。
- 48B（活性3B）の Kimi Linear に入れて 1.4T トークンで事前学習し、評価したすべての課題で基準を上回った。

読んだのは冒頭と手法の説明まで。スケーリング則の実験の大きさの範囲は確かめていない。
