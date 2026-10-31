# Attention Residuals

固定の残差の接続を、前の層の出力に対する注意に置き換える。[Kimi Team](studies/attention-residuals.md)は、1.25倍の計算量に相当する改善としている。小さいモデルでの公開の結果は見つけていない。

modded-nanogpt の MUDD の混ぜ合わせも、前の層の出力を混ぜる点で近いが、[記録 #92](studies/modded-nanogpt-anvil2.md)での効果は 6.1 ミリナットだった。
