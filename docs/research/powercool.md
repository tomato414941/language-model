# PowerCool

学習率を一定に保ったあと、終点までの残りステップのべき乗で下げる。lr = min(初期値, c·(終点 − step)^p)。

[modded-nanogpt の最適化の部門](studies/modded-nanogpt-optimization-track.md)の #20 で入り、#46 では p=1.2。#46 の説明には、この記録では WSD の下げ方とあまり変わらず、効果は小さいかもしれないとある。
