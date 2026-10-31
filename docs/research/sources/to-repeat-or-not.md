# To Repeat or Not To Repeat

To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis（arXiv 2305.13230）。Xue ら。

事前学習のデータを繰り返すと、過学習で悪くなる。悪くなり方に大きく効くのは、データの量、モデルのパラメータ、学習の目的で、データの質とモデルの計算量の影響は小さかった。正則化はほとんど効かないが、dropout だけはよく効く。ただしモデルを大きくするときの調整が難しい。MoE を使うと、計算の重い密なモデルの振る舞いを安く予測できるとしている。

読んだのは要旨、導入、結論まで。
