# Cautious Weight Decay

Cautious Weight Decay（arXiv 2510.12402）。UT Austin と Google。

更新の符号と重みの符号がそろっている要素にだけ weight decay をかける、1行の変更。AdamW、Lion、Muon のどれにも使え、新しいハイパーパラメータは要らない。普通の decoupled weight decay と違って、元の損失を最適化するとしている（bilevel の解釈）。

言語モデルの事前学習と ImageNet で、百万から十億パラメータの範囲で最終の損失と正解率が良くなったとしている。C4 で Chinchilla の比率で学習した 338M の図では、weight decay の係数を変えても、普通の weight decay より低い損失になっている。

読んだのは要旨、導入、結論まで。
