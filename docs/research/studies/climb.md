# CLIMB

Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training（arXiv 2504.13161）。

データを意味の空間で分けてグループにし、小さな代理モデルと予測器で、課題の成績が上がる混ぜ方を繰り返し探す。

- 見つけた混ぜ方で 400B トークン学習した 1B モデルは、Llama-3.2-1B を2.0%上回った。比べたのは12課題の平均。
- 特定の分野（社会科学など）に合わせて探すと、無作為に選ぶより5%良くなった。
- 20のグループに分けた 1.2T トークンの Nemotron-ClimbLab と、400B トークンの Nemotron-ClimbMix を公開している。

読んだのは要旨、導入、結論まで。
