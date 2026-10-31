# H-Net

Dynamic Chunking for End-to-End Hierarchical Sequence Modeling（arXiv 2507.07955）。

トークナイザをなくし、バイトをどこでまとめるかを、モデルと一緒に学習する。計算量とデータをそろえると、1段の H-Net は BPE の Transformer を上回った。段を重ねるとデータに対する伸びが良くなり、2倍の大きさのトークンの Transformer に並ぶ。中国語、コード、DNA のようにトークナイザの決まりが弱いものほど差が大きいとしている。

読んだのは要旨と導入まで。改良版の結果は [From Tokens to Bytes](from-tokens-to-bytes.md) にある。
