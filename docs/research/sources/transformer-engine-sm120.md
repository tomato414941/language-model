# TransformerEngine issue #2668（MXFP8 on SM120）

NVIDIA/TransformerEngine の issue。ページを読んだ。

- SM120 では、Float8BlockScaling は MXFP8 の経路を通して動く。スケールの変換の負担は小さいとされる。
- MXFP8 は順伝播は動くが、逆伝播に必要な cuBLAS の配置がない。2026-07 の文書の変更では、MXFP8 は計算能力 12.0 に対応しないとしている。
- 保守者は 2026-02 に、SM120 の対応は SM100/103 より遅れていると答えた。
