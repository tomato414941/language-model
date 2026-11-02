# Leveraging High-Resource English Corpora for Japanese Medicine

Kobayashi ら、Leveraging High-Resource English Corpora for Cross-lingual Domain Adaptation in Low-Resource Japanese Medicine via Continued Pre-training（Findings of EMNLP 2025）。節3.1 の一部、節4.2 の一部、節6、限界、付録 D の一部を読んだ。

LLM-jp-3 の 13B（2.1T トークンで学習済み）に、日英の割合を変えた6つの医療のコーパスを続けて学習し、日英の医療の課題との相関を見た。コーパスは最大 79B トークンで、機械翻訳も含む。

- 英語のトークン数は、日本語の課題とも強く相関した（USMLEQA-Jp r = 0.989、MMLU-Medical-Jp r = 0.975）。
- 日本語のトークン数は、日本の医師国家試験（IgakuQA）と強く相関した（r = 0.956）。
- 英語の割合は、日本語の専門的な固有表現の課題とは負に相関した。
- 対訳は控えめに使うのがよいとしている。

標本は6つで相関の分析。全トークンも同時に増えている。医療の分野の結果で、一般の課題の結果ではない。
