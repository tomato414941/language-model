# FineWeb-Edu

HuggingFaceFW/fineweb-edu（ODC-By）は、FineWeb の英語の Web の文書を、教育的な質の判定器の点数で選んだもの。文書ごとに点数（int_score など）が付いている。

nanochat は FineWeb-Edu を別のデータに替えようとして、ClimbMix（climbmix.md）より前の5回はどれも負けた。そのうち LOG で読んだもの：

- ただの FineWeb：CORE が 0.2602 → 0.2241 と大きく悪くなった（2026-02-17、d26）。
- FinePDFs・DCLM・FineWeb-Edu を混ぜたもの：0.2602 → 0.2549 と少し悪くなった（2026-02-17、d26）。
- Olmo 3 の混ぜ方：d16 で CORE が 15.5 → 13.8（2026-01-15）。
