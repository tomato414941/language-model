# FineWeb-Edu

v5 から引き続き、英語の Web データの中心に使う（安定期の20%）。安定期の en-web は、int_score が4以上の文書だけを使う。v6 のために見直してはいない。

nanochat は FineWeb-Edu を別のデータに替えようとして、ClimbMix（climbmix.md）より前の5回はどれも負けた。そのうち LOG で読んだもの：

- ただの FineWeb：CORE が 0.2602 → 0.2241 と大きく悪くなった（2026-02-17、d26）。
- FinePDFs・DCLM・FineWeb-Edu を混ぜたもの：0.2602 → 0.2549 と少し悪くなった（2026-02-17、d26）。
- Olmo 3 の混ぜ方：d16 で CORE が 15.5 → 13.8（2026-01-15）。

v5 の、FineWeb-Edu と DCLM を混ぜる形が良いという根拠も、特にない。
