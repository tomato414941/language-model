# Sliding window attention

入れていない。

nanochat は、4層のうち3層の注意を文脈の半分の窓に限り、最後の層は必ず文脈全体を見る（SSSL）。LOG（2026-01-11）の根拠は「Quick experiments showed `SSSL` … works well」だけ。

v6 は注意を文書の中だけで計算するので（document-masking.md）、文書が窓（4096トークン）より短ければ計算は減らない。v6 のデータの文書の長さは確かめていない。
