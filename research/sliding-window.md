# Sliding window attention

一部の層の注意を、直前の決まった長さの窓に限る。nanochat は、4層のうち3層の注意を文脈の半分の窓に限り、最後の層は必ず文脈全体を見る（SSSL）。LOG（2026-01-11）の根拠は「Quick experiments showed `SSSL` … works well」だけ。

注意を文書の中だけで計算する場合（document-masking.md）、文書が窓より短ければ、窓を短くしても計算は減らない。
