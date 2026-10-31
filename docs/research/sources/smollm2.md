# SmolLM2

Allal ら、SmolLM2（arXiv 2502.02737）。Web データの比較と、小さなモデルの節を読んだ。

## Web データ

FineWeb-Edu と DCLM をそれぞれ 350B トークン学習して比べた。FineWeb-Edu は MMLU、ARC、OpenBookQA で高く、DCLM は HellaSwag と CommonsenseQA で高かった。

## 小さなモデル

135M を 2T トークン、360M を 4T トークン学習した。

- その長さで混ぜ方を比べ直し、DCLM を FineWeb-Edu の判定器で選んで、点0を除き、点1と2を減らしたものが最良だった。
- 1.7B と違い、段階を分けず、質の高いデータで通す形が良かった。
- SFT は SmolTalk から、関数呼び出しのような複雑な指示と難しい例を除いたものを使った。モデルの容量に合わせるためと書いている。そのあとで UltraFeedback の DPO を行った。
