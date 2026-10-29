# 英語の Web データ

v6 の安定期では、v5 の DCLM（10%）を DCLM-Edu に替えた（026e91d）。FineWeb-Edu（20%）はそのまま。

DCLM-Edu（HuggingFaceTB/dclm-edu、CC-BY-4.0）は、DCLM のうち、FineWeb-Edu の教育的な質の判定器のスコアが3より高いもの。データのカードによると、SmolLM2 の設定の 360M を2000億トークンで学習した比較で、DCLM と FineWeb-Edu より MMLU と ARC が良かった。一方、SmolLM2 1.7B の途中から入れた比較では、効果がその比較ほどは出なかったとも書かれている（数値は画像で、読んでいない）。

## ClimbMix

最初は ClimbMix に替えるつもりだった。nanochat の LOG（2026-03-04）では、FineWeb-Edu から ClimbMix に替えて、GPT-2 の性能に届くまでの時間が 2時間46分から2時間1分になった。それまでに試した5つのデータは、どれも FineWeb-Edu に勝てなかった。

しかし nvidia/Nemotron-ClimbMix のライセンスは CC BY-NC 4.0（非商用）だった。nanochat が使う karpathy/climbmix-400b-shuffle には MIT とあるが、元の制限が外れるわけではない。今のソースはすべて商用でも使えるライセンスなので、見送った。

ClimbMix の混ぜ方は、PIQA、ARC-Easy、HellaSwag の検証用の問題での成績で選ばれている（CLIMB、arXiv 2504.13161）。テストの問題が入っているわけではないが、その種類の問題に向いた文章が多めになる。CLIMB は、ほかのタスクにも効果が及んだとしている。

## ほかの候補

- Ultra-FineWeb（openbmb/Ultra-FineWeb、Apache-2.0）：FineWeb を独自の判定器で選び直したもの。MiniCPM4 と MiniCPM5 の主な Web データ。カードに 1.2B・1000億トークンでの比較があるが、作った人たち自身の比較しかないので見送った。
- Nemotron-CC v2 / v2.1：商用でも使える。ただしカードに、これで学習したモデルには Qwen、DeepSeek、Phi-4 のライセンスの条件がかかりうるとある。他のモデルが書いた文章を含むと読み、他の人のモデルからの蒸留を禁じるルールに従って見送った。

## nanochat が試して悪かったもの

- ただの FineWeb：CORE が 0.2602 → 0.2241 と大きく悪くなった。
- FinePDFs・DCLM・FineWeb-Edu を混ぜたもの：0.2602 → 0.2549 と少し悪くなった。
- Olmo 3 の混ぜ方：悪くなった。

v5 の FineWeb-Edu と DCLM の混ぜ方が良いという根拠も、特にない。
