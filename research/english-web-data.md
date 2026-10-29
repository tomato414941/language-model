# 英語の Web データ

v6 の安定期では、v5 の DCLM（10%）を DCLM-Edu に替えた（026e91d）。FineWeb-Edu（20%）はそのまま。

DCLM-Edu（HuggingFaceTB/dclm-edu、CC-BY-4.0）は、DCLM を FineWeb-Edu の教育的な質の判定器で選び直したもの。カードには「educational score higher than 3」とあるが、データの先頭の文書は edu_int_score が 3、edu_score が 2.59 だった。データのカードによると、SmolLM2 の設定の 360M を2000億トークンで学習した比較で、DCLM と FineWeb-Edu より MMLU と ARC が良かった。一方、SmolLM2 1.7B の途中から入れた比較では、効果がその比較ほどは出なかったとも書かれている（数値は画像で、読んでいない）。

## ClimbMix

最初は ClimbMix に替えるつもりだった。nanochat の LOG（2026-03-04）では、FineWeb-Edu から ClimbMix に替えて、GPT-2 の性能に届くまでの時間が 2時間46分から2時間1分になった。それまでに試した5つのデータは、どれも FineWeb-Edu に勝てなかった。

しかし nvidia/Nemotron-ClimbMix のライセンスは CC BY-NC 4.0（非商用）だった。nanochat が使う karpathy/climbmix-400b-shuffle には MIT とあるが、元の制限が外れるわけではない。今のソースはすべて商用でも使えるライセンスなので、見送った。ただし「商用で使えないデータは使わない」は私が置いた前提で、ユーザーが決めたルールではない。

ClimbMix の混ぜ方は、PIQA、ARC-Easy、HellaSwag の検証用の問題での成績で選ばれている（CLIMB、arXiv 2504.13161）。テストの問題が入っているわけではないが、その種類の問題に向いた文章が多めになる。CLIMB は、ほかのタスクにも効果が及んだとしている。

## ほかの候補

- Ultra-FineWeb（openbmb/Ultra-FineWeb、Apache-2.0）：FineWeb を独自の判定器で選び直したもの。MiniCPM4 と MiniCPM5 の主な Web データ。カードに 1.2B・1000億トークンでの比較があるが、作った人たち自身の比較しかないので見送った。
- Nemotron-CC v2 / v2.1：商用でも使える。ただしカードに、これで学習したモデルには Qwen、DeepSeek、Phi-4 のライセンスの条件がかかりうるとある。他のモデルが書いた文章を含むと読み、他の人のモデルからの蒸留を禁じるルールに触れるとして見送った。ところが今のデータにも、他のモデルが書き直した Swallow の数学とコードが入っている。公開されているデータセットに含まれる生成文がこのルールに入るかは決まっておらず、この見送りは私の解釈による。
- Nemotron-ClimbLab：ClimbMix の元になった 1.2兆トークン。CC BY-NC 4.0。
- Ultra-FineWeb-L3：Ultra-FineWeb を、問答の生成と文体を変えた書き直しで作り直したもの。Nemotron-CC と同じ理由で見送った。
- FinePDFs-Edu（ODC-By）：調べていない。下の nanochat の結果は、Edu でない FinePDFs を混ぜたもの。
- Essential-Web（ODC-By）：文書ごとのメタデータで選り分けて使うためのデータで、データの中身を細かく選り分けないルールに触れるとして見送った。ただし、作った人たちが切り出した数学、コード、STEM、医療のデータセットも公開されていて、それらは丸ごと使える。調べていない。今のソースにも、ClimbLab-Ja の quality や FineWeb-Edu の int_score で切っているものがあり、ルールの線引きは決まっていない。

## nanochat が試して悪かったもの

- ただの FineWeb：CORE が 0.2602 → 0.2241 と大きく悪くなった。
- FinePDFs・DCLM・FineWeb-Edu を混ぜたもの：0.2602 → 0.2549 と少し悪くなった。
- Olmo 3 の混ぜ方：悪くなった。

v5 の FineWeb-Edu と DCLM の混ぜ方が良いという根拠も、特にない。
