# DCLM-Edu

HuggingFaceTB/dclm-edu（CC-BY-4.0）は、DCLM を FineWeb-Edu の教育的な質の判定器で選び直したもの。カードには「educational score higher than 3」とあるが、データの先頭の文書は edu_int_score が 3、edu_score が 2.59 だった。

カードによると、SmolLM2 の設定の 360M を2000億トークンで学習した比較で、DCLM と FineWeb-Edu より MMLU と ARC が良かった。一方、SmolLM2 1.7B の途中から入れた比較では、効果がその比較ほどは出なかったとも書かれている（数値は画像で、読んでいない）。
