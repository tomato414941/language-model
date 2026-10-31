# OLMES

Gu ら、OLMES: A Standard for Language Model Evaluations（arXiv 2406.08446）と、実装の [allenai/olmes](https://github.com/allenai/olmes)。論文は要旨、結論、導入の比較の部分を読み、実装は README と設定の一部を読んだ。

- 同じ課題でも、問いの形、例の選び方、確率の正規化、課題の形式で点が大きく変わる。論文は ARC-Challenge の公表値が文献ごとに違う例を挙げている。これらを決めた標準を提案している。
- 小さな事前学習のモデルに必要な穴埋めの形と、大きなモデルに使える選択式の形を、比べられるようにしている。
- 実装は Base と Instruct で評価の手順を分けていて、設定に `use_chat_format` がある。どの課題に付いているかは確かめていない。
