# ELYZA-tasks-100

2026-10-28 に、指示学習のあとで日本語の対話を測るために足した（`python -m pretrain judge`、bc6f0e7）。

ELYZA が公開した採点の指示文のまま、採点役のモデルに1〜5点をつけさせ、数字として読めない採点は除く。採点役は openai/gpt-4o-2024-08-06 で、温度0。ELYZA は GPT-4 で採点していたので、ELYZA の公開した点とは比べられない。llm-jp-3-150m-instruct3 で、数問だけ動くことを確かめた。
