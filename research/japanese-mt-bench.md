# Japanese MT-Bench

2026-10-28 に、指示学習のあとで日本語の対話を測るために足した（`python -m pretrain judge`、bc6f0e7）。

Stability AI の FastChat の日本語の採点の指示文のまま、1ターン目だけを1〜10点で採点させ、数学・推論・コードには GPT-4 の参考解答を渡す。採点役は、Swallow の評価が Japanese MT-Bench に使っている openai/gpt-4o-2024-08-06 で、温度0。公開元は GPT-4 で採点していたので、採点役が違う点とは比べられない。私は llm-jp-3-150m-instruct3 で数問だけ動くことを確かめた。
