# 日本語の対話の評価

2026-10-28 に、指示学習のあとで日本語の対話を測る手段がなかったので足した。どれも公開されていて広く使われている評価を、公開されている問題文と採点のまま使う。

- M-IFEval-Ja：公式の採点コードを取り込み（lightblue-tech/M-IFEval、commit 10b874d）、公開されている Qwen2.5-0.5B と gpt-4o-mini の結果と 172/172 で一致することを確かめた（f217b96）。その後 win1 が llm-jp-eval v2.1.5 で測る形に移した（a95b46f）。
- ELYZA-tasks-100 と Japanese MT-Bench：採点役のモデルに点をつけさせる（`python -m pretrain judge`、bc6f0e7）。採点役は Swallow の評価が Japanese MT-Bench に使っている openai/gpt-4o-2024-08-06 で、温度0。ELYZA は ELYZA が公開した採点の指示文で1〜5点。MT-Bench は Stability AI の日本語の採点の指示文で、1ターン目だけを1〜10点で採点し、数学・推論・コードには GPT-4 の参考解答を渡す。私は llm-jp-3-150m-instruct3 で数問だけ動くことを確かめた。

公開元はどちらも GPT-4 で採点していた。採点役が違うと点は比べられない。
