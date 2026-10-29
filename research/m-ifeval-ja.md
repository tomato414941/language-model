# M-IFEval-Ja

2026-10-28 に、指示学習のあとで日本語の指示に従えるかを測るために足した。

公式の採点コード（lightblue-tech/M-IFEval、commit 10b874d）を取り込み、公開されている Qwen2.5-0.5B と gpt-4o-mini の結果と、strict と loose のどちらも 172/172 で一致することを確かめた（f217b96）。その後 win1 が、公開されている問題文と採点のまま測れる llm-jp-eval v2.1.5 に移した（a95b46f）。
