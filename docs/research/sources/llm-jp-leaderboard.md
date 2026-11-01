# Open Japanese LLM Leaderboard の結果

Hugging Face の `llm-jp/leaderboard-contents`（最終更新 2025-10-23）の表を読んだ。862行で、llm-jp-eval で評価したモデルの点とパラメータ数がある。評価の版と例の数は行ごとに確かめていない。

1B 以下の15モデルの中央値は、JCommonsenseQA .170、KUCI .244、JMMLU .015。JNLI は5モデルが .553 で同じ値になる（llm-jp-3-150m、Qwen2-0.5B など）。

| モデル | 種類 | パラメータ | JCommonsenseQA | JNLI | KUCI | JMMLU |
|---|---|---|---|---|---|---|
| llm-jp-3-150m | 事前学習 | 0.15B | .203 | .553 | .248 | .239 |
| Qwen2-0.5B | 事前学習 | 0.49B | .308 | .553 | .249 | .277 |
| Qwen2.5-0.5B | 事前学習 | 0.49B | .369 | .518 | .265 | .300 |
| Qwen2.5-0.5B-Instruct | 指示学習 | 0.49B | .382 | .303 | .301 | .315 |
| llm-jp-3-980m | 事前学習 | 0.99B | .184 | .553 | .251 | .254 |
| Llama-3.2-1B-Instruct | 指示学習 | 1.24B | .468 | .531 | .286 | .280 |
