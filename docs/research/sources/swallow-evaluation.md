# Swallow の評価

[Swallow LLM Leaderboard の説明](https://swallow-llm.github.io/leaderboard/about.en.html)と、[swallow-evaluation-instruct](https://github.com/swallow-llm/swallow-evaluation-instruct) の README の一部を読んだ。

指示学習のモデルの評価を作り直し、例を付けずに思考させて答えさせる形に変えた。日本語は JEMHopQA、MMLU-ProX、GPQA、MATH-100、JHumanEval、M-IFEval-Ja の6つ。実装は Chat Completion API を通す。思考するモデルは確率的に生成する（temperature 0.6、top-p 0.95）。
