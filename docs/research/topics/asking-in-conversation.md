# 指示学習のモデルへの問い方

指示学習をしたモデルは会話形式で問い、事前学習だけのモデルは印のない文章で問う、という使い分けが一般的。2026-10-31 に調べた範囲では、次のとおり。

- OpenAI の [simple-evals](../sources/simple-evals.md) は、MMLU を user のメッセージとして Chat Completions に渡し、返答の最後の行の「Answer: A」を読む。公開する数字の説明のために出したもので、2025年7月から更新していない。
- DeepSeek は、[V3](../sources/deepseek-v3.md) で事前学習のモデル（5-shot など）と chat のモデルを分けて評価した。[R1](../sources/deepseek-r1.md) は、MMLU などに simple-evals の問いを使い、例を付けると成績が下がるとしている。自分たちのモデルに印をどう付けたかは、読んだ範囲には書かれていない。
- Ai2 の [OLMES](../sources/olmes.md) は、Base と Instruct で評価を分けている。設定に会話形式の指定があるが、どの課題の設定かは読んでいない。
- [lm-evaluation-harness](../sources/lm-evaluation-harness-chat.md) は、チャットや指示学習のモデルに `--apply_chat_template` を使うよう説明している。例は複数のターンになる。
- Swallow の [swallow-evaluation-instruct](../sources/swallow-evaluation.md) は、指示学習のモデルを Chat Completion API で評価し、例を付けず思考させて答えさせる形に変えた。
- Anthropic の[公開の API](https://docs.anthropic.com/en/api/messages) は user と assistant のターンで渡す。社内の評価の方法は確認していない。

llm-jp-eval の JMMLU は例なしで1トークンだけ答えさせる（[自分たちのモデルの結果](../experiments/choice-answers.md)）。自由に書かせて答えを読み取る形より、かなり狭い。

文章の単語の確率を比べる BLiMP と LAMBADA には問いがないので、会話形式では問わない。

問いの形で点は大きく動く。[例なしの問いでは、指示学習で GSM8K が下がる](../sources/instruction-tuned-vs-base.md)という比較があり、[採点方法を変えるとモデルの順位が変わる](../sources/multiple-choice-scoring.md)という比較もある。

今の評価（`evaluation.md`）も、知っていることは事前学習の後のモデルに印のない文章で、指示に従う力は指示学習の後のモデルに会話形式で問う。
