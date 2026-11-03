# 各課題の形式

v6 の評価で使う課題を、作者が定めた手順と照らした。2026-11-01 に、各リポジトリの README（MMLU と M-IFEval は評価コード）と、lm-evaluation-harness の定義を読んだ。論文の本文は読んでいない。

## 作者が手順を定めている課題

| 課題 | 作者の定め | 使っていた形 | 結果 |
|---|---|---|---|
| MMLU | 科目ごとの5例を付け、A〜D の確率を比べる | 続きの形（OLMES）だけ | 作者の形式を足した |
| M-IFEval-Ja | 公式の採点コードで、返答を切らず、すべての指示を確かめる | llm-jp-eval。既定は最初の空行で切る | 公式の採点コードに替える |
| MBPP | タスク 2〜4 の3例、同じ文面のプロンプト、テストは 11〜510 | harness の3例と同じ文面 | 合っている。例のタスク番号までは確かめていない |
| HumanEval | 補完だけを渡してテストを実行し、pass@k を出す | harness の pass@1 | 合っている |
| IFEval | 公式の採点コードで採点する | harness の移植 | 形は合っている。移植のコードは比べていない |
| GSM8K | 答えは `####` の後の数。プロンプトと例の数は定めていない | harness の5例と `####` の一致 | 合っている |
| BLiMP | 文の組で、容認できる方の確率が高いかを見る | harness の同じ比較 | 合っている |

## 作者がプロンプトを定めていない課題

README に定めがないので、公開の道具の定義で問う。

- KUCI、JMMLU：llm-jp-eval
- MGSM：harness。例文は配布されているが、数は README に書かれていない
- JCommonsenseQA、JNLI、JSQuAD：harness。JGLUE の表の指標は正解率で、JSQuAD は完全一致と F1

## 確かめていない課題

HellaSwag（README を取得できなかった）、ARC-Easy、ARC-Challenge、SciQ、PIQA、LAMBADA、xWinograd。

## 小さいモデルを伸ばす線の課題

2026-11-03 に、各課題の README とデータのカード、MMLU-Pro の評価コード、lm-evaluation-harness の MMLU-ProX の定義を読んだ。論文の本文は読んでいない。

Swallow の評価は、例を付けずに長く思考させる形に作り直している。たとえば MMLU-Pro と MMLU-ProX は例なし、温度 0.6、最長 32,768 トークンで、作者の定め（5例、温度0、出力は 2,048 トークンまで）と違う。IFBench は温度 0.6 で指示ごとの strict を、LiveCodeBench はプロンプトとコードの取り出し方を変えて測っている。

| 課題 | 作者の定め | 問い方 |
|---|---|---|
| MMLU-ProX（日本語と、他の言語の軽量版） | カードが lm-evaluation-harness の `mmlu_prox_{言語}` を指定している。日本語は5例、温度0、出力は 2,048 トークンまでで、「答えは (X) です」から取り出す | harness。軽量版は作者が公開した `mmlu_prox_lite_{言語}` の全問 |
| MMLU-Pro | 作者の `evaluate_from_local.py`。5例、温度0、出力は 2,048 トークンまでで、`answer is (X)` から取り出す | 作者のコード |
| IFBench | 作者の `run_eval`。温度0で、思考するモデルは出力の上限を広げ、思考を除いて採点する。論文は設問ごとの loose を報告する | 作者のコード |
| LiveCodeBench | 作者の `lcb_runner`。pass@1 と pass@5 | 作者のコード。問題は v6 で追加された175問 |
| PolyMath | 作者のプロンプト（`instruction.py`）と採点コード。正解率と、問いと同じ言語で答えたかを出す | 作者のコード |
| JamC-QA | 選択肢の記号でなく文字列を出力させ、完全一致を見る。FlexEval の `jamcqa` で測れる | FlexEval |
| BenchMAX Science | lm-evaluation-harness と BenchMAX のコードで測る | 作者のコード |
| GPQA Diamond | 作者のコードに zero_shot、5_shot、zero_shot_chain_of_thought、chain_of_thought の型がある | 作者のコードの zero_shot_chain_of_thought |
| JHumanEval | HumanEval と同じく、関数の頭と説明文を入力にする | HumanEval と同じ補完の形 |
| M-IFEval-Ja | 公式の採点コード | 公式の採点コード（v6 と同じ） |
| MT-Bench（英語、日本語） | GPT-4 が採点する。日本語版は Stability AI の FastChat | 測れない。gpt-4-0613 は 2026-10-23 に提供を終えた |
| AnswerCarefully | 作者の llm-jp-judge。採点は OpenAI か Azure の API で、例は gpt-4o-2024-08-06 | llm-jp-judge |
| BFCL | 作者の `bfcl generate` と `bfcl evaluate` | 作者のコード |
| RULER | 作者の生成と採点のスクリプト | 作者のコード |

作者が手順を定めていない課題は、公開の道具で問う。

- MGSM：harness（v6 と同じ）
- MATH-500、AIME 2024–2025、WMT20 英日・日英：Swallow の評価

作者の手順を確かめられなかった課題：

- HellaSwag：作者のリポジトリが DMCA の通知で見られない（2026-09-14 から）。v6 の外との比較と同じく harness で問う。
- XSTest：作者の説明を取得できなかった。
