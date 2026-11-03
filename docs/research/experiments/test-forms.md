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
