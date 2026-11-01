# 結果

## 世代ごとの費用

| 世代 | 読んだトークン | 費用 | 備考 |
|---|---|---|---|
| v2 | 約10億 | $3.96 | 試しを含む |
| v3 | 約30億 | 累計 $10.47 | v2 の続き |
| v4 | 約130億 | 累計 約$28.4 | v3 の続き |
| v5 | 約117億 | 約$21.7（長い文脈の学習まで含めて約$27.3） | ゼロから |

v6 の見込みと試しの費用は `plan.md` にある。

## v5 の点

評価の決まりは `evaluation.md` にある。下の点は lm-evaluation-harness で全問を評価した。指示学習の後の列が、v6 と比べる基準になる。

| 言語 | 課題 | 指標 | 事前学習の後 | 指示学習の後 |
|---|---|---|---|---|
| 英語 | HellaSwag | acc_norm | .349 | .356 |
| 英語 | ARC-Easy | acc_norm | .458 | .456 |
| 英語 | ARC-Challenge | acc_norm | .260 | .265 |
| 英語 | SciQ | acc | .824 | .830 |
| 英語 | PIQA | acc_norm | .609 | .611 |
| 英語 | LAMBADA | acc | .332 | .349 |
| 英語 | BLiMP | acc | .804 | .785 |
| 英語 | MMLU（続き） | acc_norm | .278 | .276 |
| 英語 | GSM8K | exact match | .023 | .030 |
| 英語 | HumanEval | pass@1 | .085 | .073 |
| 英語 | MBPP | pass@1 | .012 | .028 |
| 英語 | IFEval | strict | - | .209 |
| 日本語 | JCommonsenseQA | acc | .373 | .336 |
| 日本語 | KUCI（確率） | acc_norm | .425 | .417 |
| 日本語 | xWinograd | acc | .557 | .553 |
| 日本語 | JNLI | acc | .505 | .322 |
| 日本語 | JMMLU（確率） | acc_norm | .287 | .290 |
| 日本語 | JSQuAD | exact match | .161 | .232 |
| 日本語 | JHumanEval | pass@1 | .061 | .079 |
| 日本語 | MGSM | acc | .000 | .000 |

KUCI（確率）と JMMLU（確率）は自前の定義で、JHumanEval は llm-jp-eval で問えないので、v6 では測らない。

## v5 の llm-jp-eval の点

v2.1.5 で全問を問うた、指示学習の後のモデルの点。同じ重みを2回評価した。

| 課題 | 1回目 | 2回目 | 2回目の別の形 |
|---|---|---|---|
| M-IFEval-Ja（strict） | .163 | .186 | 返答全体 .209（loose .233） |
| KUCI | .243 | .242 | 会話形式 .001 |
| JMMLU | .004 | .004 | 会話形式 .015 |
| JCommonsenseQA（harness） | .336 | .335 | 会話形式 .426 |

KUCI と JMMLU の低さは主に答え方の形式による。`research/experiments/choice-answers.md` にある。
