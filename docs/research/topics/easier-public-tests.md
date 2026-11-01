# より簡単な公開テスト

点が低くて差がつかない課題があるときに足す候補。v5 の指示学習の後では、MGSM .000、JMMLU .004、MBPP .028、GSM8K .030、HumanEval .073 が低い。どの候補も v5 と v6 では測っていないので、足すときは v5 も測り直す。

[llm-jp-eval](../sources/llm-jp-eval-datasets.md) に入っている候補。

| 分野 | 課題 | 答えの形式 |
|---|---|---|
| 文法 | [JBLiMP](../sources/jblimp.md)、JCoLA | 記号 |
| 算数 | MAWPS | LaTeX の答え |
| 文の類似と含意 | JSTS、JSICK、JaNLI、Jamp、JSeM | タグ付きの答え |
| 常識と知識 | JCommonsenseMorality、JamC-QA | 記号 |
| 知識 | NIILC、JEMHopQA | タグ付きの答え |
| 英語の常識 | OpenBookQA、WinoGrande_xl | 記号 |

JBLiMP は GPT-2 や n-gram でも約75%で、小さなモデルでも差を読める可能性がある。ただ331組しかなく、作者の比較は文の確率で、llm-jp-eval の記号の形式とは違う。

[Arkios](../sources/arkios.md) では、小さなモデルが記号の形式だけで、言語によらず偶然の水準に落ちた。記号で答える課題は、形式を身につけるまで簡単にはならない。v6 は記号で答える学習をしたので、効くかは v6 の点で分かる。

OpenBookQA、WinoGrande と、harness の BoolQ は、[外のモデルの表](outside-comparison.md)と直接並べられる利点もある。

コードの簡単な公開テストは見つけていない。

v6 の本番の計画には入れていない。v6 の点で差がつかない課題が分かったときに足す。見直し期限：2026-11-15。
