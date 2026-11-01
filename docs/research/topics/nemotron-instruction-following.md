# Nemotron-Instruction-Following-Chat-v1

nvidia/Nemotron-Instruction-Following-Chat-v1（commit 83dcd3a）。カードのライセンスは CC-BY-4.0 で、商用に使えるとある。英語だけ。`chat_if.jsonl` が 7GB、約43.6万行、`structured_outputs.jsonl` が 0.09GB、約5千行。

指示に従う部分は、[Tülu 3](../sources/tulu-3.md) の方法で問いを作り、GPT-OSS-120B と Qwen3-235B で答え、IFEval と IFBench の検証に全ターンが通ったものだけを残している。

2026-11-01 に `chat_if.jsonl` の12か所から1,862行を読んだ。

- `capability_target` は `chat` が1,438行、`instruction_following` が424行（23%）。全体では約10万行になる。
- `instruction_following` のうち、思考なしは79行（19%）で、全体では約2万行になる。
- 読んだ行のライセンスはすべて `odc-by-1.0`。
- `instruction_following` の返答は平均2.45回で、1行の返答の合計は平均3,139字、中央値2,371字。

評価の問題が混ざっていないかは確かめていない。指示学習のデータを作るときに除く処理がある。
