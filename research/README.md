# 技術の記録

調べた技術ごとに1ファイル。新しく調べる前にここを見る。

## モデルの形

- [value-embeddings.md](value-embeddings.md)
- [head-dim-128.md](head-dim-128.md)
- [depth.md](depth.md)
- [compute-optimal-size.md](compute-optimal-size.md)
- [swiglu.md](swiglu.md)
- [qk-norm.md](qk-norm.md)
- [attention-gate.md](attention-gate.md)
- [document-masking.md](document-masking.md)
- [zero-output-init.md](zero-output-init.md)
- [resid-lambdas.md](resid-lambdas.md)
- [x0-lambdas.md](x0-lambdas.md)
- [value-residual.md](value-residual.md)
- [untied-embeddings.md](untied-embeddings.md)
- [sliding-window.md](sliding-window.md)
- [bigram-hash-embedding.md](bigram-hash-embedding.md)
- [logit-softcap.md](logit-softcap.md)
- [layernorm-scaling.md](layernorm-scaling.md)
- [partial-rope.md](partial-rope.md)
- [moe.md](moe.md)
- [mtp.md](mtp.md)
- [leaky-relu-squared.md](leaky-relu-squared.md)
- [orthogonal-init.md](orthogonal-init.md)
- [xsa.md](xsa.md)
- [h-net.md](h-net.md)

## 最適化と学習

- [batch-size.md](batch-size.md)
- [normuon.md](normuon.md)
- [cautious-weight-decay.md](cautious-weight-decay.md)
- [lr-schedule.md](lr-schedule.md)
- [checkpoint-averaging.md](checkpoint-averaging.md)
- [embedding-learning-rate.md](embedding-learning-rate.md)
- [muoneq.md](muoneq.md)
- [muon-plus.md](muon-plus.md)
- [polar-express.md](polar-express.md)
- [z-loss.md](z-loss.md)
- [preference-optimization.md](preference-optimization.md)
- [rlvr.md](rlvr.md)

## トークナイザ

- [pretokenization.md](pretokenization.md)
- [single-digit-tokens.md](single-digit-tokens.md)
- [vocabulary-size.md](vocabulary-size.md)

## データ

- [fineweb-edu.md](fineweb-edu.md)
- [dclm-edu.md](dclm-edu.md)
- [climbmix.md](climbmix.md)
- [nemotron-climblab.md](nemotron-climblab.md)
- [ultra-fineweb.md](ultra-fineweb.md)
- [ultra-fineweb-l3.md](ultra-fineweb-l3.md)
- [nemotron-cc.md](nemotron-cc.md)
- [finepdfs-edu.md](finepdfs-edu.md)
- [essential-web.md](essential-web.md)
- [japanese-web-data.md](japanese-web-data.md)
- [data-repetition.md](data-repetition.md)
- [self-rephrasing.md](self-rephrasing.md)
- [nemotron-v2-ja-sft.md](nemotron-v2-ja-sft.md)

## 計算

- [h100.md](h100.md)
- [fp8.md](fp8.md)
- [liger.md](liger.md)
- [interruptible-gpu.md](interruptible-gpu.md)

## 評価

- [m-ifeval-ja.md](m-ifeval-ja.md)
- [elyza-tasks-100.md](elyza-tasks-100.md)
- [japanese-mt-bench.md](japanese-mt-bench.md)
- [olmes-commonsense-tasks.md](olmes-commonsense-tasks.md)

## 結果

- [jcommonsenseqa-drop.md](jcommonsenseqa-drop.md)

## 調べていないこと

- 根拠のほとんどは英語だけのモデルの結果で、日英半々のモデルにも当てはまるかは調べていない。
- 日英の比率、コードと数学の割合は調べていない。
- 既存研究の結果を、自分たちのデータで比べて確かめてはいない。
- ローカル LLM のコミュニティ（r/LocalLLaMA）は読めず、調べていない。
