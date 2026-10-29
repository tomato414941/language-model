# 技術の記録

調べた技術ごとに1ファイル。何を調べ、どうしたか、取り入れたものはどうだったかを書く。新しく調べる前にここを見る。

## v6 に入れたもの

- [value-embeddings.md](value-embeddings.md)
- [head-dim-128.md](head-dim-128.md)
- [depth.md](depth.md)：32層
- [compute-optimal-size.md](compute-optimal-size.md)：約4億パラメータ × 約84億トークン
- [batch-size.md](batch-size.md)：約100万トークン
- [normuon.md](normuon.md)
- [cautious-weight-decay.md](cautious-weight-decay.md)
- [resid-lambdas.md](resid-lambdas.md)
- [x0-lambdas.md](x0-lambdas.md)
- [zero-output-init.md](zero-output-init.md)
- [lr-schedule.md](lr-schedule.md)：最後の20%で1%まで
- [checkpoint-averaging.md](checkpoint-averaging.md)
- [single-digit-tokens.md](single-digit-tokens.md)
- [dclm-edu.md](dclm-edu.md)

## v5 から引き継いだもの

- [swiglu.md](swiglu.md)：替えるかは決めていない
- [qk-norm.md](qk-norm.md)
- [attention-gate.md](attention-gate.md)
- [document-masking.md](document-masking.md)
- [fineweb-edu.md](fineweb-edu.md)
- [sft-learning-rate.md](sft-learning-rate.md)

## 決めていないもの、後回しにしたもの

- [data-repetition.md](data-repetition.md)
- [preference-optimization.md](preference-optimization.md)
- データの候補：[nemotron-cc.md](nemotron-cc.md)、[ultra-fineweb-l3.md](ultra-fineweb-l3.md)、[essential-web.md](essential-web.md)、[finepdfs-edu.md](finepdfs-edu.md)

## 入れなかったもの

- データ：[climbmix.md](climbmix.md)、[nemotron-climblab.md](nemotron-climblab.md)、[ultra-fineweb.md](ultra-fineweb.md)、[japanese-web-data.md](japanese-web-data.md)、[self-rephrasing.md](self-rephrasing.md)
- モデル：[vocabulary-size.md](vocabulary-size.md)、[untied-embeddings.md](untied-embeddings.md)、[value-residual.md](value-residual.md)、[sliding-window.md](sliding-window.md)、[bigram-hash-embedding.md](bigram-hash-embedding.md)、[logit-softcap.md](logit-softcap.md)、[layernorm-scaling.md](layernorm-scaling.md)、[partial-rope.md](partial-rope.md)、[moe.md](moe.md)、[mtp.md](mtp.md)、[leaky-relu-squared.md](leaky-relu-squared.md)、[orthogonal-init.md](orthogonal-init.md)、[xsa.md](xsa.md)
- 学習：[embedding-learning-rate.md](embedding-learning-rate.md)、[muoneq.md](muoneq.md)、[muon-plus.md](muon-plus.md)、[polar-express.md](polar-express.md)、[z-loss.md](z-loss.md)、[rlvr.md](rlvr.md)
- 計算：[liger.md](liger.md)、[fp8.md](fp8.md)、[h100.md](h100.md)、[interruptible-gpu.md](interruptible-gpu.md)
- 評価：[olmes-commonsense-tasks.md](olmes-commonsense-tasks.md)

## v5 の結果について調べたこと、評価

- [jcommonsenseqa-drop.md](jcommonsenseqa-drop.md)
- [nemotron-v2-ja-sft.md](nemotron-v2-ja-sft.md)
- [m-ifeval-ja.md](m-ifeval-ja.md)
- [elyza-tasks-100.md](elyza-tasks-100.md)
- [japanese-mt-bench.md](japanese-mt-bench.md)

## 前提にしていて、調べていないこと

- 根拠のほとんどは英語だけのモデルの結果で、日英半々のモデルにも当てはまるかは調べていない。
- 日英の比率、コードと数学の割合は v5 のままで、調べていない。
- 既存研究の結果を、自分たちのデータで比べて確かめてはいない。
- ローカル LLM のコミュニティ（r/LocalLLaMA）は読めず、調べていない。
