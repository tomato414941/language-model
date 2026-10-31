# 調査の記録

テーマごとのまとめは `topics/` に、研究ごとの記録は `studies/` に、1つで1ファイル。研究は、論文、記事、公開の実験記録、自分たちの調査のどれか1つ。まとめは新しい結果が出たら書き換え、研究の記録は原則として書き換えない。新しく調べる前にここを見る。

## テーマ

### モデルの形

- [value-embeddings.md](topics/value-embeddings.md)
- [head-dim-128.md](topics/head-dim-128.md)
- [depth.md](topics/depth.md)
- [compute-optimal-size.md](topics/compute-optimal-size.md)
- [swiglu.md](topics/swiglu.md)
- [qk-norm.md](topics/qk-norm.md)
- [attention-gate.md](topics/attention-gate.md)
- [document-masking.md](topics/document-masking.md)
- [zero-output-init.md](topics/zero-output-init.md)
- [resid-lambdas.md](topics/resid-lambdas.md)
- [x0-lambdas.md](topics/x0-lambdas.md)
- [value-residual.md](topics/value-residual.md)
- [untied-embeddings.md](topics/untied-embeddings.md)
- [sliding-window.md](topics/sliding-window.md)
- [bigram-hash-embedding.md](topics/bigram-hash-embedding.md)
- [logit-softcap.md](topics/logit-softcap.md)
- [layernorm-scaling.md](topics/layernorm-scaling.md)
- [partial-rope.md](topics/partial-rope.md)
- [moe.md](topics/moe.md)
- [mtp.md](topics/mtp.md)
- [leaky-relu-squared.md](topics/leaky-relu-squared.md)
- [orthogonal-init.md](topics/orthogonal-init.md)
- [xsa.md](topics/xsa.md)
- [h-net.md](topics/h-net.md)
- [attention-residuals.md](topics/attention-residuals.md)
- [screening.md](topics/screening.md)
- [mudd.md](topics/mudd.md)
- [narrow-qk.md](topics/narrow-qk.md)
- [virtual-attention-cap.md](topics/virtual-attention-cap.md)
- [completep.md](topics/completep.md)

### 最適化と学習

- [batch-size.md](topics/batch-size.md)
- [normuon.md](topics/normuon.md)
- [cautious-weight-decay.md](topics/cautious-weight-decay.md)
- [lr-schedule.md](topics/lr-schedule.md)
- [checkpoint-averaging.md](topics/checkpoint-averaging.md)
- [embedding-learning-rate.md](topics/embedding-learning-rate.md)
- [muoneq.md](topics/muoneq.md)
- [muon-plus.md](topics/muon-plus.md)
- [polar-express.md](topics/polar-express.md)
- [z-loss.md](topics/z-loss.md)
- [preference-optimization.md](topics/preference-optimization.md)
- [rlvr.md](topics/rlvr.md)
- [patch-level-training.md](topics/patch-level-training.md)
- [ema-nesterov.md](topics/ema-nesterov.md)
- [sampled-softmax.md](topics/sampled-softmax.md)
- [soap-muon.md](topics/soap-muon.md)
- [u-w-floor.md](topics/u-w-floor.md)
- [radial-brake.md](topics/radial-brake.md)
- [powercool.md](topics/powercool.md)
- [anvil.md](topics/anvil.md)
- [sequence-length-schedule.md](topics/sequence-length-schedule.md)
- [batch-size-schedule.md](topics/batch-size-schedule.md)

### トークナイザ

- [pretokenization.md](topics/pretokenization.md)
- [single-digit-tokens.md](topics/single-digit-tokens.md)
- [vocabulary-size.md](topics/vocabulary-size.md)

### データ

- [fineweb-edu.md](topics/fineweb-edu.md)
- [dclm-edu.md](topics/dclm-edu.md)
- [climbmix.md](topics/climbmix.md)
- [nemotron-climblab.md](topics/nemotron-climblab.md)
- [ultra-fineweb.md](topics/ultra-fineweb.md)
- [ultra-fineweb-l3.md](topics/ultra-fineweb-l3.md)
- [nemotron-cc.md](topics/nemotron-cc.md)
- [finepdfs-edu.md](topics/finepdfs-edu.md)
- [essential-web.md](topics/essential-web.md)
- [japanese-web-data.md](topics/japanese-web-data.md)
- [data-repetition.md](topics/data-repetition.md)
- [self-rephrasing.md](topics/self-rephrasing.md)
- [finephrase.md](topics/finephrase.md)
- [nemotron-v2-ja-sft.md](topics/nemotron-v2-ja-sft.md)

### 計算

- [h100.md](topics/h100.md)
- [fp8.md](topics/fp8.md)
- [liger.md](topics/liger.md)
- [interruptible-gpu.md](topics/interruptible-gpu.md)

### 評価

- [m-ifeval-ja.md](topics/m-ifeval-ja.md)
- [elyza-tasks-100.md](topics/elyza-tasks-100.md)
- [japanese-mt-bench.md](topics/japanese-mt-bench.md)
- [olmes-commonsense-tasks.md](topics/olmes-commonsense-tasks.md)

## 研究

- [completep.md](studies/completep.md)
- [depth-delusion.md](studies/depth-delusion.md)
- [impact-of-depth.md](studies/impact-of-depth.md)
- [scale-efficiently.md](studies/scale-efficiently.md)
- [patch-level-training.md](studies/patch-level-training.md)
- [ema-nesterov.md](studies/ema-nesterov.md)
- [modded-nanogpt-optimization-track.md](studies/modded-nanogpt-optimization-track.md)
- [modded-nanogpt-anvil2.md](studies/modded-nanogpt-anvil2.md)
- [attention-residuals.md](studies/attention-residuals.md)
- [multiscreen.md](studies/multiscreen.md)
- [llm-jp-4.md](studies/llm-jp-4.md)
- [imu-1.md](studies/imu-1.md)
- [nanochat-log.md](studies/nanochat-log.md)
- [marin-nano-arch-ablations.md](studies/marin-nano-arch-ablations.md)
- [smol-training-playbook.md](studies/smol-training-playbook.md)
- [scaling-laws.md](studies/scaling-laws.md)
- [chinchilla.md](studies/chinchilla.md)
- [power-lines.md](studies/power-lines.md)
- [gemstones.md](studies/gemstones.md)
- [depth-to-width-interplay.md](studies/depth-to-width-interplay.md)
- [normuon.md](studies/normuon.md)
- [cautious-weight-decay.md](studies/cautious-weight-decay.md)
- [muon-plus.md](studies/muon-plus.md)
- [muoneq.md](studies/muoneq.md)
- [polar-express.md](studies/polar-express.md)
- [soap-muon.md](studies/soap-muon.md)
- [lawa.md](studies/lawa.md)
- [straight-to-zero.md](studies/straight-to-zero.md)
- [value-residual.md](studies/value-residual.md)
- [muddformer.md](studies/muddformer.md)
- [h-net.md](studies/h-net.md)
- [from-tokens-to-bytes.md](studies/from-tokens-to-bytes.md)
- [large-vocabulary.md](studies/large-vocabulary.md)
- [climb.md](studies/climb.md)
- [datasets-documents-repetitions.md](studies/datasets-documents-repetitions.md)
- [to-repeat-or-not.md](studies/to-repeat-or-not.md)
- [minimax-01.md](studies/minimax-01.md)
- [liger-kernel.md](studies/liger-kernel.md)
- [context-language-models.md](studies/context-language-models.md)
- [finephrase.md](studies/finephrase.md)
- [local-language-models.md](studies/local-language-models.md)
- [jcommonsenseqa-drop.md](studies/jcommonsenseqa-drop.md)

## 調べていないこと

- 根拠のほとんどは英語だけのモデルの結果で、日英半々のモデルにも当てはまるかは調べていない。
- 日英の比率、コードと数学の割合は調べていない。
- 既存研究の結果を、自分たちのデータで比べて確かめてはいない。
- ローカル LLM のコミュニティ（r/LocalLLaMA）は読めず、調べていない。
