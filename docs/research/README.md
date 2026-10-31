# 調査の記録

新しく調べる前にここを見る。

- `topics/`：テーマごとの、今分かっていることのまとめ。1テーマ1ファイル。技術、データセット、評価の課題、道具、GPU など、使うかどうかを判断する対象を書く。新しい結果が出たら書き換える。
- `sources/`：他の人が公開した1つの出典に書いてあること。1出典1ファイル。論文、記事、公開の実験記録を置く。原則として書き換えない。読書ノートなど又聞きで読んだものは、原文を読んでいないことを書く。
- `experiments/`：自分たちの1回の調査や実験でしたことと、分かったこと。1回1ファイル。他の人のモデルでも、自分たちが測ったものはここに置く。

リンクは `topics/` から他の2つへの一方向にする。データセットのカードの内容はそのデータの `topics/` のメモに書き、コードは引用するメモに出典を書く。論文の本文やデータそのものは置かない。チャットの発言は入れず、公開の論文や記録で裏を取ってから、その出典のメモに書く。

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
- [llm-jp-4-sft-data.md](topics/llm-jp-4-sft-data.md)
- [choice-answer-data.md](topics/choice-answer-data.md)
- [fineinstructions.md](topics/fineinstructions.md)
- [curriculum-learning.md](topics/curriculum-learning.md)

### 計算

- [h100.md](topics/h100.md)
- [fp8.md](topics/fp8.md)
- [liger.md](topics/liger.md)
- [interruptible-gpu.md](topics/interruptible-gpu.md)

### 評価

- [asking-in-conversation.md](topics/asking-in-conversation.md)
- [m-ifeval-ja.md](topics/m-ifeval-ja.md)
- [elyza-tasks-100.md](topics/elyza-tasks-100.md)
- [japanese-mt-bench.md](topics/japanese-mt-bench.md)
- [olmes-commonsense-tasks.md](topics/olmes-commonsense-tasks.md)

## 出典

- [completep.md](sources/completep.md)
- [depth-delusion.md](sources/depth-delusion.md)
- [impact-of-depth.md](sources/impact-of-depth.md)
- [scale-efficiently.md](sources/scale-efficiently.md)
- [patch-level-training.md](sources/patch-level-training.md)
- [ema-nesterov.md](sources/ema-nesterov.md)
- [modded-nanogpt-optimization-track.md](sources/modded-nanogpt-optimization-track.md)
- [modded-nanogpt-anvil2.md](sources/modded-nanogpt-anvil2.md)
- [attention-residuals.md](sources/attention-residuals.md)
- [multiscreen.md](sources/multiscreen.md)
- [llm-jp-4.md](sources/llm-jp-4.md)
- [imu-1.md](sources/imu-1.md)
- [nanochat-log.md](sources/nanochat-log.md)
- [marin-nano-arch-ablations.md](sources/marin-nano-arch-ablations.md)
- [smol-training-playbook.md](sources/smol-training-playbook.md)
- [scaling-laws.md](sources/scaling-laws.md)
- [chinchilla.md](sources/chinchilla.md)
- [power-lines.md](sources/power-lines.md)
- [gemstones.md](sources/gemstones.md)
- [depth-to-width-interplay.md](sources/depth-to-width-interplay.md)
- [normuon.md](sources/normuon.md)
- [cautious-weight-decay.md](sources/cautious-weight-decay.md)
- [muon-plus.md](sources/muon-plus.md)
- [muoneq.md](sources/muoneq.md)
- [polar-express.md](sources/polar-express.md)
- [soap-muon.md](sources/soap-muon.md)
- [lawa.md](sources/lawa.md)
- [straight-to-zero.md](sources/straight-to-zero.md)
- [value-residual.md](sources/value-residual.md)
- [muddformer.md](sources/muddformer.md)
- [h-net.md](sources/h-net.md)
- [from-tokens-to-bytes.md](sources/from-tokens-to-bytes.md)
- [large-vocabulary.md](sources/large-vocabulary.md)
- [climb.md](sources/climb.md)
- [datasets-documents-repetitions.md](sources/datasets-documents-repetitions.md)
- [to-repeat-or-not.md](sources/to-repeat-or-not.md)
- [minimax-01.md](sources/minimax-01.md)
- [liger-kernel.md](sources/liger-kernel.md)
- [context-language-models.md](sources/context-language-models.md)
- [finephrase.md](sources/finephrase.md)
- [local-language-models.md](sources/local-language-models.md)
- [curriculum-learning.md](sources/curriculum-learning.md)
- [curriculum-dynamics.md](sources/curriculum-dynamics.md)
- [fineinstructions.md](sources/fineinstructions.md)
- [instruction-pretraining-composition.md](sources/instruction-pretraining-composition.md)
- [smollm2.md](sources/smollm2.md)

## 実験

- [jcommonsenseqa-drop.md](experiments/jcommonsenseqa-drop.md)
- [choice-answers.md](experiments/choice-answers.md)
- [tokens-per-character.md](experiments/tokens-per-character.md)

## 調べていないこと

- 根拠のほとんどは英語だけのモデルの結果で、日英半々のモデルにも当てはまるかは調べていない。
- 日英の比率、コードと数学の割合は調べていない。
- 既存研究の結果を、自分たちのデータで比べて確かめてはいない。
- ローカル LLM のコミュニティ（r/LocalLLaMA）は読めず、調べていない。
