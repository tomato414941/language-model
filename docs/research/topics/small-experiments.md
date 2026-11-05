# 小さな実験で決める

本番より小さなモデルと少ないトークンで比べて、データや設定を決める。

- [DataDecide](../sources/datadecide.md) では、150M の順位から 1B での最良を約80%当てられた。正解率より連続の尤度の指標のほうが、少ない計算でよく予測した。
- [Signal and Noise](../sources/signal-and-noise.md) は、正解率より perplexity のほうが信頼できるとしている。
- [Smol Training Playbook の読書ノート](../sources/smol-training-playbook.md)には、小さな規模で悪いものは捨ててよいが、良かったものは十分長く学習してから判断するとある。
- [SmolLM2](../sources/smollm2.md) は、小さなモデルでは混ぜ方を比べ直すと、大きなモデルと違う結果になった。

どれも読む量が桁違いに多いか、英語だけの結果で、日英を混ぜた学習の結果ではない。

[Repetition Mismatch](../sources/repetition-mismatch.md)：質の高いデータを繰り返す場合、小さな実験と本番で繰り返しの回数を揃えないと、混ぜ方の予想が外れる。揃えると16分の1のトークンで当てられた。

[小さなモデル群で予測する研究](../sources/model-ladders.md)は、本番の1%の計算で、7B と 13B の正解率を絶対値で2点以内で当てた。[途中の点の揺れの研究](../sources/checkpoint-score-instability.md)は、近い重みの平均で点が安定するとした。[3500万の MoE の比較](../sources/moe-below-measurement-floor.md)は、課題が当てずっぽうの水準のままでは差を測れないことの例。
