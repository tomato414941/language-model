# 小さな実験で決める

本番より小さなモデルと少ないトークンで比べて、データや設定を決める。

- [DataDecide](../sources/datadecide.md) では、150M の順位から 1B での最良を約80%当てられた。正解率より連続の尤度の指標のほうが、少ない計算でよく予測した。
- [Signal and Noise](../sources/signal-and-noise.md) は、正解率より perplexity のほうが信頼できるとしている。
- [Smol Training Playbook の読書ノート](../sources/smol-training-playbook.md)には、小さな規模で悪いものは捨ててよいが、良かったものは十分長く学習してから判断するとある。
- [SmolLM2](../sources/smollm2.md) は、小さなモデルでは混ぜ方を比べ直すと、大きなモデルと違う結果になった。

どれも読む量が桁違いに多いか、英語だけの結果で、日英を混ぜた学習の結果ではない。
