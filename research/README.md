# 技術の記録

調べた技術と、それをどうしたか。新しい技術を調べる前に、まずここを見る。

- [ヘッド幅128](head-dim-128.md)：v6で採用。Marin と nanochat の両方で損失が下がった
- [Value embeddings](value-embeddings.md)：v6で採用。1層おき・ゲート付きの nanochat の形。計算をほぼ増やさずに容量を足す
- [モデルの大きさと深さ（32層）](depth-and-size.md)：v6で採用。同じ計算量なら少し細く長いほうが良い。計算最適の約4億に合わせた
- [バッチの大きさ（約100万トークン）と学習率・weight decay の換算](batch-size.md)：v6で採用。Power Lines の法則による nanochat の自動バッチ。学習率は平方根で換算
- [NorMuon](normuon.md)：v6で採用。Muon の更新の大きさをニューロンごとにそろえる
- [Cautious weight decay と weight decay の cos 形の下げ方](cautious-weight-decay.md)：v6で採用。更新がゼロへ向かう重みだけを減衰させ、減衰は学習の終わりに0へ
- [x0 / resid lambdas](residual-scalars.md)：v6で採用。層ごとに、流れを縮めたり最初の入力を混ぜ直したりする数を学習する
- [出力の重みのゼロ初期化](zero-output-init.md)：v6で採用。各層が流れに書き戻す重みを0から始め、層が恒等写像から始まる
- [DCLM-Edu](dclm-edu.md)：v6で採用。DCLM を FineWeb-Edu の判定器で選び直したデータ。v6 と同じ大きさのモデルで比べた結果がある
- [学習率の下げ方（20%で、ほぼ0まで）](lr-decay-schedule.md)：v6で採用（未実装）。下げる期間は20%が最良で、最後は0近くまで下げるほうが良い
- [最後のチェックポイントの平均](checkpoint-averaging.md)：v6で採用（未実装）。学習の後で重みを平均するだけ。学習の費用は増えない
- [数字を1桁ずつトークンにする](single-digit-tokens.md)：v6で採用（未実装）。LLM-jp-4 で数学が良くなり、nanochat でも3桁ずつまとめるのが最も悪かった
- [質の高いデータを何回か繰り返して読む](data-repetition.md)：v6で保留。質の高いデータの繰り返しは、量の多い1回読みに勝つという報告が多い
- [選好の最適化（APO-zero など）](preference-optimization.md)：v6 の後で保留。SmolLM3 では IFEval が大きく上がった。事前学習の後の段階
- [Liger のまとめたカーネル](liger-kernels.md)：v6で不採用。torch.compile がすでに小さな演算をまとめている
- [Sliding window attention](sliding-window.md)：v6で不採用。注意は文書の中だけで計算しているので、窓を短くしても計算はあまり減らない
- [Bigram hash embedding](bigram-hash-embedding.md)：v6で不採用。nanochat が大きなモデルでは効果が消えたとして取り除いた
- [Logit softcap](logit-softcap.md)：v6で不採用。softcap どうしの比較しかなく、入れない場合との比較がない
- [入力と出力の埋め込みを分ける](untied-embeddings.md)：v6で不採用。同じパラメータなら、層を増やすほうが効く
- [埋め込みだけの学習率](embedding-learning-rate.md)：v6で不採用。nanochat の値は埋め込みを分ける形が前提
- [LayerNorm scaling](layernorm-scaling.md)：v6で不採用。IMU-1 では効いたが nanochat では効かなかった
- [Z-loss](z-loss.md)：v6で不採用。学習を安定させるためのもの。v5 は安定していた
- [Partial RoPE](partial-rope.md)：v6で不採用。Marin でわずかに良く、nanochat では悪かった
- [Muon の細かい改良（MuonEq、Muon+、Polar Express、momentum の変化）](muon-variants.md)：v6で保留（低優先）。nanochat 自身が「わずか」と書いている
- [ClimbMix](climbmix.md)：v6で不採用。非商用のライセンス（CC BY-NC 4.0）
- [Ultra-FineWeb](ultra-fineweb.md)：v6で不採用。作った人たち自身の比較しかない
- [Nemotron-CC v2 / v2.1](nemotron-cc.md)：v6で不採用。他のモデル（Qwen、DeepSeek）で書き直した合成データを含む
- [自分のモデルによるデータの書き直し](self-rephrasing.md)：v6で不採用。書き直すモデルは1B以上が必要という報告。v5（3億）では力が足りない
- [語彙を大きくする](larger-vocabulary.md)：v6で不採用。日本語では大きいほど良いが、v6 では埋め込みが大きくなりすぎる
- [日本語のコーパスを替える（LLM-jp-corpus v4、FineWeb2 など）](japanese-corpus.md)：v6で不採用。コーパスごとに比べた数字が見つからない
- [1B 未満のモデルへの強化学習（RLVR、GRPO）](rlvr-small.md)：v6で不採用。効果の報告は1.5B以上ばかり
- [H100 で学習する](h100.md)：v6で不採用。今の料金では、A100 より費用あたりで不利
- [中断されうる GPU（interruptible）](interruptible-gpu.md)：v5, v6で不採用。使い物にならなかった

## 出典

- nanochat: https://github.com/karpathy/nanochat （dev/LOG.md、scripts/base_train.py、nanochat/optim.py、commit 92d63d4e）
- Marin speedrun: https://github.com/marin-community/marin （experiments/speedrun）
- IMU-1: https://arxiv.org/abs/2602.02522
- Straight to Zero: https://arxiv.org/abs/2502.15938
- LAWA: https://arxiv.org/abs/2306.03241
- LLM-jp-4: https://arxiv.org/abs/2610.19331
- CLIMB: https://arxiv.org/abs/2504.13161
- Smol Training Playbook: https://huggingfacetb-smol-playbook-toc.static.hf.space/
- DCLM-Edu: https://huggingface.co/datasets/HuggingFaceTB/dclm-edu
- Ultra-FineWeb: https://huggingface.co/datasets/openbmb/Ultra-FineWeb
- Fang ら（データの繰り返し）: https://arxiv.org/abs/2503.07879
- MiniMax-01: https://arxiv.org/abs/2501.08313
- Large Vocabulary Size Improves LLMs: https://arxiv.org/abs/2406.16508
- BeyondWeb: https://www.datologyai.com/blog/beyondweb
