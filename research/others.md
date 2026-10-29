# 見送ったもの・後回しにしたもの

- Liger：v5 はすでに torch.compile で小さな演算をまとめている。残る利点は、出力の層と損失の計算をまとめてメモリを減らすことだけ。メモリが足りなくなったら考える。
- Sliding window：v6 は注意を文書の中だけで計算するので、窓を短くしても計算はあまり減らない。nanochat の根拠も短い実験だけ。
- Bigram hash embedding：Marin では −0.0030。nanochat は d25 で効果が消えたとして取り除いた。
- Logit softcap：nanochat が比べたのは softcap の値どうし（20が最良で約0.001）。入れない場合との比較がない。
- 入力と出力の埋め込みを分ける：Smol Training Playbook では層を増やすほうが効いた。nanochat の埋め込みの学習率（入力0.3、出力0.008）も分けた形が前提なので使わない。
- LayerNorm scaling：IMU-1 では −0.36%、nanochat では効かなかった。
- Partial RoPE：Marin で −0.0011、nanochat では少し悪化。
- Z-loss：安定化のためのもの。v5 は安定していた。
- Muon の細かい改良（MuonEq、Muon+ など）：nanochat 自身が「very slight and optional」としている。
- 自分のモデルでのデータの書き直し：BeyondWeb では書き直すモデルが1B→3Bで質が上がった。v5（3億）では足りない見込み。
- 日本語のコーパスの入れ替え：LLM-jp-4 にもコーパスごとの比較の数字がない。
- 1B 未満の強化学習：見つかった報告は1.5B以上ばかり（論文本文は読んでいない）。
- 選好の最適化（APO-zero など）：Smol Training Playbook で IFEval が15〜20ポイント上がった。v6 の SFT の後に考える。
