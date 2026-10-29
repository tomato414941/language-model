# 見送ったもの

- Liger のまとめたカーネル：v5 から torch.compile で学習していて、RMSNorm や SwiGLU のような小さな演算はすでにまとまる。残る利点は、出力の層と損失の計算をまとめてメモリを減らすことだけ。速度の測定でメモリが足りなければ考える。
- sliding window（nanochat の SSSL）：v6 は注意を文書の中だけで計算するので、文書が窓（4096トークン）より短ければ計算は減らない。v6 のデータの文書の長さは確かめていない。nanochat の根拠も「Quick experiments」だけ。
- bigram hash embedding：Marin では 1.2787 → 1.2757。nanochat は d25 で改善がごく小さく、時間で比べると消え、メモリも増えたとして取り除いた。
- logit softcap：nanochat は値を5〜30で比べて、20 が最良（検証損失が約0.001良い）とした。softcap を入れない場合とは比べていない。v6 には入っていない。
- 入力と出力の埋め込みを分ける：Smol Training Playbook の読書ノート（jianyuh.github.io）によると、同じパラメータ数なら、埋め込みを分けるより層を増やすほうが効いた。nanochat は分けているが、分けない場合と比べた記録は見つけていない。nanochat の埋め込みの学習率（入力 0.3、出力 0.008）は分けた形が前提なので、これも使わない。
- LayerNorm scaling：IMU-1 では学習損失が −0.36%、nanochat の LOG では「Did not help」。
- partial RoPE：Marin では 1.2838 → 1.2827、nanochat では4分の1にかけて「Slightly worse」。
- 自分のモデルによるデータの書き直し：自分のモデルならルール上は許される。ただ、3億の v5 で書き直して良くなったという報告は見つけていない。
- 日本語のコーパスの入れ替え（LLM-jp-corpus v4、FineWeb2 など）：日本語のコーパスどうしを比べた数字を見つけていない。
- 1B 未満への強化学習（RLVR）：検索で見つかった報告は 1.5B 以上ばかりだった。

## 後回しにしたもの

- 選好の最適化：同じ読書ノートによると、SmolLM3 では APO-zero で IFEval などが15〜20ポイント上がり、学習率は SFT の約10分の1がよかった。事前学習の後のことなので、v6 の SFT の後に考える。
