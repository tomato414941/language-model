# 技術の記録

調べた技術と、それをどうしたか。主に v6 のためのもので、v6 が v5 から引き継いだものと、v5 の結果について調べたことも入っている。新しく調べる前にここを見る。

- [value-embeddings.md](value-embeddings.md)：v6 に入れた
- [model-shape.md](model-shape.md)：ヘッド幅128、32層
- [batch-size.md](batch-size.md)：約100万トークン、学習率と weight decay の換算
- [normuon.md](normuon.md)：NorMuon と cautious weight decay
- [residual-scalars.md](residual-scalars.md)：x0/resid lambdas と出力のゼロ初期化
- [lr-schedule.md](lr-schedule.md)：最後の20%で1%まで下げる
- [checkpoint-averaging.md](checkpoint-averaging.md)：最後の10個の重みを平均する
- [tokenizer.md](tokenizer.md)：数字を1桁ずつにする
- [english-web-data.md](english-web-data.md)：DCLM-Edu に替えた。ClimbMix などは見送り
- [data-repetition.md](data-repetition.md)：まだ決めていない
- [gpu.md](gpu.md)：A100 のまま
- [inherited-from-v5.md](inherited-from-v5.md)：SwiGLU、QK-norm、注意のゲート、文書ごとの注意
- [not-adopted.md](not-adopted.md)：見送ったもの、後回しにしたもの
- [v5-checks.md](v5-checks.md)：JCommonsenseQA の低下、SFT の Nemotron v2 日本語
- [japanese-chat-evaluation.md](japanese-chat-evaluation.md)：M-IFEval-Ja、ELYZA-tasks-100、Japanese MT-Bench

## 前提にしていて、調べていないこと

- 根拠のほとんどは英語だけのモデルの結果で、日英半々のモデルにも当てはまるかは調べていない。
- 日英の比率、コードと数学の割合は v5 のままで、調べていない。
- 既存研究の結果を、自分たちのデータで比べて確かめてはいない。既存研究を信用すると決めたのは VE についてで、ほかにも広げたのは私の判断。
- ローカル LLM のコミュニティ（r/LocalLLaMA）は読めず、調べていない。
- 他のモデルからの蒸留、データの中身の選り分け、商用で使えないライセンスについて、どこまでを禁止とするかは決まっていない。
