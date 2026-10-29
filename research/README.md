# 技術の記録

v6 のために調べた技術と、それをどうしたか。新しく調べる前にここを見る。v5 までの判断は入っていない。

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
- [not-adopted.md](not-adopted.md)：見送ったもの、後回しにしたもの
