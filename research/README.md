# 技術の記録

調べた技術と、どうしたか。調べる前にここを見る。

- [value-embeddings.md](value-embeddings.md)：v6 で採用
- [shape.md](shape.md)：ヘッド幅128、32層（v6）
- [batch-size.md](batch-size.md)：約100万トークン、学習率と weight decay の換算（v6）
- [normuon.md](normuon.md)：NorMuon、cautious weight decay（v6）
- [residual-scalars.md](residual-scalars.md)：x0/resid lambdas、出力のゼロ初期化（v6）
- [lr-schedule.md](lr-schedule.md)：20%で1%まで下げる（v6、未実装）
- [checkpoint-averaging.md](checkpoint-averaging.md)：v6 で採用（未実装）
- [tokenizer.md](tokenizer.md)：数字を1桁ずつ（v6、未実装）。語彙は増やさない
- [english-web-data.md](english-web-data.md)：DCLM-Edu を採用。ClimbMix ほかは見送り
- [data-repetition.md](data-repetition.md)：保留
- [gpu.md](gpu.md)：A100 のまま
- [others.md](others.md)：見送ったもの・後回しにしたもの
