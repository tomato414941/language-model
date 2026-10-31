# Marin の nano_arch_ablations

Marin の speedrun の結果（marin.community/speedrun、データは marin-community/speedrun の data/runs.json）。約1.5億〜2.8億パラメータで、前の行に1つずつ足していく比較。数字は Paloma C4 の BPB で、低いほど良い。

| 足したもの | BPB |
|---|---|
| 基準 | 1.3160 |
| QK-norm | 1.3156 |
| ReLU² | 1.3156 |
| resid lambdas | 1.3123 |
| x0 lambdas | 1.3112 |
| value embeddings | 1.3007 |
| value embeddings のゲート | 1.2962 |
| ヘッド幅128（12ヘッドを6ヘッドに） | 1.2838 |
| 半分の RoPE | 1.2827 |
| 注意のゲート | 1.2787 |
| バイグラムのハッシュの表 | 1.2757 |

一番大きく下がったのはヘッド幅128で、次が value embeddings。足す順番を変えた比較ではないので、それぞれの差は前の行までの形での差。
