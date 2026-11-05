# MoE

MLP を複数の専門家（expert）に分け、トークンごとに一部だけを動かす。計算を増やさずにパラメータを増やせる。v7 で採用すると決めた（2026-11-05）。

## 他の人の結果

- **規模と効率：** [Ling の効率の規模則](../sources/moe-efficiency-leverage.md)は、試した全ての大きさで MoE が計算あたり効率的で、使う割合が小さいほど、計算が大きいほど得になるとした。細かさは12前後が最良で、共有の専門家は 1e20 FLOPs 以下で 16.7%。[細かい MoE の規模則](../sources/fine-grained-moe-scaling.md)も、差は規模とともに広がるとした。[同じ資源での比較](../sources/moe-equal-resources.md)は、2B と 7B で、全体・計算・データをそろえても、使う割合が最適な範囲なら MoE が勝つとした。[DeepSeekMoE](../sources/deepseekmoe.md) は 2B で、1.5倍の計算の GShard に並んだ。[OLMoE](../sources/olmoe.md) は 7B のうち 1B を使い、router の z-loss を使う。
- **小さな規模：** [2500万以下の比較](../sources/tiny-moe.md)では、使う大きさをそろえると MoE が勝ち、全体の大きさをそろえると負けた。[3500万の比較](../sources/moe-below-measurement-floor.md)は、1000万トークンでは課題が当てずっぽうの水準で、差を測れなかった。
- **時間：** [nanochat の LOG](../sources/nanochat-log.md)（2026-02-19）は、ステップあたりの検証損失は良くなったが、d18 で MFU が約46%から約35%に落ち、時間で負けとした。課題の点では比べていない。
- **知識と推論：** [Mixture of Parrots](../sources/mixture-of-parrots.md) と[推論と疎さの研究](../sources/moe-sparsity-reasoning.md)は、専門家を増やすと知識と記憶は上がり、推論は使う計算とトークン数で決まるとした。
- **負荷の均し方：** [補助の損失を使わない均し方](../sources/loss-free-balancing.md)（DeepSeek-V3 の形）、[更新全体で均す損失](../sources/global-batch-balancing.md)（専門家が分野ごとに分かれ、点も上がる）。
- **密なモデルから作る：** [Drop-Upcycling](../sources/drop-upcycling.md)、[作り替えの規模則](../sources/upcycling-scaling-laws.md)。
- **専門家の FP8：** [PyTorch の MXFP8](../sources/mxfp8-moe-torchao.md) は、データセンター向けの Blackwell で学習全体が1.3倍。RTX 5090 では使えない。

## 私たちの測定

[v7 の速さ](../experiments/v7-throughput.md)。RTX 5090 で、1トークンに使う大きさを v6 の形と同じにした MoE（専門家 1024×12 から3つと共有の専門家1つ、全体約11億）は、1窓ずつで毎秒 38,200 トークン、メモリ 27.5GiB。同じ機械の普通のモデル（1窓ずつ）より34%遅く、GPU が計算していたのは時間の51%だった。専門家の数え方の同期をなくしても変わらなかった。512×24 と 256×48 は 32GB に収まらず、専門家を8つにしても2窓ずつは収まらなかった。

DeepSeek-V3 の偏りで均すと、120回の更新のうちに全トークンが同じ専門家を選ぶ状態になった。手元の小さな MoE で、全トークンが同じ専門家を好む状態から始めると、偏りは働くが、全トークンがそろって次の専門家に移るだけで均等にならなかった。トークンごとの好みが十分ばらけていない学習の初めには、偏りだけでは均せない。

## コード

`pretrain/model.py` の `Experts`：OLMoE の形に、二乗 ReLU の専門家、共有の専門家、DeepSeek-V3 の偏りによる均し方（更新全体で数える）を足した。Hyperball は専門家の行列にも使える。Hugging Face 形式への書き出しにも対応した。
