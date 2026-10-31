# MoE

MLP を複数の専門家（expert）に分け、トークンごとに一部だけを動かす。

[nanochat の LOG](../sources/nanochat-log.md)（2026-02-19）では、ステップあたりの検証損失は良くなったが、d18 で MFU が約46%から約35%に落ち、時間で比べると負けた。

[OLMoE](../sources/olmoe.md) は 7B のうち入力ごとに 1B を使う形で、router の z-loss を使っている。
