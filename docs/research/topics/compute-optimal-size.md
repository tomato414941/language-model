# パラメータ数と学習トークン数の配分

決まった計算量を、パラメータ数と学習トークン数にどう分けるか。[Chinchilla](../sources/chinchilla.md) の目安は、1パラメータ約20トークン。

[nanochat の LOG](../sources/nanochat-log.md)（2026-01-17）は、value embeddings でパラメータが増えたので、最適なトークン数とパラメータの比が8から4に下がったとしている（Chinchilla の20よりずっと低い）。

[Kaplan ら](../sources/scaling-laws.md)は、形の影響はとても小さいとしている。
