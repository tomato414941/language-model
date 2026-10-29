# Logit softcap

出力のロジットを `c·tanh(x/c)` で抑える。

- nanochat の dev/LOG.md（commit 92d63d4e） 2026-03-02: c を5〜30で比べ、20が最良で約0.001良かった。比べたのは softcap どうし。

2026-10-29 不採用。

見直すとき：softcap あり・なしの比較が見つかったとき。
