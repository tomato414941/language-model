# Sliding window attention

一部の層の注意を、文脈の半分の窓に限る（nanochat の SSSL）。

- nanochat の dev/LOG.md（commit 92d63d4e） 2026-01-11: 「Quick experiments showed SSSL works well」。短い実験だけ。
- v6 は文書ごとに注意を区切っていて（varlen）、ほとんどの文書は窓より短い。

2026-10-29 不採用。

見直すとき：文脈を大きく伸ばすとき（長い文脈の段階）。
