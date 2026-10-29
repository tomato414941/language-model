# ヘッド幅とモデルの形

v6 はヘッド幅128（query 8、key-value 2）、幅1024、32層。表を除いて約3.9億パラメータで、v5 の計算量での計算最適（約4億 × 84億トークン）に合わせた。

- ヘッド幅128：Marin で BPB −0.0124（比較の中で最大）。nanochat LOG 2026-01-17「Head dim definitely prefers to be 128」。（326f637）
- 深さ：nanochat LOG 2026-01-17 で、同じ計算量なら少し細く長いほうが良い。Smol Training Playbook でも、同じパラメータなら埋め込みを分けるより層を増やすほうが効いた。（f517a19）
