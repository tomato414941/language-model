# 深さ（32層）

v6 は幅1024のまま、層を24から32にした（f517a19）。ユーザーの「少し大きくしたい」を、幅ではなく層で実現した。

nanochat の LOG（2026-01-17）に、計算量を 1e18 に固定した比較で「Aspect ratio of 128 is worse than 64 … The LLM prefers to be slightly thinner and longer」とある。Smol Training Playbook の読書ノートにも、同じパラメータ数なら、埋め込みを分けるより層を増やすほうが効いたとある。
