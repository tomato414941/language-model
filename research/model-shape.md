# ヘッド幅と層の数

v6 はヘッド幅を64から128にした（query 8、key-value 2。326f637）。幅1024はそのままで、層を24から32にした（f517a19）。

ヘッド幅は、Marin の nano_arch_ablations で 1.2962 → 1.2838 と、並んだ変更の中で一番大きく下がった（12ヘッドを6ヘッドに）。nanochat の LOG（2026-01-17）にも「Head dim definitely prefers to be 128 instead of 64」とある。

深さは、同じ LOG の「Aspect ratio of 128 is worse than 64 … The LLM prefers to be slightly thinner and longer」（計算量 1e18 で固定した比較）による。ユーザーの「少し大きくしたい」を、幅ではなく層で実現した。

大きさは、v5 の計算量（6 × 3億 × 117億 ≈ 2.1×10^19）を1パラメータ約20トークンで分けた、約4億パラメータ × 約84億トークンに合わせた。v6 は VE の表を除いて約3.9億、表を含めて約5.2億。nanochat は VE で最適な比が下がった（8から4）としているが、うちのデータでも同じかは分からないので使っていない。
