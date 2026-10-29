# ClimbMix

見送った。

Nemotron-CC と smollm-corpus をグループに分け、小さなモデルでの成績が上がるように混ぜ方を探して作った4000億トークン（CLIMB、arXiv 2504.13161）。混ぜ方は PIQA、ARC-Easy、HellaSwag の検証用の問題での成績で選ばれている。テストの問題が入っているわけではないが、その種類の問題に向いた文章が多めになる。CLIMB は、ほかのタスクにも効果が及んだとしている。

nanochat の LOG（2026-03-04）では、FineWeb-Edu から ClimbMix に替えて、GPT-2 の性能に届くまでの時間が2時間46分から2時間1分になった。nanochat がそれまでに試した5つのデータは、どれも FineWeb-Edu に勝てなかった（fineweb-edu.md）。

nvidia/Nemotron-ClimbMix のライセンスは CC BY-NC 4.0（非商用）。nanochat が使う karpathy/climbmix-400b-shuffle には MIT とあるが、元の制限が外れるわけではない。
