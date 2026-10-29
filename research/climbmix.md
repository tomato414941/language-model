# ClimbMix

Nemotron-CC と SmolLM のコーパスを20のグループに分け、小さなモデルでの成績が上がる混ぜ方を探した4000億トークン（CLIMB、arXiv 2504.13161）。

- nanochat の dev/LOG.md（commit 92d63d4e） 2026-03-04: FineWeb-Edu から替えて、GPT-2 と同じ性能に届く時間が27%短くなった。それまでの5つのデータはすべて FineWeb-Edu に勝てなかった。
- CLIMB: 混ぜ方は PIQA、ARC-Easy、HellaSwag の検証用の問題で選んだ。探索に使っていないタスクでも、混ぜ方を探す他の方法より上だった。データどうしをタスク別に比べた数字は本文にない。
- nvidia/Nemotron-ClimbMix のライセンスは CC BY-NC 4.0。karpathy の写しには MIT と表示されているが、元が非商用。

2026-10-29 不採用。今のソースはすべて商用でも使えるライセンス。

見直すとき：非商用でもよいとユーザーが決めたとき。
