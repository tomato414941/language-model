# 英語の Web データ

v6 の安定期では、DCLM を DCLM-Edu に替えた（026e91d）。FineWeb-Edu はそのまま。

- DCLM-Edu（CC-BY-4.0）：DCLM を FineWeb-Edu の判定器で選び直したもの。SmolLM2 チームの 360M・2000億トークンの比較で、DCLM と FineWeb-Edu より MMLU と ARC が良かった。1.7B の途中から入れた比較では安定しなかった。
- ClimbMix：nanochat では FineWeb-Edu から替えて GPT-2 に届く時間が27%縮んだ。ただしライセンスが CC BY-NC 4.0（karpathy の写しの MIT 表示は元の制限を外さない）。今のソースはすべて商用で使えるので見送った。混ぜ方を PIQA・ARC-Easy・HellaSwag の検証用の問題で選んでいる点も、評価のときは気にする。
- Ultra-FineWeb（Apache-2.0）：作った人たち自身の比較しかないので見送った。
- Nemotron-CC v2/v2.1：データのカードに、学習したモデルに Qwen・DeepSeek・Phi-4 のライセンス条件がかかりうるとある。他のモデルが書いた文章を含むと読み、見送った。
- nanochat では、FineWeb-Edu に DCLM などを混ぜたもの（finepdfs/dclm/fineweb_edu）が FineWeb-Edu だけより少し悪かった。
