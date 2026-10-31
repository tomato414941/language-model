# LLM-jp-4 の SFT のデータ

llm-jp/llm-jp-4-thinking-sft-data。複数のデータの問いに、gpt-oss-120b が思考と答えを書いたもの。思考の長さで `reasoning_low`、`reasoning_medium`、`reasoning_high` に分かれる。ライセンスは元のデータごとに違う。[LLM-jp-4](../sources/llm-jp-4.md) が SFT に使った。DPO の組も llm-jp/llm-jp-4-8b-thinking-dpo-data などで公開されている。

## jaster_v1.4.1

llm-jp-eval v1.4.1 の課題の形をした日本語の問いで、`reasoning_low` だけで248,357行ある。2026-10-31 に480行を読むと、選択肢や番号で答えさせる問いが121行あり、最後の答えは171行が記号や数字1文字で、中央値は5文字だった。読んだ範囲の思考は英語だった。評価の課題の学習用の問題を含むはずなので、使うなら評価の問題との重なりを確かめる。
