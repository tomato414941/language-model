# Japanese MT-Bench

Stability AI の FastChat にある、日本語版の MT-Bench。採点役のモデルが1〜10点をつける。日本語の採点の指示文は1ターン目のぶんで、数学・推論・コードには GPT-4 の参考解答を渡す。公開元は GPT-4 で採点した。Swallow の評価は、採点役に openai/gpt-4o-2024-08-06 を使っている。採点役が違うと、点は比べられない。
