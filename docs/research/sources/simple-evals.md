# simple-evals

[openai/simple-evals](https://github.com/openai/simple-evals)。公開する数字を説明するために OpenAI が出したコード。`common.py`、`mmlu_eval.py`、Chat Completions の sampler を読んだ。2025年7月から、新しいモデルでは更新していない。

MMLU の問題を user のメッセージとして渡す。問いには「最後の行を Answer: $LETTER の形で書く」とあり、返答からこの行を読み取って採点する。
