# Scaling FP8 Training to Trillion-Token LLMs

arXiv 2409.12517。要旨を読んだ。

2T トークンまでの長い FP8 の学習で、短い学習では出ない不安定さが現れた。SwiGLU が外れ値を増幅するためで、Smooth-SwiGLU で直した。Adam の2つの状態も FP8 にし、7B で bf16 と並び、処理量が最大約34%上がった。
