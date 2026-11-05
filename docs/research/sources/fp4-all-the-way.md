# FP4 All the Way: Fully Quantized Training of LLMs

arXiv 2505.19115。要旨を読んだ。

16個ずつの値が E4M3 のスケールを共有する NVFP4 が最良だった。7B を 200B トークン、Intel Gaudi2 で FP4 で学習し、下流の課題で bf16 と並んだ。勾配のノルムが量子化の雑音の約√3倍を下回ると、量子化した学習は効きにくくなる。
