# Liger Kernel

Liger Kernel: Efficient Triton Kernels for LLM Training（arXiv 2410.10989）。実装は GitHub の linkedin/Liger-Kernel。

演算をまとめたカーネルと入力の分割で、Hugging Face の実装と比べて、学習の速さが平均20%上がり、GPU のメモリが60%減ったとしている。まとめる理由は、高帯域のメモリと共有メモリのあいだの読み書きが律速になるため。

読んだのは要旨と導入まで。
