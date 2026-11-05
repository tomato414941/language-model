# MXFP8 Training for MoEs（PyTorch ブログ、2026-03-12）

全文を読んだ。

TorchAO と TorchTitan で、Llama4 Scout の専門家の計算を MXFP8 にし、GB200 の 256 台で bf16 より 30.2% 速く、3,000 ステップの損失は bf16 と同じだった。MXFP8 の専門家の行列計算は、コンパイルした bf16 の約1.8倍、MoE の層は1.43倍。出力の射影、router、小さな Wk と Wv は bf16 のまま。データセンター向けの Blackwell（tcgen05）の命令を使う。
