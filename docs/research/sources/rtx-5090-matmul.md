# RTX 5090 Matmul Performance

main-horse の記事（2025-04-12）。全文を読んだ。

- 行列計算のピークの実測：bf16 は乱数の入力で約187 TFLOPS、mxf8 は 737、nvf4 と mxf4 は 1474。
- PyTorch の実測：bf16 の `torch.matmul` は 183〜228、FP8 の `torch._scaled_mm` は 400〜450 TFLOPS。
- 当時の sm120 では、cuBLASLt のブロックごとのスケールの行列計算が `CUBLAS_STATUS_NOT_SUPPORTED` になった。torchao の MX は、cuBLAS の経路が壊れ、CUTLASS の経路は sm100 に固定されていた。テンソルごとと行ごとのスケールの `_scaled_mm` は動いた。
