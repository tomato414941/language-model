# Liger のまとめたカーネル

RMSNorm、SwiGLU、出力の層と損失の計算などを、それぞれ1つの Triton のカーネルにまとめる。torch.compile でも、RMSNorm や SwiGLU のような小さな演算はまとまる。torch.compile と比べたときの利点は、主に、出力の層と損失の計算をまとめてメモリを減らすこと。
