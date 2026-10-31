# Value Residual Learning

Value Residual Learning（arXiv 2410.17897）。Zhou ら。v1 の題は Value Residual Learning For Alleviating Attention Concentration In Transformers。

ResFormer は、普通の残差に加えて、最初の層の value と今の層の value のあいだに残差を足す。

- 82M、180M、320M、468M を 20B トークンで学習し、同じ検証損失に要るパラメータが16.11%少なく、データが20.3%少ない。メモリと計算はほぼ同じ。
- 下流の課題の平均は、学習する形で1.7ポイント上がった。
- 効くのは最初の層の value からの接続だけで、2層目やあとの層からの接続は効かないか、悪くする。
- 8層のモデルでは、接続は後ろの層（6〜8層）に足すのが一番効いた。8層と24層で学習させた重みでも、後ろの層ほど大きくなった。

SVFormer は、すべての層で最初の層の value を共有して、KV キャッシュをほぼ半分にする。小さな性能の低下と引き換えになる。
