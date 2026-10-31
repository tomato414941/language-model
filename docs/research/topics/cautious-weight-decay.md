# Cautious weight decay

更新がその重みを0に近づける向きのときだけ、weight decay をかける（arXiv 2510.12402）。nanochat は、強さを学習を通して cos 形で0まで下げている。既定の強さは 0.28。

IMU-1 の 7000万パラメータの比較では、NorMuon に足して学習損失が 3.156 から 3.125 になった（AdamW からは合わせて −3.85%）。
