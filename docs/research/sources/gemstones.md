# Gemstones

Gemstones: A Model Suite for Multi-Faceted Scaling Laws（arXiv 2502.06857）。McLeish ら。コードは GitHub の mcleish7/gemstone-scaling-laws。

Gemma を小さくした形で、5000万、1億、5億、10億、20億のパラメータのそれぞれで幅と層を変え、4000以上のチェックポイントを公開している。スケーリング則の結論は、実験の設計と当てはめに使うチェックポイントに強く左右されるとしている。

形については次のとおり。

- 計算量（FLOPs）で測ると、深いモデルのほうが損失もベンチマークの誤差も低い。検証のデータを変えても、モデルの順番は変わらなかった。
- GPU の時間で測ると、広いモデルのほうが早く損失が下がる。一般的なオープンソースの並列の仕組みでは、深いモデルは時間が大きくかかる。

ベンチマークは MMLU、WinoGrande、OpenBook QA、ARC、CommonSense QA、PIQA、SIQA、HellaSwag。

CompleteP（arXiv 2505.01618）はこの研究を「浅い形が最適と結論した」と書いているが、読んだ範囲の本文には、計算量で測ると深いほうが良いとある。

読んだのは要旨、導入、形についての節まで。
