# DeepSeek-V3

DeepSeek-V3 Technical Report（arXiv 2412.19437）。評価の節と結果の一部を読んだ。

事前学習のモデル（5-shot など）と chat のモデルを別々に評価している。chat のモデルは、MMLU、DROP、GPQA、SimpleQA で simple-evals の問いを使い、他社の非公開のモデルはそれぞれの API で評価した。自分たちのモデルに印をどう付けたかは、読んだ範囲には書かれていない。MMLU は chat のモデルで 88.5 と報告している。
