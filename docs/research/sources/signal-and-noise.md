# Signal and Noise

Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation（arXiv 2508.13144）。要旨を読んだ。

評価が、良いモデルと悪いモデルを分ける力を信号、学習の1ステップごとのばらつきを雑音として測る。信号と雑音の比が良い評価ほど、小さな実験での判断が信頼できる。正解率より perplexity を使うこと、雑音の多い下位の課題を除くこと、途中のチェックポイントの出力を平均することが効くとしている。
