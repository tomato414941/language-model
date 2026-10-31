# FineInstructions

FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale（arXiv 2601.22146）。要旨、作り方の説明、結果の表の一部を読んだ。

実際の質問から作った約1800万の指示のひな形から、文書に合うものを探し、小さなモデル（Llama 3.2 3B を調整したもの）で指示を具体化して、文書の抜粋を答えにする。答えの多くは人が書いた文章になる。

1.8B のモデルで、同じ文書から同じトークン数を読んで比べた。著者らによると、Nemotron-CC の文書では MixEval が通常の事前学習より約39%高く、書き直しや問答を作る他の方法より高かった。AlpacaEval でも好まれた。MT-Bench-101 だけは傾向が違うという記述があるが、その先は読んでいない。評価は英語だけ。
