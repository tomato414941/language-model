# Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

arXiv 2501.11873。要旨を読んだ。

負荷を均す損失は、ふつう少数の系列しか持たない小さな単位ごとに計算され、系列の中で専門家を均等に使わせてしまう。更新全体（グローバルバッチ）で計算すると、専門家が分野ごとに分かれ、事前学習の perplexity と下流の課題が良くなった。42.8B・400B トークンまで。
