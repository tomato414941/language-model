# Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics

arXiv 2601.21698。要旨だけを読んだ。

14M から 1B のモデルを 300B トークン学習し、Age-of-Acquisition、語の頻度、Verb Variation の順番をランダムな順番と比べた。順番は学習の段階を新しく作らず、各段階にいる時間を変える。14M〜70M では、ランダムな順番のほうが勾配の雑音が大きかった。大きいモデルでは、この差は小さい。Verb Variation を逆順にすると、正解率の利点の多くを失った。
