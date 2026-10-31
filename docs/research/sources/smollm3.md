# SmolLM3

Hugging Face の記事 SmolLM3: smol, multilingual, long-context reasoner（huggingface.co/blog/smollm3）。SFT のあとの選好の最適化と、モデルの混ぜ合わせの節を読んだ。

SFT のあとで APO（Anchored Preference Optimization、DPO の変形）を行った。Tülu 3 の組と、Qwen3 で作った思考ありの組を使った。著者らは、DPO より安定し、社内の比較で下流の点も高かったとしている。数学、科学、指示に従う力、コード、会話、多言語の評価が上がり、長い文脈の RULER は下がった。読んだ範囲に上がり幅の数字はない。

RULER の低下は、APO のモデルを混ぜたものと中間の学習のモデルを 0.9 と 0.1 で混ぜて取り戻した。
