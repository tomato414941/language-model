# 選好の最適化

SFT のあとで、好ましい答えと好ましくない答えの組から学習する。

[SmolLM3](../sources/smollm3.md) は SFT のあとで APO を行い、数学、指示に従う力、コードなどの評価が上がり、長い文脈の評価は下がった。読んだ範囲に上がり幅の数字はない。このメモに以前あった「IFEval などが15〜20ポイント上がる」「学習率は SFT の約10分の1がよい」という数字は、SmolLM3 の記事と [Smol Training Playbook](../sources/smol-training-playbook.md) の原文で見つからず、出どころを確かめられなかったので外した。

[SmolLM2](../sources/smollm2.md) の小さなモデルも、SFT のあとで DPO を行っている。

[LLM-jp-4](../sources/llm-jp-4.md) は SFT のあとに DPO を行い、作った組を公開している。
