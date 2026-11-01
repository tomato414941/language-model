# Tülu 3

Tülu 3: Pushing Frontiers in Open Language Model Post-Training（arXiv 2411.15124 v3）。8B と 70B の Llama 3.1 に、SFT、DPO、RLVR の順で指示学習をした。SFT のデータ、指示に従う力、評価の表の部分を読んだ。全体は読んでいない。

- 25種類の制約を IFEval から取り、GPT-4o で29,980組の指示と答えを作った（If-Persona-Sft）。`personahub_ifdata_manual_seed_v3_29980` がこのデータ。
- 8B の SFT で、Persona のデータ（指示に従うもの、数学、コード）を外すと、IFEval（prompt loose）が 72.8 から 53.6 に下がった。指示に従うものだけを外した比較ではない。
- 8B の IFEval は、SFT 72.8、DPO 81.1、RLVR の後 82.4 と上がった。
- IFEval と別の種類の制約を集めた IFEval-OOD では、SFT 17.6、DPO 23.9、最終 24.3 だった。著者は、モデルが IFEval の制約に過学習していて、新しい制約に広がりにくいと見ている。
