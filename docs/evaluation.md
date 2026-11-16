# Evaluation

A generation is judged by two definitions of language-model-evaluation, the language-model projects' shared evaluation: one measures what a model knows, the other how it follows instructions. A definition fixes what is measured and how two results compare. It is not changed to suit a model's results; a change makes a new definition with a new name, and the generations to be compared are measured again by it. A way of measuring that is confirmed to be wrong is dropped, even if results then no longer compare with the past.

## What a model knows

The model after pretraining is measured in plain text by knowledge-ja-en-part-1 (`python -m pretrain knowledge`). The artifacts that the definition names are made by language-model-evaluation and kept by their names in `data/evaluation/`.

- The definition is a part of knowledge-ja-en-1, which holds OLMES's English questions, Japanese questions asked the same way, and English and Japanese text. A model is measured by the bits per byte of each question's correct answer and of each document, which move with what it knows rather than with how it answers. Two models compare by the mean of the two languages' mean rates of improvement over the definition's categories.
- The part holds at most 3,000 questions of each task, drawn at random, and 1 MB of each source of text, spread evenly over its domains: one rule for every task and source. It reads about a fifth of the whole's tokens, 50 million a model. From the measurements of v5 to v7, its comparisons' standard errors come to an estimated 1.08 to 1.10 times the whole's.

## How a model follows instructions

The model after instruction tuning is measured by instructions-ja-en-1 (`python -m pretrain instructions`).

- IFEval's 541 English prompts and M-IFEval's 172 Japanese prompts, scored by M-IFEval's code at a fixed commit. A language's score is the share of prompts whose every instruction the reply follows (strict). Two models compare by the mean of the two languages' rates of improvement.
- Each prompt is the user's message of a conversation in the model's format, and the model replies greedily up to 2,048 tokens.
- The Japanese prompts are few: from v5 to v7, a change in Japanese alone was readable only beyond about a third of the base's score.

## Comparing two results

`python -m pretrain improvement --base A --other B` gives the rate of improvement, overall and by category, with standard errors from drawing the questions, documents and prompts again. A difference is read beyond two standard errors; between two models that differ in nothing, about one comparison in twenty goes beyond it. `--chance` compares an instruction result with the score its replies get when moved to other prompts.

## Pass line

A generation passes when, against the base,

- its model after pretraining improves by knowledge-ja-en-part-1 by more than two standard errors, and
- its model after instruction tuning does not fall by instructions-ja-en-1 by more than two standard errors.

Each is the overall improvement of the comparison. The base is v7; a generation that passes becomes the base. The model after instruction tuning is also measured by knowledge-ja-en-part-1, to see what tuning keeps of what the model knows, but that result does not judge. This pass line was set on 2026-11-16, before the next generation's results.

## Training data

The questions, prompts and texts of the evaluation are removed from the training data. The recipe leaves out every document and conversation that holds a question or a stretch of a text of knowledge-ja-en-1, the whole that the part is drawn from, as its overlap artifacts tell them, or a prompt of instructions-ja-en-1 (`pretrain/overlap.py`): from the corpora, the tokenizer's sample, and the instruction and preference data. v5 to v7 were trained before these definitions, and their data were checked only against the questions of the evaluation they had then.

## Cautions when reading results

- Do not compare the training logs' validation across generations. Each generation reads 64 random windows, and in the v5 and v6 decay phases at most 31% of the Japanese text and 16% of the English text is shared. Even for the same final v5 weights, the 64 windows and the whole validation set gave .553 and .524 for Japanese and .66 and .639 for English. Trends within one generation can be compared, since they use the same windows.
