# Evaluation

## Rules

- Run each test in the form its provider defines. The provider is the test's author when the author defines an evaluation procedure, and otherwise a widely used public tool. Do not change a test's form to suit a model's performance. When scores are too low to separate models, add an easier test that someone else provides.
- A way of measuring that is confirmed to be wrong is dropped, even if results then can no longer be compared with the past.
- Evaluate every question.
- The tools are lm-evaluation-harness and llm-jp-eval v2.1.5. The tasks are in the `evaluation` table of each generation's recipe in `configs/`. Each task is checked against its provider's definition in `research/experiments/test-forms.md`. RULER is also measured with lm-evaluation-harness's version (see "Long context (RULER)" below).
- Remove evaluation questions from the training data.

## How models are asked

- Models after pretraining are asked in plain text.
- Models after instruction tuning are asked in the conversation format, because lm-evaluation-harness's documentation says to use the chat format for chat and instruction-tuned models. Only BLiMP and LAMBADA, which have no question, are asked in plain text. Plain-text scores are also measured but not used for pass or fail.
- Comparisons are made only between results asked the same way.
- MMLU is asked in its authors' form (five examples, answered with the choice's letter). The continuation form (OLMES) is measured separately as an easier test.
- M-IFEval-Ja is scored by its official scoring code on the saved replies (`python -m pretrain score-m-ifeval`). llm-jp-eval's default score (up to the first blank line) is not used.
- From v7, tasks in lm-evaluation-harness that have the model write an answer also keep each question's written answer in the record. Scoring does not change.

## Long context (RULER)

Added on 2026-11-05, so that long-context ability earns some score too.

- **Measured with lm-evaluation-harness's RULER:** the project's own model is asked, as for the other harness tasks (`python -m pretrain ruler`). Switched from the authors' procedure on 2026-11-05, because that procedure runs models on vLLM or similar engines and cannot measure models with parts that read the preceding tokens, such as COUNT.
  - The harness writes RULER's 13 tasks with the model's tokenizer, 500 questions per task at each length, and scores them by RULER's string matching.
  - Answers are written greedily, up to the harness's per-task limit (30 to 128 tokens).
- **Lengths:** 8,192 and 16,384 tokens for now. A question longer than the model's context is cut from the front, as the harness's rules do.
  - 4,096 was left out. In the authors' code, a HotpotQA question whose paragraphs alone exceed the length is never finished, and with this project's tokenizer 4,096 hit that case.
  - The lengths will not necessarily stay the same. v7 is measured at these two, and v8 is measured at the same two for comparison. They change only when a later generation's pass line is decided. They are also reviewed when a model's context exceeds 16,384.
- **Score:** the 13 tasks are averaged at each length, and the lengths are then averaged. The record keeps each question's score for every task.
- **Models asked:** models after instruction tuning, in the conversation format. Each task's answer prefix (such as "The special magic number is") is placed at the start of the reply, as the harness defines, and the model continues it.
- **Pass line:** measured from v7, but reference only in v7, because neither v5 nor v6 has a score to compare with, and they are not measured again. From v8 it enters the pass line as one task in the English mean, compared with v7. The judgment passes `--ruler-base` and `--ruler-other` to `python -m pretrain pass-line`. The 0.451 that v5's 32k model scored under the authors' procedure is measured differently and is not used as a base.
- **Training data:** RULER's QA tasks use SQuAD and HotpotQA questions. From v7, these questions are also removed from the training data.

## Cautions when reading results

- Do not compare the training logs' validation across generations. Each generation reads 64 random windows, and in the v5 and v6 decay phases at most 31% of the Japanese text and 16% of the English text is shared. Even for the same final v5 weights, the 64 windows and the whole validation set gave .553 and .524 for Japanese and .66 and .639 for English. Trends within one generation can be compared, since they use the same windows.
- M-IFEval-Ja has only 172 questions, and the same weights scored .163 and .186. Differences of about .03 cannot be read.
- llm-jp-eval's M-IFEval-Ja code reads as if both strict and loose look only at the first of several instructions (44 questions; not confirmed by running it). Check it against the official scoring code.
- llm-jp-eval's KUCI and JMMLU scores are near 0 unless the model answers with the choice's symbol alone, and about .25 by guessing if it does. The answer form alone moves the Japanese mean a lot, so reports add the mean without these two.
- Always answering "neutral" on JNLI scores .555, and v5 has not exceeded this at any stage. After instruction tuning, JCommonsenseQA answers lean toward the first choice. Both scores can move because of answer bias (`research/experiments/v5-answer-patterns.md`), so reports add JNLI's score per correct label and JCommonsenseQA's score per position of the correct choice.

## Pass line

Decided for v7 before seeing its results.

- **Bases:** v5 and v6 after instruction tuning, asked the same way (the conversation format above); v5 is its 8k version, with v7's context of 8,192 tokens. v7 must pass against each.
  - v6 did not pass against v5: its English mean fell, and five tasks fell too far. Against v6 alone, v7 could pass while below v5 in English.
  - Against v5 alone, the Japanese line would be lenient. The instruction data v6 and v7 share teaches KUCI's and JMMLU's answer form: v6's Japanese mean rose .043 over v5's, but fell .027 without these two.
  - v6 is the better of two instruction-tuning runs, chosen on the same questions, so the comparison with v6 is slightly strict.
- **Scores:** the means of the Japanese tasks and of the English tasks must both rise by more than the readable difference, two standard errors of the difference between the means. `python -m pretrain pass-line` judges against one base at a time.
- **Tasks in the means:** every task both models were asked in its provider's form. Japanese: JCommonsenseQA, xWinograd, JNLI, JSQuAD, MGSM, KUCI, JMMLU and M-IFEval-Ja (strict). English: the 12 tasks in `results.md` and MMLU in its authors' form. OpenBookQA, WinoGrande and BoolQ are reference only, because neither base was asked them after instruction tuning; `pass-line` lists them apart from the means. RULER is also reference only (see "Long context (RULER)" above).
- **Falling tasks:** no task may fall by more than three standard errors of the difference between the two models. At two, some task would cross it about four times in ten even if nothing had changed (`research/experiments/readable-differences.md`).
- **Cost:** recorded in `results.md`, not judged. v7's budget was given to raise the scores, not to cost less.

## Comparison with outside models

Models after pretraining are measured without examples on the same seven tasks as HobbyLM's table (arXiv 2610.34582) and listed beside it. The tasks are HellaSwag, OpenBookQA, WinoGrande, ARC-Challenge, ARC-Easy, BoolQ and PIQA, with the table's metrics: acc for WinoGrande and BoolQ, and acc_norm for the others. The comparison is in `research/topics/outside-comparison.md`.

From v7, OpenBookQA, WinoGrande and BoolQ are also asked after instruction tuning; they are reference only in v7's pass line. Their questions are not removed from the training data up to v7; whether to remove them is decided when v8's data is built.
