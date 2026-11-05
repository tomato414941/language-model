# language-model

Train Japanese and English language models from scratch. The goal is a better model than the current one on a smaller budget. A smaller budget is not a requirement for a new generation but one of the points by which an improvement is judged, so the next generation need not be cheaper than the previous one. Costs exclude trials.

What is known about how to build the model is in `docs/ideal-model.md`, and the results are in `docs/results.md`.

```bash
bash scripts/setup_environment.sh   # install uv and the training dependencies
.tools/uv run --no-sync python -m pretrain --help
.tools/uv run --no-sync python -m unittest discover -s tests
```

## How we work

- Write each generation's settings with that generation's values and reasons, not relative to another generation. Compare generations by scores and costs. Write the amount read as a ratio to the model's size (tokens or bytes per parameter).
- Name a time by its generation or date, never as "now".
- When writing what does not apply or what is left out, add the condition under which it would apply and a review date, and reconsider it before that date.
- Put in docs only decisions and findings needed later. Keeping writing up to date is a cost too, so leave out states and plans that soon go stale, and anything written elsewhere.
- After renting a machine, log in and check that it is working. If it stops, keep the machine, look inside, and resume from where it stopped.
- Send finished parts to R2 as they are done.
- Serve with vLLM. A model that uses a technique vLLM does not run (such as COUNT, which reads the preceding tokens) also needs its vLLM integration.
