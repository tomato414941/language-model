# language-model

Train Japanese and English language models from scratch. The goal is a better model than the current one on a smaller budget. A smaller budget is not a requirement for a new generation but one of the points by which an improvement is judged, so the next generation need not be cheaper than the previous one. Costs exclude trials.

What is known about how to build the model is in `docs/ideal-model.md`, how a generation is judged in `docs/evaluation.md`, and the results in `docs/results.md`.

```bash
bash scripts/setup_environment.sh   # install uv and the training dependencies
.tools/uv run --no-sync python -m pretrain --help
.tools/uv run --no-sync python -m unittest discover -s tests
```

## How we work

- Put in docs only decisions and findings needed later. Keeping writing up to date is a cost too, so leave out states and plans that soon go stale, and anything written elsewhere.
- Serve with vLLM. A model that uses a technique vLLM does not run (such as COUNT, which reads the preceding tokens) also needs its vLLM integration.
