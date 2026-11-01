# language-model

日本語と英語の言語モデルをゼロから学習する。新しい世代は、前の世代より少ない費用で、公開ベンチマークの点が高いことを目指す。

状態と次にやることは `docs/plan.md` にある。

```bash
bash scripts/setup_environment.sh   # uv と学習の依存関係を入れる
.tools/uv run --no-sync python -m pretrain --help
.tools/uv run --no-sync python -m unittest discover -s tests
```
