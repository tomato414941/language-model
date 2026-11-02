# language-model

日本語と英語の言語モデルをゼロから学習する。目標は、今より少ない予算で、今より良いモデルを作ること。少ない予算は世代を上げる要件ではなく、改善を評価する点の1つで、次の世代が前の世代より安くなくてもよい。

状態と次にやることは `docs/plan.md` に、モデルの作り方について分かっていることは `docs/ideal-model.md` にある。

```bash
bash scripts/setup_environment.sh   # uv と学習の依存関係を入れる
.tools/uv run --no-sync python -m pretrain --help
.tools/uv run --no-sync python -m unittest discover -s tests
```
