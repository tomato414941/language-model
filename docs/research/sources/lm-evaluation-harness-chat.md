# lm-evaluation-harness の会話形式

[Chat Templates の説明](https://lm-evaluation-harness.readthedocs.io/running_evals/chat_templates/)と、v0.4.13 の `docs/interface.md` の該当箇所を読んだ。説明の本文は検索の抜粋で読んだ。

チャットや指示学習のモデルには `--apply_chat_template` を使う。有効にすると例は複数のターンになる（`--fewshot_as_multiturn`）。選択式の課題では、回答の前の区切りが空になる。確率が要る課題は completions の API でしか動かない。
