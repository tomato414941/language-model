# v6 の最適化手法の割り当て

Metis-1.6 では、設定した手法が動いておらず、注意の行列が AdamW で学習されていた。同じことがないかを 2026-11-01 に確かめた。

`configs/model-v6.toml` の設定で幅64、4層の小さなモデルを作り、`make_optimizer` の割り当てを見た。

- NorMuon：注意の query、key_value、attention_output、attention_gate、value_gate と、MLP の up、down。
- AdamW：埋め込み、value embeddings、正規化、residual scales、input scales。
- 割り当てのない重みはなかった。

本番の機械のオブジェクトを直接見てはいない。本番のスナップショットから `pretrain/training.py` と `pretrain/model.py` は変わっていない。
