# 計画

## 目標

日英のモデルをゼロから作り、世代を重ねて良くする。新しい世代は、前の世代より少ない費用で、公開ベンチマークの点が前の世代より高いことを目指す。費用には試しの分を入れない。

## v6

モデルの形と学習の組み立ては `configs/model-v6.toml`、`configs/model-v6-patch.toml`、`configs/pretrain-v6.toml` にある。

| 工程 | 状態 | 費用 |
|---|---|---|
| データ準備 | 完了。R2 の `data/pretrain-v6/` にある | $0.68 |
| 事前学習、指示学習、評価 | 実行中。2026-11-01 に開始 | 約$17 の見込み |
| 学習率を変えた2本目の指示学習と評価 | 上の後に同じ機械で走る | 約$2 の見込み |

費用の見込みは、この機械で測った速さから計算した。パッチ段階は133,500トークン/秒で11.7時間かかり、別の RTX 4090 で測った値（`research/experiments/v6-throughput.md`）の91%だった。安定期の始めは32,700トークン/秒で、同じく93%。2本目まで入れて約47時間かかる。

この機械の GPU テストは4件のうち1件が失敗した。Gated DeltaNet の層を bfloat16 と float32 の混在で解くテストで、0.6%の要素が許容差を超えた。v6 は全層が注意の層でこの層を作らないので、本番には影響しない。この層を使う世代では、先に原因を調べる。見直し期限：2026-11-15。

予算に入れない費用は、速さの測定が $0.29、途中で失敗したデータ準備が $0.34、選ばなかった方の指示学習と評価。

指示学習は学習率 0.00016 と 0.0006 を比べる。2本目は `sft.toml` の学習率、最小学習率、`scalar_rate` だけを 0.0006、0.00006、0.375 に変えた。実行中の計画を止めないように、最初の段階が正常に終わると機械の上の待機処理が計画をもう一度起動し、終わった工程を飛ばして2本目だけを走らせる。機械が替わった場合は、新しい機械が2本目まで続ける。

## 本番の操作

操作の一式は `runs/pretrain-v6-control/` にあり、git には入っていない。転送の URL を含むファイルは表示しない。

- `vast_job.py status` は機械の状態と工程の記録を、`balance` は残高を出す。`ssh` は機械の中でコマンドを実行する。どれも Vast の鍵を環境変数で渡す。
- `vast_job.py watch --seconds 300 --max-run 22 --keep-hours 4` が監視として動いている。結果が R2 にそろうと機械を消す。失敗か沈黙なら4時間残してから消す。host が死ぬと借り直し、新しい機械が R2 の記録から続ける。
- 結果は R2 の `stages/pretrain-v6/20261101T053916Z/main/` に置かれる。`final/` が完成品、`logs/` が記録、`checkpoints/` が途中の重み。転送の URL の期限は 2026-11-08 の 12:12 UTC。
- 機械の中では `/workspace/lm` が作業場所で、`runs/pretrain-v6-results/main/` に工程の記録と学習のログがある。2本目を起動する待機処理は `/workspace/second_tuning.sh`。

## 次にやること

1. 本番の各段階の切り替わりで、機械の中を確かめる。
2. 本番の結果が R2 にそろったら、別の機械で測る。操作の一式は `runs/measure-v6-control/` にあり、git には入っていない。`prepare_snapshot.sh`、`runs/storage-control/r2.py` の下で `build_manifest.py`、`vast_job.py launch-cheapest --max-run 2` の順に動かし、機械の中を確かめてから `vast_job.py watch --relaunches 1` を動かす。測るのは次の4つで、2時間、約$0.9 の見込み（未実測）。`configs/pretrain-v6.toml` の課題には入れていない。
   - v5 の 8k 版の指示学習の後（重みは R2 の `stages/instructions-v5/20261028T182342Z/main/final/instructions/last.pt`）を、v6 の指示学習の後と同じコマンドに MMLU の作者の形式を足して測る。
   - v6 の2本を MMLU の作者の形式で測る。
   - 外のモデルと並べるために、v5 の 8k 版の事前学習の後（重みは R2 の `stages/pretrain-v5/20261025T152135Z/main/final/decay/last.pt`）と v6 の事前学習の後（SFT の出発点の `last.pt`）を、`--tasks openbookqa winogrande boolq` で測る。
   - 3つのモデルの M-IFEval-Ja の返答を、公式の採点コードで採点する。
3. 両方の `fetch_results.py` で結果を取り、`runs/measure-v6-control/merge_results.py` で判定の記録を `runs/measure-v6-results/judged/` に作る。`python -m pretrain pass-line` で比べ、`evaluation.md` の決まりで v6 とする方を選ぶ。答えの偏りの内訳は `runs/pretrain-v6-control/analysis/` のスクリプトで出す。
4. 結果を `results.md` に書き、合格の線と比べる。差がつかない課題があれば、他の人が提供している簡単なテストを足す。
5. 2026-11-15 より前に、見直し期限の記述を再考し、期限を延ばす。

## 決まっていないこと

- 指示学習のデータを足すか（`candidates.md`）。v6 の結果を見て決める

## 進め方

- 機械は借りたら中に入って動いていることを確かめる。止まったら機械を残し、中を調べて続きから直す。
- できた部分から順に R2 に送る。
- 決めたことと分かったことは、そのコミットでこの `docs/` に反映する。
- 時期は「今」と書かず、世代の名前か日付で書く。
- 当てはまらないことや入れないことを書くときは、当てはまるようになる条件と見直し期限を添える。期限の前に再考し、期限を延ばす。`grep -rn 見直し期限 docs` で探せる。
