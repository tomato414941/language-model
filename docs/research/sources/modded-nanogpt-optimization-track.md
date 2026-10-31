# modded-nanogpt の最適化の部門

KellerJordan/modded-nanogpt の records/track_3_optimization（2026-04 開始）。データ、バッチ、形を固定して、3.28 に届くまでのステップ数をオプティマイザだけで競う。統計的な有意性を条件にしていて、時間は問わないので、遅い手法も入っている。

| 結果 | ステップ | 内容 |
|---|---|---|
| #2 | 5625 | Adam（調整不足の可能性あり） |
| #36 | 3250 | 調整した Muon と補助の AdamW |
| #10 | 3250 | NorMuon |
| #39 | 3125 | Muon（#12、3325）に EMA-Nesterov |
| #20 | 3030 | SOAP の前処理、PowerCool、Contra-Muon と Soft-Muon |
| #29 | 2990 | #20 に radial brake |
| #44 | 2750 | すべての行列に SOAP-Muon |
| #45 | 2720 | #44 に最後の Tail-EMA |
| #46 | 2690 | #45 に RowFloor と cautious weight decay |

README によると、記録は調整済みの基準より20.8%少ない。どの手法がどれだけ効いたかは分かっていないとしていて、以前は効くと思われたものの一部は外されている。

#46 に残っているのは、SOAP-Muon（毎ステップ）、更新が小さい行の持ち上げ（u/w floor）、radial brake、PowerCool、Muon のモメンタムのスケジュール、EMA-Nesterov、補助の Adam の調整、初期値の調整、Tail-EMA（2400ステップから、地平150、0.4·θ + 0.6·EMA）。

A100 では、torch 2.10 と torch.compile の組み合わせで nan になるという注意がある。
