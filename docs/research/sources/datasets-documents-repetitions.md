# Datasets, Documents, and Repetitions

Datasets, Documents, and Repetitions: The Practicalities of Unequal Data Quality（arXiv 2503.07879）。Fang ら。

厳しく選んだデータセットを10回まで繰り返すほうが、その10倍の量の元のデータセットを1回読むより良かった。計算量を何桁も変えても同じだった。ただし、weight decay など学習の設定を繰り返しに合わせて変えることが条件。比べたのは DCLM-baseline と、その上位の RefinedWeb。

繰り返しすぎると、どのデータでも伸びは鈍る。文書の単位で見ると、選んだデータには部分的に重複した文書が多く、文書ごとの回数を変えると、同じトークン数でより良いデータセットを作れるとしている。

読んだのは要旨と導入まで。
