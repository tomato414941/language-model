# MuonEq

MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration（arXiv 2603.28254）。

少ない回数の直交化の精度は入力の行列の形に左右されるとして、直交化の前に行や列の大きさをそろえる。行と列の両方（RC）、行（R）、列（C）の3つがあり、隠れ層の行列には R を既定とする。Muon と同じ収束の保証を示している。

LLaMA2 を C4 で学習した比較では、130M、350M、1B のすべてで R が Muon より速く収束し、検証の perplexity も低かったとしている。

読んだのは要旨、導入、結論まで。表の数字は確かめていない。
