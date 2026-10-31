# DataComp-LM

Li ら、DataComp-LM: In search of the next generation of training sets for language models（arXiv 2406.11794）。要旨と結論を読んだ。

Common Crawl の 240T トークンと、412M から 7B のモデルで、データの作り方を比べる場。モデルによるフィルタが鍵だとしている。そこから作った DCLM-baseline で、7B のモデルを 2.6T トークン学習して MMLU（5-shot）が64%に届いた。計算の制約から、要素を1つずつしか比べられず、走らせるごとの差も十分に調べていないとしている。
