# 複数の言語で学習する

言語を増やすと、何が変わるか。2026-11-02 に広く調べ直した。日英の比率を変えて一般の点を比べた結果は見つけていない。多くは1パラメータ100トークン前後以上の結果で、数十トークン以下の短い学習の結果は少ない。

## 計算

- [ATLAS](../sources/atlas.md)：言語を r 倍にして損失を保つ計算は、最小で r^0.97 倍。この式で2言語にすると、計算は約2倍、パラメータは1.18倍、全トークンは1.66倍になる（式からの計算）。計算をそろえて比べると、最適なパラメータあたりのトークンは1言語の0.95〜1.04倍で、ほとんど変わらない（同じ。論文に書かれた値ではない）。
- [Compute Optimal Tokenization](../sources/compute-optimal-tokenization.md)：6言語を一緒に学習すると、最適はどの言語も1パラメータ約70バイトにそろう。
- 損は容量から来るとする報告が多い（[Chang ら](../sources/curse-of-multilinguality.md)、ATLAS、[Foroutan ら](../sources/revisiting-multilingual-mixtures.md)）。大きなモデルほど小さい。

## 日本語と英語の組

[ATLAS](../sources/atlas.md)（2B、実測）：半々で読むと、単独と同じ損失に届くのに、日本語は1.3倍、英語は1.05倍のトークンが要る。中国語と読むと、日本語は1.05倍。

## 比率

- [He ら](../sources/multilingual-scaling-law.md)：5つの語族で、均等と最適の損失の合計は 10.974 と 10.969。
- [Schäfer ら](../sources/language-imbalance.md)（85M、12億トークン）：英語と仏語の 90対10 は、多い言語が良くなり、少ない言語が大きく悪くなる。
- [PLaMo-100B](../sources/plamo-100b.md) は英語 1.3T、日本語 0.7T で読んだ。比べた結果ではない。

## 英語を増やす

- [Foroutan ら](../sources/revisiting-multilingual-mixtures.md)（1.1B、3B）：全トークンを固定して英語を50%以上に増やすと、他の言語の損失は悪くなるが、下流の点は下がらない。
- [Gamayun](../sources/gamayun.md)（1.5B、2T）：英語を37%から70%に増やす学習は、最初から70%以上より多言語の点が高く、英語でも追いつく。
- [Kobayashi ら](../sources/japanese-medicine-mixture.md)（13B、医療）：英語のトークン数は日本語の課題とも強く相関し、日本の医師国家試験は日本語のトークン数と相関する。6コーパスの相関の分析。
- [Saito ら](../sources/local-language-models.md)の観察では、英語と日本語の点の相関は学校の科目で0.91、算数で0.94、コードで0.98と高い。日本についての知識と英日翻訳は、日本語のテキストで伸びる。
- [言語間の知識の研究](../sources/cross-lingual-compartmentalization.md)では、別の言語で学んだ事実を使える効率は、母語で学ぶ場合の 0.9%（360M）〜5.9%（7B）だった。トークンが別々であることが大きな原因で、言語を混ぜる工夫はほとんど効かなかった。英語とアラビア語、ロシア語の結果で、日本語では確かめられていない。日英半々の v6 は、同じ知識を2回学ぶことになる可能性がある。
- [Mix, Don't Tune](../sources/mix-dont-tune.md) は、データが足りない言語に英語を混ぜると下流の点が固有データの2〜13倍に相当するほど上がるとした。繰り返しを避ける効果が含まれるので、日本語のデータを繰り返さない v6 には当たらない可能性がある。読む量が増えて日本語のデータを繰り返す世代では当たる。見直し期限：2026-11-15。対象の言語の検証損失は混ぜる価値を過小に見るとも報告している。

事実の知識が移らないことと、下流の点が下がらないことは、同時に成り立つ可能性がある。測っているものが違う。

## 順番

- [Foroutan ら](../sources/revisiting-multilingual-mixtures.md)：言語を読む順番の工夫は、よく混ぜたものより良くならない。
- [Schäfer ら](../sources/language-imbalance.md)：複製した言語では効いたが、実際の言語では確かめられていない。

## データの質

[Foroutan ら](../sources/revisiting-multilingual-mixtures.md)、[Gamayun](../sources/gamayun.md)、[Seto ら](../sources/data-quality-bilingual.md)は、言語を混ぜたときの悪化の一部は質の低いデータから来るとしている。

## 指示学習

- [Kodama、Oda](../sources/instruction-pretraining-composition.md) では、日本語の指示のデータは英語の点も上げるが、英語の推論には英語のデータが要る。
- [Chen ら](../sources/multilingual-instruction-tuning.md)：計算をそろえると、言語ごとより、まとめて学習するほうが良いか同等。
- [コードスイッチング](../sources/multilingual-code-switching.md)：文ごとに言語を混ぜた指示学習のデータで、多言語の点が上がった。
- [継続学習の混ぜ方](../sources/multilingual-continual-pretraining.md)：2言語で学ぶと分類の点は上がるが、生成で言語が混ざることがある。

## 対訳

[対訳の使い方の研究](../sources/parallel-pretraining.md)（1.3B、120B）では、文書と翻訳を同じ窓で読んでから普通に読むと、一般の点と翻訳の点が上がった。対訳だけで読むと一般の点は安定しない。英語とフランス語、スワヒリ語の結果で、日英の文書の対訳の公開データは確かめていない。v6 には入れていない。次の世代を設計するときにデータの有無を調べる。見直し期限：2026-11-15。
