# Nemotron-CC v2 / v2.1

見送った。

NVIDIA の、品質で選んだ Common Crawl。カードに「ready for commercial use」とあり、NVIDIA Data Access Agreement のもとで使える（申請が必要）。ただしカードに、これで学習して配布したモデルには Qwen、DeepSeek、Phi-4 のライセンスの条件がかかりうるとある。他のモデルが書いた文章を含むと読み、他の人のモデルからの蒸留を禁じるルールに触れるとして見送った。

ところが今のデータにも、他のモデルが書き直した Swallow の数学とコードが入っている（sources.toml に「rewrites」とある）。公開されているデータセットに含まれる生成文がこのルールに入るかは決まっておらず、この見送りは私の解釈による。
