# SOAP-Muon

Improving SOAP using Iterative Whitening and Muon（Vyas ら、nikhilvyas.github.io/SOAP_Muon.pdf）。実装は GitHub の nikhilvyas/SOAP_Muon。

行列の小さいほうの辺に SOAP の前処理をかけ、その勾配に Muon をかける。小さいほうの辺だけなので、時間と記憶の負担は SOAP より小さい。バッチが大きいときに、片側の SOAP と Muon のどちらより良いとしている。

著者は、解析は単純なモデルによるもので、実験は予備的なものだと書いている。
