# advanced_pytorch_torchtext_new_ver

torchtextの最新バージョンで「PyTorchによる発展ディープラーニング」のテキストデータを読み込むために
第7章のdataloader.pyの関数get_IMDb_DataLoaders_and_TEXTを修正した。

# Features

修正前のコードでは文書の読み込みにtorchtext.dataのFeild, TabularDataset, Iteratorなど
を用いていたが、torchtextの最新バージョンではそれらが廃止されているために作成した。

# Requirement

* torchtext 0.14.0
* torchdata 0.5.0


# Installation


```bash
pip install torchdata
```

# Note

読み込んだデータを利用するにはdataloader.py以外のコードを適宜修正する必要があります。

# License

advanced_pytorch_torchtext_new_ver is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
