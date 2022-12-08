# advanced_pytorch_torchtext_new_ver

torchtextの最新バージョンで「PyTorchによる発展ディープラーニング」のテキストデータを読み込むために第7章のdataloader.pyの関数get_IMDb_DataLoaders_and_TEXTを修正した。

# Features

torchtextの最新バージョンではtorchtext.dataのFeild, TabularDataset, Iteratorなどが廃止されているが、修正したコードでは
それらを用いずにテキストデータを読み込むことができる。torchtextの0.14.0に対応している。

# Requirement

* torchtext 0.14.0
* torchdata 0.5.0


# Installation


```bash
pip install torchdata
```

# Note

読み込んだデータを利用するにはdataloader.py以外のコードを適宜修正する必要がある。

# License

advanced_pytorch_torchtext_new_ver is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
