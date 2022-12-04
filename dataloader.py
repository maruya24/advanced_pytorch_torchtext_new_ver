# torchtextの0.14.0に対応
# torchdataをパッケージとしてインストールする必要がある
# datapipes方式で記述
# trainデータからボキャブラリーを構成せず学習済みベクトルをそのまま用いるため、単語数は99万個のままである
# データの読み込みについての参考URL: https://qiita.com/Nezura/items/268826b61f46f67705e4
# 学習済みの重みについての参考URL: https://extensive-nlp.github.io/TSAI-DeepNLP-END2.0/10_Seq2Seq_Attention/index.html

import string
import re
import pandas as pd
import torchdata.datapipes as dp
from torchtext.vocab import vocab
from torchtext.vocab import Vectors
import torchtext.transforms as T
import torch
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24):

    def preprocessing_text(text):
        # 改行コードを消去
        text = re.sub('<br />', '', text)

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        # ピリオドなどの前後にはスペースを入れておく
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text

    # 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）

    def tokenizer_punctuation(text):
        return text.strip().split()


    # 前処理と分かち書きをまとめた関数を定義
    def tokenizer_with_preprocessing(text):
        text = preprocessing_text(text)
        ret = tokenizer_punctuation(text)
        return ret

    # tsvファイルがそのままだとdatapipesで読み込みにくいので加工する
    df_train = pd.read_table('./data/IMDb_train.tsv', names=('text', 'label'),usecols=[0,1])
    df_test = pd.read_table('./data/IMDb_test.tsv', names=('text', 'label'),usecols=[0,1])
    df_train.to_csv('./data/IMDb_train_for_dp.tsv', sep="\t", index=False)
    df_test.to_csv('./data/IMDb_test_for_dp.tsv', sep="\t", index=False)

    # データ読み込み、単語分割
    train_dp = dp.iter.FileOpener(['./data/IMDb_train_for_dp.tsv'], mode='rt').parse_csv(delimiter='\t', skip_lines=1).map(lambda x: tokenizer_with_preprocessing(x), input_col=0)
    test_dp = dp.iter.FileOpener(['./data/IMDb_test_for_dp.tsv'], mode='rt').parse_csv(delimiter='\t', skip_lines=1).map(lambda x: tokenizer_with_preprocessing(x), input_col=0)

    # trainをtrain, valに分割
    N_ROWS = 25000
    train_dp, val_dp = train_dp.random_split(total_length=N_ROWS, weights={"train": 0.8, "valid": 0.2}, seed=0)

    # torchtextで単語ベクトルとして英語学習済みモデルを読み込む
    english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')

    pretrained_embeddings = english_fasttext_vectors.vectors # テンソルを取り出す。サイズは(全単語数, 一単語を表すベクトルの次元)

    # '<unk>', '<pad>', '<cls>', '<eos>'の埋め込みベクトルを適当に設定し、pretrained_embeddingsと組み合わせる
    # cls_vec, eso_vecは学習させるべきだがやり方が不明
    unk_vec = torch.mean(pretrained_embeddings, dim=0, keepdim=True)
    pad_vec = torch.zeros(1, pretrained_embeddings.shape[1])
    cls_vec = torch.rand(1, pretrained_embeddings.shape[1])
    eos_vec = torch.rand(1, pretrained_embeddings.shape[1])

    pretrained_embeddings = torch.cat((unk_vec, pad_vec, cls_vec, eos_vec, pretrained_embeddings),dim=0) # transformer.pyのEmbedderモジュールの引数として利用する

    # 単語辞書作成
    # vocabの引数は辞書。辞書のkeyは単語、valueは単語の頻度。
    # english_fasttext_vectors.stoiは辞書であり、keyが単語でvalueはその単語のid。
    # vocabに渡すenglish_fasttext_vectors.stoiのvalueは頻度ではなくidであるが、問題なく単語辞書を作成できる。
    # ただし、min_freq=0としないとenglish_fasttext_vectorsでidが0の単語の出現頻度が0と見なされてしまい、vocabオブジェクトに登録されない。※デフォルトはmin_freq=1
    text_vocab = vocab(english_fasttext_vectors.stoi, min_freq=0, special_first=True, specials=('<unk>', '<pad>', '<cls>', '<eos>'))
    text_vocab.set_default_index(text_vocab['<unk>']) # set_default_index: 辞書に含まれない単語には引数に与えた数字をindexとして対応させるように設定する

    cls_idx = text_vocab['<cls>']
    eos_idx = text_vocab['<eos>']

    # transform生成
    # 辞書による変換とパディング、Tensor型への変換
    # Sequentialを用いることで、複数のtransformを順番に実行可能
    text_transform = T.Sequential(
        T.VocabTransform(text_vocab), # 辞書を設定
        T.Truncate(max_length - 2), # 長い文章を切り捨てる
        T.AddToken(token=cls_idx, begin=True),
        T.AddToken(token=eos_idx, begin=False),
        T.ToTensor(padding_value=text_vocab['<pad>']), # ToTensorでパディングとテンソル型への変換。ミニバッチごとに系列長が統一される
    )

    # ミニバッチ時のデータ変換関数
    def collate_batch(batch):
        texts = text_transform([text for (text, label) in batch])
        labels = torch.tensor([int(label) for (text, label) in batch])
        return texts, labels


    # mapに変換
    train_ds = to_map_style_dataset(train_dp)
    val_ds = to_map_style_dataset(val_dp)
    test_ds = to_map_style_dataset(test_dp)

    # DataLoader設定
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_dl= DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # # 動作確認 検証データのデータセットで確認
    # batch = next(iter(val_dl))
    # print(batch[0].shape)
    # print(batch[0])
    # print(batch[1].shape)
    # print(batch[1])

    return train_dl, val_dl, test_dl, pretrained_embeddings


# _, _, _, pretrained_embeddings = get_IMDb_DataLoaders_and_TEXT()

# from transformer import Embedder

# net1 = Embedder(pretrained_embeddings)