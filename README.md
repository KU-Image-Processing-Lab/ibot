# 医用画像処理

## このリポジトリの意義

医用画像処理に関するコードを共通化するためのリポジトリです。

## ルール

1. mainに直接プッシュしないようにしよう。
2. プルリクエストを送る前にmainブランチを最新の状態にしよう。
3. コミットメッセージはちゃんと書こう。
4. データセットやモデルの重みファイルはgitに上げないようにしよう。
5. ディレクトリ構成を変更する場合は、他の人に事前に相談しよう。



## ディレクトリ構成

ディレクトリ | 説明
--- | ---
utils | 便利な関数
lib | モデルの定義
weights | モデルの重みファイル
snapshots | 学習済みモデルのスナップショット
scripts | 実行スクリプト
dataset | データセット
results | 推論結果
docs | ドキュメント

## 参考ドキュメント

- [環境構築](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/settings.md)
- [データセット](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/datasets.md)
- [ブランチの切り方](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/branch.md)
- [リモートブランチを取り込む](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/remote_branch.md)
- [プルリクエストについて](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/pull_request.md)
- [コミットメッセージについて](https://github.com/KU-Image-Processing-Lab/medical-image-processing/tree/main/doc/commit.md)
