# MAEの実験
# python main_pretrain.py
修論で用いたMAEの事前学習済み重みの作成コード

健常者データを含む1141枚で事前学習
--data_path './dataset/TrainDataset/images'

非健常者データのみの491枚で事前学習
--data_path './dataset2/TrainDataset/images'
 
output_dirに重みが保存される





# ディレクトリ
# dataset
健常者データを含む1141枚のデータセット

# dataset2
非健常者データのみの491枚のデータセット

# output_dir
MAEによる事前学習をした重みが保存される


