# 必要なライブラリとモジュールをインポート
import json
import logging
import os
import random

import Config as config
import numpy as np
import torch.optim
from Load_Dataset_U import ImageToImage2D, RandomGenerator, ValGenerator
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from Train_one_epoch import train_one_epoch
from utilsU import CosineAnnealingWarmRestarts, WeightedDiceBCE

# 異なるネットワークアーキテクチャをインポート
from lib.ACC_UNetl import ACC_UNet

# from nets.MResUNet1 import MultiResUnet
# from nets.SwinUnet import SwinUnet
# from nets.UNet_base import UNet_base
# from nets.SMESwinUnet import SMESwinUnet
# from nets.UCTransNet import UCTransNet


# ロガーの設定を行う関数
def logger_config(log_path):
    loggerr = logging.getLogger()  # ロガーを取得
    loggerr.setLevel(level=logging.INFO)  # INFOレベル以上のログを出力
    handler = logging.FileHandler(
        log_path, encoding="UTF-8"
    )  # ファイルハンドラーを設定
    handler.setLevel(logging.INFO)  # ファイルハンドラーのログレベルを設定
    formatter = logging.Formatter("%(message)s")  # ログフォーマットを定義
    handler.setFormatter(formatter)  # フォーマッターをハンドラーに設定
    console = logging.StreamHandler()  # コンソール出力用のストリームハンドラーを設定
    console.setLevel(logging.INFO)  # コンソールのログレベルを設定
    loggerr.addHandler(handler)  # ファイルハンドラーをロガーに追加
    loggerr.addHandler(console)  # ストリームハンドラーをロガーに追加
    return loggerr  # 設定されたロガーを返す


# モデルのチェックポイントを保存する関数
def save_checkpoint(state, save_path):
    logger.info("\t Saving to {}".format(save_path))  # ログ出力
    if not os.path.isdir(save_path):  # 保存ディレクトリが存在しない場合
        os.makedirs(save_path)  # ディレクトリを作成

    epoch = state["epoch"]  # エポック番号を取得
    best_model = state["best_model"]  # 最良モデルかどうかのフラグ
    model = state["model"]  # モデル名

    if best_model:  # 最良モデルの場合
        filename = (
            save_path + "/" + "best_model-{}.pth.tar".format(model)
        )  # ファイル名を設定
    else:  # 最良モデルでない場合
        filename = (
            save_path + "/" + "model-{}-{:02d}.pth.tar".format(model, epoch)
        )  # ファイル名を設定
    torch.save(state, filename)  # モデルを保存


# データローダーのワーカー初期化関数
def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)  # 乱数シードを設定


# 主要な処理を行うメインループ関数
def main_loop(batch_size=config.batch_size, model_type="", tensorboard=True):
    # トレーニングとバリデーションデータのロード
    train_tf = transforms.Compose(
        [RandomGenerator(output_size=[config.img_size, config.img_size])]
    )  # トレーニングデータの変換設定
    val_tf = ValGenerator(
        output_size=[config.img_size, config.img_size]
    )  # バリデーションデータの変換設定
    train_dataset = ImageToImage2D(
        config.train_dataset, train_tf, image_size=config.img_size
    )  # トレーニングデータセットを作成
    val_dataset = ImageToImage2D(
        config.val_dataset, val_tf, image_size=config.img_size
    )  # バリデーションデータセットを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=8,
        pin_memory=True,
    )  # トレーニング用のデータローダー
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=8,
        pin_memory=True,
    )  # バリデーション用のデータローダー

    lr = config.learning_rate  # 学習率を設定

    logger.info(model_type)  # モデルタイプをログ出力
    logger.info("n_filts : " + str(config.n_filts))  # フィルタ数をログ出力

    # モデルタイプに応じてモデルを初期化
    if model_type == "ACC_UNet":
        config_vit = config.get_CTranS_config()  # Transformer設定を取得
        model = ACC_UNet(
            n_channels=config.n_channels, n_classes=config.n_labels
        )  # ACC_UNetモデルを初期化

    # elif model_type == 'UCTransNet':
    #     config_vit = config.get_CTranS_config()  # Transformer設定を取得
    #     logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))  # Transformerのヘッド数をログ出力
    #     logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))  # Transformerのレイヤー数をログ出力
    #     logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))  # Transformerの拡張比をログ出力
    #     model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)  # UCTransNetモデルを初期化

    # elif model_type == 'UNet_base':
    #     config_vit = config.get_CTranS_config()  # Transformer設定を取得
    #     model = UNet_base(n_channels=config.n_channels, n_classes=config.n_labels)  # UNet_baseモデルを初期化

    # elif model_type == 'SMESwinUnet':
    #     config_vit = config.get_CTranS_config()  # Transformer設定を取得
    #     model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels)  # SMESwinUnetモデルを初期化
    #     model.load_from()  # 事前学習したパラメータをロード
    #     lr = 5e-4  # 学習率を設定

    # elif model_type == 'SwinUnet':
    #     model = SwinUnet()  # SwinUnetモデルを初期化
    #     model.load_from()  # 事前学習したパラメータをロード
    #     lr = 5e-4  # 学習率を設定

    # elif model_type.split('_')[0] == 'MultiResUnet1':
    #     model = MultiResUnet(n_channels=config.n_channels, n_classes=config.n_labels, nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))  # MultiResUnetモデルを初期化

    else:
        raise TypeError(
            "Please enter a valid name for the model type"
        )  # 無効なモデルタイプの場合は例外を発生

    # オプティマイザの設定
    if model_type == "SwinUnet":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001
        )  # SGDオプティマイザを設定
    elif model_type == "SMESwinUnet":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001
        )  # SGDオプティマイザを設定
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )  # Adamオプティマイザを設定

    model = model.cuda()  # モデルをGPUに移動

    logger.info("Training on " + str(os.uname()[1]))  # ホスト名をログ出力
    logger.info(
        "Training using GPU : "
        + torch.cuda.get_device_name(torch.cuda.current_device())
    )  # 使用中のGPU名をログ出力

    criterion = WeightedDiceBCE(
        dice_weight=0.5, BCE_weight=0.5, n_labels=config.n_labels
    )  # 損失関数を設定

    # 学習率スケジューラの設定
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-4
        )  # コサインアニーリングスケジューラを設定
    else:
        lr_scheduler = None  # スケジューラなし

    # TensorBoardの設定
    if tensorboard:
        log_dir = config.tensorboard_folder  # ログディレクトリを設定
        logger.info("log dir: ".format(log_dir))  # ログディレクトリをログ出力
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)  # ログディレクトリが存在しない場合、ディレクトリを作成
        writer = SummaryWriter(log_dir)  # SummaryWriterを設定
    else:
        writer = None  # Writerなし

    max_dice = 0.0  # 最高Diceスコアを初期化
    best_epoch = 1  # 最良エポックを初期化
    for epoch in range(config.epochs):  # エポック数だけ繰り返し
        logger.info(
            "\n========= Epoch [{}/{}] =========".format(epoch + 1, config.epochs + 1)
        )  # 現在のエポック数をログ出力
        logger.info(config.session_name)  # セッション名をログ出力
        # トレーニングを実行
        model.train(True)  # モデルをトレーニングモードに設定
        logger.info(
            "Training with batch size : {}".format(batch_size)
        )  # バッチサイズをログ出力
        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            writer,
            epoch,
            None,
            model_type,
            logger,
        )  # 1エポック分のトレーニングを実行
        # バリデーションを実行
        logger.info("Validation")  # バリデーション開始のログ出力
        with torch.no_grad():  # 勾配計算を無効化
            model.eval()  # モデルを評価モードに設定
            val_loss, val_dice = train_one_epoch(
                val_loader,
                model,
                criterion,
                optimizer,
                writer,
                epoch,
                lr_scheduler,
                model_type,
                logger,
            )  # 1エポック分のバリデーションを実行

        # 最良モデルの保存処理
        if val_dice > max_dice:
            logger.info(
                "\t Saving best model, mean dice increased from: {:.4f} to {:.4f}".format(
                    max_dice, val_dice
                )
            )  # 最良モデルの更新をログ出力
            max_dice = val_dice  # 最高Diceスコアを更新
            best_epoch = epoch + 1  # 最良エポックを更新
            save_checkpoint(
                {
                    "epoch": epoch,
                    "best_model": True,
                    "model": model_type,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                config.model_path,
            )  # 最良モデルを保存
        else:
            logger.info(
                "\t Mean dice:{:.4f} does not increase, the best is still: {:.4f} in epoch {}".format(
                    val_dice, max_dice, best_epoch
                )
            )  # Diceスコアが改善されなかった場合のログ出力
        early_stopping_count = (
            epoch - best_epoch + 1
        )  # アーリーストッピングカウンタを更新
        logger.info(
            "\t early_stopping_count: {}/{}".format(
                early_stopping_count, config.early_stopping_patience
            )
        )  # アーリーストッピングカウンタをログ出力

        if (epoch + 1) % 50 == 0:
            torch.save(
                model.state_dict(), config.save_path + "Transfuse-%d.pth.tar" % epoch
            )
            print(
                "[Saving Snapshot:]", config.save_path + "Transfuse-%d.pth.tar" % epoch
            )

        if (
            early_stopping_count > config.early_stopping_patience
        ):  # アーリーストッピング条件を満たした場合
            logger.info("\t early_stopping!")  # トレーニング停止のログ出力
            break  # ループから脱出

    return model  # トレーニングされたモデルを返す


# スクリプトが直接実行された場合のみ以下の処理を実行
if __name__ == "__main__":
    deterministic = True  # 決定論的な動作を選択
    if not deterministic:  # 決定論的でない場合
        cudnn.benchmark = True  # ベンチマークモードを有効化（パフォーマンス向上のための最適化が行われる）
        cudnn.deterministic = False  # 決定論的な動作を無効化
    else:  # 決定論的な場合
        cudnn.benchmark = False  # ベンチマークモードを無効化
        cudnn.deterministic = True  # 決定論的な動作を有効化
    random.seed(config.seed)  # 乱数のシードを設定
    np.random.seed(config.seed)  # NumPyの乱数のシードを設定
    torch.manual_seed(config.seed)  # PyTorchの乱数のシードを設定
    torch.cuda.manual_seed(config.seed)  # GPU用の乱数のシードを設定
    torch.cuda.manual_seed_all(config.seed)  # すべてのGPUに対する乱数のシードを設定
    if not os.path.isdir(config.save_path):  # 保存パスが存在しない場合
        os.makedirs(config.save_path)  # 保存ディレクトリを作成

    # if os.path.isfile(config.logger_path):  # ログファイルがすでに存在する場合
    #     import sys
    #     sys.exit()  # スクリプトを終了
    logger = logger_config(log_path=config.logger_path)  # ロガーを設定
    model = main_loop(
        model_type=config.model_name, tensorboard=True
    )  # メインループを実行し、モデルを取得

    fp = open("log.log", "a")  # ログファイルを追記モードで開く
    fp.write(f"{config.model_name} completed\n")  # ログに完了メッセージを追加
    fp.close()  # ログファイルを閉じる
