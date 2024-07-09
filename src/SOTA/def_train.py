import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.ACC_UNet_lite import ACC_UNet_Lite
from lib.ResNet_10 import ResNet10
# TODO Add pre-processing model import

from utils.weight_methods import WeightMethods
from utils.mtl import extract_weight_method_parameters_from_args

from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as torch_model
from utils.smooth_cross_entropy import SmoothCrossEntropy

import random
import numpy as np

import time


# データローダー、モデル、オプティマイザー、損失関数などを受け取り、エポックごとのトレーニングと検証を実行する関数
def train(dataloaders_dict, models, optimizer, criterion, epoch, best_loss, best2_loss):
    val_loss = 0  # 検証損失の初期化

    # 'train'と'val'のフェーズで処理を行う
    for phase in ['train', 'val']:
        # トレーニングフェーズの場合
        if phase == 'train':
            for model in models.values():  # 全モデルに対して
                model.train()  # モデルをトレーニングモードに設定
        else:  # 検証フェーズの場合
            for model in models.values():  # 全モデルに対して
                model.eval()  # モデルを評価モードに設定

        # マルチスケールトレーニング用のサイズレート設定
        size_rates = [1]
        loss_record2, d_loss_record = AvgMeter(), AvgMeter()  # 損失の平均を計算するためのメーター

        # データローダーからデータパックを取得
        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:  # サイズレートに基づく処理
                optimizer.zero_grad()  # オプティマイザの勾配をゼロクリア

                images, gts = pack  # 画像と正解ラベルを取得
                labels = torch.einsum("ijkl->i", gts) > 0  # 正解ラベルからバイナリラベルを計算

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))  # 条件に応じてラベルを更新

                images = Variable(images).cuda()  # 画像をCUDA変数に設定
                gts = Variable(gts).cuda()  # 正解ラベルをCUDA変数に設定
                labels = Variable(labels).cuda()  # ラベルをCUDA変数に設定

                # レートに応じた画像サイズの調整
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                with torch.set_grad_enabled(phase == 'train'):  # 勾配計算の設定
                    lateral_map = models['Transfuse'](images)  # Transfuseモデルに画像を入力して処理

                    loss = structure_loss(lateral_map, gts)  # 構造損失の計算

                    lateral_map = lateral_map.repeat(1, 3, 1, 1)  # 損失マップの調整

                    d_out = models['Discriminator'](lateral_map)  # Discriminatorモデルに損失マップを入力して処理
                    d_loss = criterion(d_out, labels)  # Discriminatorの出力に基づく損失の計算

                    losses = torch.stack((loss, d_loss))  # トータルの損失をスタック

                    if phase == 'train':  # トレーニングフェーズの場合
                        loss, extra_outputs = weight_method.backward(
                            losses=losses,
                            shared_parameters=shared_parameters,
                            task_specific_parameters=task_specific_parameters,
                        )
                        clip_gradient(optimizer, opt.clip)  # 勾配のクリッピング
                        optimizer.step()  # オプティマイザによるパラメータ更新

                if rate == 1:
                    d_loss_record.update(d_loss.data, opt.batchsize)  # Discriminatorの損失の記録
                    loss_record2.update(loss.data, opt.batchsize)  # トレーニング損失の記録

            if (i % 900 == 0 or i == total_step) and phase == 'train':  # 一定のステップでログを表示
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral: {:.4f}, d_loss: {:.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record2.show(), d_loss_record.show()))

        if phase == 'train':  # トレーニングフェーズ後の損失の保存
            train_loss = loss_record2.show()
            train_d_loss = d_loss_record.show()
        elif phase == 'val':  # 検証フェーズ後の処理
            val_loss = loss_record2.show()
            val_d_loss = d_loss_record.show()
            if val_loss < best_loss:  # 最良の損失を更新
                best_loss = val_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best.pth')
                print('[Saving best Snapshot:]', save_path + 'Transfuse-best.pth')
            if val_d_loss < best2_loss:  # 別の損失基準で最良を更新
                best2_loss = val_d_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best2.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best2.pth')
                print('[Saving best Snapshot:]', save_path + 'Discriminator-best2.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)  # スナップショットの保存パス設定
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 20 == 0:  # 特定のエポックごとにモデルを保存
        torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-%d.pth' % epoch)
        torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-%d.pth' % epoch)

        print('[Saving Snapshot:]', save_path + 'Transfuse-%d.pth' % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))  # トレーニングと検証損失の表示
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))  # Discriminatorのトレーニングと検証損失の表示

    return epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss  # 結果を返す
