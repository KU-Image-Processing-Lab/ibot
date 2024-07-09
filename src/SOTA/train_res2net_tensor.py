import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing

from lib.Discriminator_ResNet import Discriminator#使いたい識別機#
# TODO Add pre-processing model import

from utils.weight_methods import WeightMethods
from utils.mtl import extract_weight_method_parameters_from_args

from utils.dataloader2 import get_loader#データローダー
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as torch_model
from utils.smooth_cross_entropy import SmoothCrossEntropy

import random
import numpy as np

import time

def seed_everything(seed=42):
    random.seed(seed)  # Pythonの組み込み乱数生成器を固定
    os.environ['PYTHONHASHSEED'] = str(seed)  # Pythonのハッシュ乱数生成器を固定
    np.random.seed(seed)  # Numpyの乱数生成器を固定
    torch.manual_seed(seed)  # CPUでの乱数生成器を固定
    torch.cuda.manual_seed(seed)  # GPUでの乱数生成器を固定
    torch.cuda.manual_seed_all(seed)  # マルチGPU環境でも乱数生成器を固定
    torch.backends.cudnn.deterministic = True  # CuDNNを使用する場合の振る舞いを決定論的に
    torch.backends.cudnn.benchmark = False  # ネットワークのアーキテクチャが変わらない場合はTrueの方が高速化されるが、再現性を確保するためFalseに設定

seed_everything(42)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(dataloaders_dict, models, optimizer, criterion, epoch, best2_loss):
    val_loss = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            model=models
            model.train()
        else:
            model=models
            model.eval()

        # ---- multi-scale training ----
        size_rates = [1]
        d_loss_record = AvgMeter()
        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # d_optimizer.zero_grad()
                images, gts = pack
                labels = torch.einsum("ijkl->i", gts) > 0

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))
                if images.dim() == 5 and images.size(1) == 1:
                    images = images.squeeze(1)
                images = images.cuda()
                gts = gts.cuda()
                labels = labels.cuda()
                # ---- rescale ----
                with torch.set_grad_enabled(phase == 'train'):
                    # ---- forward ---
                    # ---- loss function ----
                    d_out = model(images)
                    d_loss = criterion(d_out, labels)
                    # ---- backward ----
                    if phase == 'train':
                        d_loss.backward()
                        optimizer.step()
                        # ---- recording loss ----
                if rate == 1:
                    d_loss_record.update(d_loss.data, opt.batchsize)

            if (i % 60 == 0 or i == total_step) and phase == 'train':#lossの変化を細かく見るかどうか
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      'd_loss: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             d_loss_record.show()))
        if phase == 'train':
            train_d_loss = d_loss_record.show()
        elif phase == 'val':
            val_d_loss = d_loss_record.show()
            if val_d_loss < best2_loss:
                best2_loss = val_d_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models.state_dict(), save_path + 'Discriminator-best2.pth')
                print('[Saving best Snapshot:]', save_path + 'Discriminator-best2.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 15 == 0:
        torch.save(models.state_dict(), save_path + 'Discriminator-%d.pth' % epoch)

        print('[Saving Snapshot:]', save_path + 'Discriminator-%d.pth' % epoch)
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))

    return epoch,train_d_loss, val_d_loss,best2_loss


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')#Res2Netなら少なくてもいい#
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')#もとは1e-4感触は4e-5のほうがいい#
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    # parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    # parser.add_argument('--train_path', type=str, default='./MTLTrans/hyper/Train/5/', help='path to train dataset')#使いたいデータセットのパス(train)#
    # parser.add_argument('--val_path', type=str, default='./MTLTrans/hyper/Val/5/', help='path to val dataset')#使いたいデータセットのパス(Val)#
    parser.add_argument('--train_path', type=str, default='./discri_dataset/split6hyper/Train/5', help='path to train dataset')#使いたいデータセットのパス(train)#
    parser.add_argument('--val_path', type=str, default='./discri_dataset/split6hyper/Val/5', help='path to val dataset')#使いたいデータセットのパス(Val)#
    parser.add_argument('--train_save', type=str, default='Res2Net50/cross5/5')#重みの保存場所#
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    # parser.add_argument('--tuning', type=bool, default=True)
    # parser.add_argument('--tuning', type=bool, default=False)

    opt = parser.parse_args()
    # print("Tuning:", opt.tuning)
    print('train save:', opt.train_save)
    print('train path:', opt.train_path)
    print('val   path:', opt.val_path)

    # ---- build models ----
    
    model2 = Discriminator()#インポートした識別機#
    # model2 = nn.DataParallel(model2)
    model2 = model2.cuda()
    params = list(model2.parameters())

    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
    criterion = SmoothCrossEntropy()

    image_root = '{}/tensor/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    image_root_val = '{}/tensor/'.format(opt.val_path)
    gt_root_val = '{}/masks/'.format(opt.val_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    val_loader = get_loader(image_root_val, gt_root_val, batchsize=opt.batchsize, trainsize=opt.trainsize, phase='val')

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 15, "Start Training", "#" * 15)

    epoch_list = []
    train_d_loss_list = []
    val_d_loss_list = []
    best2_loss = 100000
    #学習開始時間の記録
    start_time = time.time()
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)  ###################################
        
        epoch,train_d_loss, val_d_loss,best2_loss = train(dataloaders_dict, model2,
                                                            optimizer,
                                                            criterion,
                                                            epoch, 
                                                            best2_loss)
        epoch_list.append(epoch)
        train_d_loss = train_d_loss.cpu().data.numpy()
        train_d_loss_list.append(train_d_loss)
        val_d_loss = val_d_loss.cpu().data.numpy()
        val_d_loss_list.append(val_d_loss)

    #学習終了時間の記録
    end_time = time.time()

    #学習にかかった時間の計算
    training_time = end_time - start_time
    print(f"学習にかかった時間:{training_time}秒")


    fig2 = plt.figure()
    plt.plot(epoch_list, train_d_loss_list, label='train_d_loss', linestyle=":")
    plt.plot(epoch_list, val_d_loss_list, label='val_d_loss', linestyle="-.")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(left=0)
    plt.legend(loc='upper right')

    fig2.savefig("fig/ten/img2.png")#lossの動きを見る#