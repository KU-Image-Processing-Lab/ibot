import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.Trans_CaraNet_ARA import Trans_CaraNet_ARA
from lib.ResNet_6 import ResNet6
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


def train(dataloaders_dict, models, optimizer, criterion, epoch, best_loss, best2_loss):
    val_loss = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            for model in models.values():
                model.train()
        else:
            for model in models.values():
                model.eval()

        # ---- multi-scale training ----
        size_rates = [1]
        loss_record5, d_loss_record = AvgMeter(), AvgMeter()

        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # d_optimizer.zero_grad()
                images, gts = pack
                labels = torch.einsum("ijkl->i", gts) > 0

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))
                # labels = labels.view(-1, 1)
                # labels = F.one_hot(labels, num_classes=2)

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                labels = Variable(labels).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                with torch.set_grad_enabled(phase == 'train'):
                    # ---- forward ----
                    lateral_map_5= models['Transfuse'](images)

                    # ---- loss function ----
                    loss5 = structure_loss(lateral_map_5, gts)
                    # loss4 = structure_loss(lateral_map_4, gts)
                    # loss3 = structure_loss(lateral_map_3, gts)
                    # loss2 = structure_loss(lateral_map_2, gts)
                    # d_loss = criterion(d_out, labels)
                    loss = loss5  # TODO: try different weights for loss

                    # TODO: try different weights for loss
                    # loss = loss1 + loss2 + loss3 + loss4 + loss5

                    # lateral_map_2 = lateral_map_2.sigmoid()#########################
                    # lateral_map_2 = 1. * (lateral_map_2 > 0.5)
                    # lateral_map_2 = images * lateral_map_2
                    lateral_map_5 = lateral_map_5.repeat(1, 3, 1, 1)

                    d_out = models['Discriminator'](lateral_map_5)
                    # d_out = models['Discriminator'](lateral_map_2, images)
                    d_loss = criterion(d_out, labels)
                    # losses = [loss, d_loss]
                    losses = torch.stack((loss, d_loss))
                    # ---- backward ----
                    if phase == 'train':
                        loss, extra_outputs = weight_method.backward(
                            losses=losses,
                            shared_parameters=shared_parameters,
                            task_specific_parameters=task_specific_parameters,
                            # last_shared_parameters=list(model.last_shared_parameters()),
                            # representation=features,
                        )
                        # optimizer.pc_backward(losses)
                        clip_gradient(optimizer, opt.clip)
                        optimizer.step()

                        # loss.backward(retain_graph=True)
                        # clip_gradient(optimizer, opt.clip)
                        # optimizer.step()
                        # # optimizer.zero_grad()
                        # d_loss.backward()
                        # clip_gradient(optimizer, opt.clip)
                        # clip_gradient(d_optimizer, opt.clip)
                        # optimizer.step()
                        # d_optimizer.step()

                        # ---- recording loss ----
                if rate == 1:
                    d_loss_record.update(d_loss.data, opt.batchsize)
                    # loss_record2.update(loss2.data, opt.batchsize)
                    # loss_record3.update(loss3.data, opt.batchsize)
                    # loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)


            if (i % 60 == 0 or i == total_step) and phase == 'train':
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      'lateral-5: {:0.4f}, d_loss: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record5.show(),
                             d_loss_record.show()))
        if phase == 'train':
            train_loss = loss_record5.show()
            train_d_loss = d_loss_record.show()
        elif phase == 'val':
            val_loss =loss_record5.show()
            val_d_loss = d_loss_record.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best.pth')
                print('[Saving best Snapshot:]', save_path + 'Transfuse-best.pth')
            if val_d_loss < best2_loss:
                best2_loss = val_d_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-best2.pth')
                torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-best2.pth')
                print('[Saving best Snapshot:]', save_path + 'Discriminator-best2.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 50 == 0:
        torch.save(models['Transfuse'].state_dict(), save_path + 'Transfuse-%d.pth' % epoch)
        torch.save(models['Discriminator'].state_dict(), save_path + 'Discriminator-%d.pth' % epoch)

        print('[Saving Snapshot:]', save_path + 'Transfuse-%d.pth' % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))

    return epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    # parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='./dataset/ValDataset', help='path to val dataset')
    # parser.add_argument('--train_path', type=str, default='./dataset/sekkai_TrainDataset', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='./dataset/sekkai_ValDataset', help='path to val dataset')
    parser.add_argument('--train_save', type=str, default='NASH6/-ARA/re')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    # parser.add_argument('--tuning', type=bool, default=True)
    parser.add_argument('--tuning', type=bool, default=False)

    parser.add_argument('--mtl', type=str, default='nashmtl')
    #parser.add_argument('--mtl', type=str, default='pcgrad')
    # parser.add_argument('--mtl', type=str, default='cagrad')
    # parser.add_argument('--mtl', type=str, default='imtl')
    # parser.add_argument('--mtl', type=str, default='mgda')
    # parser.add_argument('--mtl', type=str, default='dwa')
    # parser.add_argument('--mtl', type=str, default='uw')
    # parser.add_argument('--mtl', type=str, default='ls')
    # parser.add_argument('--mtl', type=str, default='scaleinvls')
    # parser.add_argument('--mtl', type=str, default='rlw')
    # parser.add_argument('--mtl', type=str, default='stl')

    opt = parser.parse_args()
    print("Tuning:", opt.tuning)
    # print('MTL:', opt.mtl)
    print('train save:', opt.train_save)
    print('train path:', opt.train_path)
    print('val   path:', opt.val_path)

    # ---- build models ----

    model1 = Trans_CaraNet_ARA(pretrained=True)
    # for name, param in model1.named_parameters():  #これいるんか？
    #     if param.requires_grad:
    #         print (name)
    if opt.tuning:
        model1.load_state_dict(torch.load('./weight2/TransCaraNet+MAE_calsification/石灰化ありのみ/Transfuse-best.pth'))

    if opt.tuning and opt.mtl == 'stl':
        for param in model1.parameters():
            param.requires_grad = False
    # model1 = nn.DataParallel(model1)
    model1.cuda()
    
    model2 = ResNet6()
    # model2 = nn.DataParallel(model2)
    model2 = model2.cuda()

    models = {'Transfuse': model1,
              'Discriminator': model2}

    weight_methods_parameters = extract_weight_method_parameters_from_args()
    weight_method = WeightMethods(
        opt.mtl, n_tasks=2, device='cuda:0', **weight_methods_parameters[opt.mtl]
    )

    params = [p for v in models.values() for p in list(v.parameters())]
    # shared_parameters = [p for v in models.values() for p in list(v.parameters())]

    # shared_parameters = [p for n, p in model1.named_parameters() if 'resnet.layer4' not in n and 'resnet.fc' not in n and 'cls' not in n]
    # task_specific_parameters = [p for n, p in model1.named_parameters() if 'resnet.layer4.0' in n or 'resnet.fc' in n or 'cls' in n]
    shared_parameters = [p for n, p in model1.named_parameters() if 'resnet.fc' not in n and 'cls' not in n]
    task_specific_parameters = [p for n, p in model1.named_parameters() if 'resnet.fc' in n or 'cls' in n]

    for n, p in model2.named_parameters():
        # if ("final_x" or 'final_1') in n:
        task_specific_parameters.append(p)


    # no=1
    # for n, p in model1.named_parameters():
    #     print(no)
    #     print(n)
    #     no+=1


    optimizer = torch.optim.Adam(
        [
            dict(params=params, lr=opt.lr, betas=(opt.beta1, opt.beta2)),
            dict(params=weight_method.parameters(), lr=0.025),
        ],
    )

    criterion = SmoothCrossEntropy()

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    image_root_val = '{}/images/'.format(opt.val_path)
    gt_root_val = '{}/masks/'.format(opt.val_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    val_loader = get_loader(image_root_val, gt_root_val, batchsize=opt.batchsize, trainsize=opt.trainsize, phase='val')

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    train_d_loss_list = []
    val_d_loss_list = []
    best_loss = 100000
    best2_loss = 100000

    #学習開始時間の記録
    start_time = time.time()

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)  ###################################
        # train(train_loader, model, optimizer, epoch)
        epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss = train(dataloaders_dict, models,
                                                                                             optimizer,
                                                                                             criterion,
                                                                                             epoch, best_loss,
                                                                                             best2_loss)
        epoch_list.append(epoch)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        train_d_loss = train_d_loss.cpu().data.numpy()
        train_d_loss_list.append(train_d_loss)
        val_d_loss = val_d_loss.cpu().data.numpy()
        val_d_loss_list.append(val_d_loss)

    #学習終了時間の記録
    end_time = time.time()

    #学習にかかった時間の計算
    training_time = end_time - start_time
    print(f"学習にかかった時間:{training_time}秒")


    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, label='train_loss')
    plt.plot(epoch_list, val_loss_list, label='val_loss', linestyle="--")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(left=0)
    plt.legend(loc='upper right')

    fig2 = plt.figure()
    plt.plot(epoch_list, train_d_loss_list, label='train_d_loss', linestyle=":")
    plt.plot(epoch_list, val_d_loss_list, label='val_d_loss', linestyle="-.")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(left=0)
    plt.legend(loc='upper right')

    fig.savefig("fig/img.png")
    fig2.savefig("fig/img2.png")
