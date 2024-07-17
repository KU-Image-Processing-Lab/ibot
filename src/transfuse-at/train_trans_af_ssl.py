import torch
from torch.autograd import Variable
import os
import argparse
import time
from datetime import datetime
# from lib.TransFuse_s import TransFuse_S
from lib.se_TransFuse_s import SE_TransFuse_S
from lib.Bi_TransFuse_s import Bi_TransFuse_S
from lib.enhanced_TransFuse import enhanced_TransFuse
from utils.dataloaderT import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from af_loss import AFLoss
from af_loss import AFLossRest
from lib.TransFuse_s_split import TransFuse_S_transformer
from lib.TransFuse_s_split import TransFuse_S_resnet
from lib.TransFuse_s_split import TransFuse_S_joint
from lib.TransFuse_s_split import TransFuse_S

import numpy as np


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(dataloaders_dict, unlabeled_dataloader, model, optimizer, epoch, best_loss, vat_alpha, vat_eps):
    # load vat_loss
    af_loss = AFLoss(eps=vat_eps, xi=1e-6, ip=1)
    af_loss_res = AFLossRest(eps=vat_eps, xi=1e-6, ip=1)
    alpha = vat_alpha  # 1.0
    val_loss = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        # ---- multi-scale training ----
        size_rates = [1]
        loss_record2, loss_record3, loss_record4, loss_record_lds, loss_record_ssl = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                with torch.set_grad_enabled(phase == 'train'):
                    # ---- forward ----
                    lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

                    # ---- loss function ----
                    loss4 = structure_loss(lateral_map_4, gts)  # BiFusion map
                    loss3 = structure_loss(lateral_map_3, gts)  # Transformer map
                    loss2 = structure_loss(lateral_map_2, gts)  # Joint map

                    normal_loss = 0.5 * loss2 + 0.2 * loss3 + 0.3 * loss4

                    # ---- calc vat_loss(LDS) only train ----
                    if phase == 'train':
                        split_num = random.randint(0, 2)# * 3
                        """
                        random_num = random.randrange(100)
                        if random_num < 10:
                            trans_split_num = 1
                        elif 10 <= random_num < 30:
                            trans_split_num = 2
                        elif 30 <= random_num < 60:
                            trans_split_num = 3
                        elif 60 <= random_num < 90:
                            trans_split_num = 4
                        elif 90 <= random_num < 100:
                            trans_split_num = 5
                        print(split_num, random_num)
                        """
                        trans_split_num = 3#random.randint(3,4)
                        """
                        if split_num == 0:
                            af_loss.eps = 1e-1
                            af_loss_res.eps = 1e-1
                        else:
                            af_loss.eps = 1e-3
                            af_loss_res.eps = 1e-3
                        """
                        if opt.eps_rumpup:
                            af_loss.eps = np.exp(-5.0*((1-((epoch+1)/opt.epoch))**2))*vat_eps
                            af_loss_res.eps = np.exp(-5.0 * ((1 - ((epoch + 1) / opt.epoch)) ** 2)) * vat_eps
                            print(af_loss.eps)
                        lds_u = af_loss(model, images, split_num,trans_split_num)#, gts)
                        lds_r = af_loss_res(model, images, split_num,trans_split_num)#, gts)
                        lds = lds_u + lds_r


                        labeled_loss = normal_loss + alpha * lds_u + alpha * lds_r
                        
                        # non-label data
                        unlabel_image, _ = iter(unlabeled_dataloader).next()
                        unlabel_image = Variable(unlabel_image).cuda()
                        unlabel_lds_u = af_loss(model, unlabel_image, split_num, trans_split_num)
                        unlabel_lds_r = af_loss_res(model, unlabel_image, split_num, trans_split_num)
                        unlabel_lds = unlabel_lds_u + unlabel_lds_r
                        
                        loss = labeled_loss + alpha * unlabel_lds
                    else:
                        loss = normal_loss

                    # ---- backward - ---
                    if phase == 'train':
                        loss.backward()
                        # clip_gradient(optimizer, opt.clip)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
                        optimizer.step()

                # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    if phase == 'train':
                        loss_record_lds.update(lds, opt.batchsize)
                        loss_record_ssl.update(unlabel_lds, 1)

            # ---- train visualization ----
            if (i % 20 == 0 or i == total_step) and phase == 'train':
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, af_loss: {:0.8f}, ssl_loss: {:0.8f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record_lds.show(), loss_record_ssl.show()))
        if phase == 'train':
            train_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()# + loss_record_lds.show()
            ssl_loss = loss_record_ssl.show()
        elif phase == 'val':
            val_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = 'snapshots/{}/'.format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), save_path + 'TransFuse-best.pth')
                print('[Saving best Snapshot:]', save_path + 'TransFuse-best.pth')

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    return epoch, train_loss, val_loss, best_loss, ssl_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    # parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='./dataset/sekkai_TrainDataset528_unity', help='path to train dataset')
    # parser.add_argument('--train_path', type=str, default='./dataset/sekkai_CrossTrain5_original', help='path to train dataset')
    parser.add_argument('--nonlabel_path', type=str, default='./DISCRI/dataset_ssl/sd3000_unity', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='./dataset/sekkai_ValDataset_unity', help='path to val dataset')
    # parser.add_argument('--val_path', type=str, default='./dataset/sekkai_CrossVal5_original', help='path to val dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--vat_alpha', type=float, default=0.01, help='alpha of vat loss')
    parser.add_argument('--vat_eps', type=float, default=0.001, help='epsilon of vat loss')
    parser.add_argument('--eps_rumpup', type=bool, default=0, help='enable epsilon rumpup')

    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device

    # model = TransFuse_S(pretrained=True).cuda()
    # split transfuse_S
    model = TransFuse_S(pretrained=True).cuda()
    #model.load_state_dict(torch.load('./snapshots/TransFuse_S/TransFuse-119.pth'))
    #model.cuda()
    # model = SE_TransFuse_S(pretrained=True).cuda()
    # model = Bi_TransFuse_S(pretrained=True).cuda()
    # model = enhanced_TransFuse(pretrained=True).cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    # optimizer = RAdam(params, lr=opt.lr)
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    image_root_val = '{}/images/'.format(opt.val_path)
    gt_root_val = '{}/masks/'.format(opt.val_path)
    
    image_root_nonlabel = '{}/images/'.format(opt.nonlabel_path)
    gt_root_nonlabel = '{}/masks/'.format(opt.nonlabel_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=0)
    total_step = len(train_loader)
   

    # semi-supervised dataset
    nonlabel_loader = get_loader(image_root_nonlabel, gt_root_nonlabel, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=0)
    print(len(nonlabel_loader))
    val_loader = get_loader(image_root_val, gt_root_val, batchsize=opt.batchsize, trainsize=opt.trainsize, phase='val', num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    ssl_loss_list = []
    best_loss = 100000

    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # train(train_loader, model, optimizer, epoch)
        epoch, train_loss, val_loss, best_loss, ssl_loss = train(dataloaders_dict, nonlabel_loader, model, optimizer, epoch, best_loss,
                                                       opt.vat_alpha, opt.vat_eps)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        ssl_loss = ssl_loss.cpu().data.numpy() * 40.0
        ssl_loss_list.append(ssl_loss)
        epoch_list.append(epoch)
    torch.cuda.synchronize()

    elapsed_time = time.time() - start
    print(elapsed_time, 'sec.')  # training time

    # loss figure
    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, label='train_loss')
    plt.plot(epoch_list, val_loss_list, label='val_loss', linestyle="--")
    plt.plot(epoch_list, ssl_loss_list, label='ssl_loss', linestyle=":")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(left=0)
    plt.legend(loc='upper right')
    fig.savefig("fig/af_ssl2000.png")
