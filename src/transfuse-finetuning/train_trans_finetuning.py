import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from lib.Discriminator_ResNet import Discriminator
from lib.TransFuse_l import TransFuse_L
from utils.dataloader import get_loader
from utils.smooth_cross_entropy import SmoothCrossEntropy
from utils.utils import AvgMeter, adjust_lr, clip_gradient


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(dataloaders_dict, models, optimizer, criterion, epoch, best_loss):
    val_loss = 0
    for phase in ["train", "val"]:
        if phase == "train":
            for model in models.values():
                model.train()
        else:
            for model in models.values():
                model.eval()

        # ---- multi-scale training ----
        size_rates = [1]
        loss_record = AvgMeter()

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
                    images = F.upsample(
                        images,
                        size=(trainsize, trainsize),
                        mode="bilinear",
                        align_corners=True,
                    )
                    gts = F.upsample(
                        gts,
                        size=(trainsize, trainsize),
                        mode="bilinear",
                        align_corners=True,
                    )
                with torch.set_grad_enabled(phase == "train"):
                    # ---- forward ----
                    lateral_map_4, lateral_map_3, lateral_map_2 = models[
                        "Segmentation"
                    ](images)

                    # TODO: try different weights for loss
                    lateral_map_2 = lateral_map_2.repeat(1, 3, 1, 1)

                    d_out = models["Discriminator"](lateral_map_2)
                    # d_out = models['Discriminator'](lateral_map_2, images)
                    loss = criterion(d_out, labels)
                    # ---- backward ----
                    if phase == "train":
                        loss.backward()
                        # clip_gradient(optimizer, opt.clip)
                        clip_gradient(optimizer, opt.clip)
                        optimizer.step()

                        # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, opt.batchsize)

            if (i % 20 == 0 or i == total_step) and phase == "train":
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}".format(
                        datetime.now(),
                        epoch,
                        opt.epoch,
                        i,
                        total_step,
                        loss_record.show(),
                    )
                )
        if phase == "train":
            train_loss = loss_record.show()
        elif phase == "val":
            val_loss = loss_record.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = "snapshots/{}/".format(opt.train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(
                    models["Discriminator"].state_dict(),
                    save_path + "Discriminator-best.pth",
                )
                print("[Saving best Snapshot:]", save_path + "Discriminator-best.pth")

    save_path = "snapshots/{}/".format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0:
        torch.save(
            models["Discriminator"].state_dict(),
            save_path + "Discriminator-%d.pth" % epoch,
        )

        print("[Saving Snapshot:]", save_path + "Discriminator-%d.pth" % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))

    return epoch, train_loss, val_loss, best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=352, help="training dataset size"
    )
    # parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--grad_norm", type=float, default=2.0, help="gradient clipping norm"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./dataset/TrainDataset",
        help="path to train dataset",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="./dataset/ValDataset",
        help="path to val dataset",
    )
    # parser.add_argument('--train_path', type=str, default='./dataset/sekkai_TrainDataset', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='./dataset/sekkai_ValDataset', help='path to val dataset')
    parser.add_argument("--train_save", type=str, default="Transfuse_S")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 of adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 of adam optimizer"
    )

    parser.add_argument("--tuning_calcification", type=bool, default=True)

    parser.add_argument("--segmentation_grad", type=bool, default=False)

    opt = parser.parse_args()

    print("Tuning_Calcification:", opt.tuning_calcification)
    if opt.segmentation_grad:
        print("Segmentation_Grad:updating")
    else:
        print("Segmentation_Grad:freezing")
    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = U_PraNet().cuda()
    # model = PraNet().cuda()

    # model2 = vit_large(pretrained=True)

    # model2 = torch_model.vgg16(pretrained=True)
    # model2.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    model1 = TransFuse_L()
    if opt.tuning_calcification:
        model1.load_state_dict(
            torch.load(
                "./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth"
            )
        )
    else:
        model1.load_state_dict(
            torch.load(
                "./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化なし含む/Transfuse-best.pth"
            )
        )

    #########################################################################
    if opt.segmentation_grad == False:
        for param in model1.parameters():
            param.requires_grad = False
    #########################################################################

    model1.cuda()

    model2 = Discriminator()

    model2 = model2.cuda()

    models = {"Segmentation": model1, "Discriminator": model2}

    params = [p for v in models.values() for p in list(v.parameters())]

    # no=1
    # for n, p in model1.named_parameters():
    #     print(no)
    #     print(n)
    #     no+=1

    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    # optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
    # optimizer = PCGrad(optimizer)

    # d_optimizer = torch.optim.Adam(params2, opt.lr)

    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = SmoothCrossEntropy()

    image_root = "{}/images/".format(opt.train_path)
    gt_root = "{}/masks/".format(opt.train_path)

    image_root_val = "{}/images/".format(opt.val_path)
    gt_root_val = "{}/masks/".format(opt.val_path)

    train_loader = get_loader(
        image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize
    )
    total_step = len(train_loader)

    val_loader = get_loader(
        image_root_val,
        gt_root_val,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        phase="val",
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    best_loss = 100000

    for epoch in range(1, opt.epoch):
        adjust_lr(
            optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch
        )  ###################################
        epoch, train_loss, val_loss, best_loss = train(
            dataloaders_dict, models, optimizer, criterion, epoch, best_loss
        )
        epoch_list.append(epoch)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)

    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, label="train_loss")
    plt.plot(epoch_list, val_loss_list, label="val_loss", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(left=0)
    plt.legend(loc="upper right")

    fig.savefig("fig/loss.png")
