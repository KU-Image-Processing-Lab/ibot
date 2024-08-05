import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.TransFuse_l import TransFuse_L
from utils.dataloader import get_loader
from utils.mtl import extract_weight_method_parameters_from_args
from utils.save_parameter import save_parameter
from utils.set_seed import seed_everything
from utils.smooth_cross_entropy import SmoothCrossEntropy
from utils.utils import AvgMeter, adjust_lr, clip_gradient
from utils.weight_methods import WeightMethods

matplotlib.use("Agg")
seed_everything(42)


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


def train(
    dataloaders_dict, models, optimizer, criterion, epoch, best_loss, best2_loss, device
):
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
        loss_record2, loss_record3, loss_record4, d_loss_record = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )

        os.makedirs(f"./snapshots/{opt.train_save}/crop_images", exist_ok=True)
        os.makedirs(f"./snapshots/{opt.train_save}/crop_gts", exist_ok=True)
        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                images, gts, name = pack
                labels = torch.einsum("ijkl->i", gts) > 0

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))

                images = Variable(images).to(device)
                gts = Variable(gts).to(device)
                labels = Variable(labels).to(device)
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
                    param_d = models["Preprocess"](images) * opt.trainsize

                    # ---- crop images ----
                    cropped_images = torch.Tensor().cuda()
                    for j in range(images.size(0)):
                        crop = images[j, :, :, : int(param_d[j])]
                        crop = F.interpolate(
                            crop.unsqueeze(0),
                            size=(opt.trainsize, opt.trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                        cropped_images = torch.cat((cropped_images, crop), dim=0)

                    # # ---- save images ----
                    # for j in range(cropped_images.size(0)):
                    #     img = cropped_images[j].mul(255).byte()
                    #     img = img.permute(1, 2, 0).cpu().numpy()
                    #     cv2.imwrite(f"./fig/train_Epoch{epoch}_{name[j].replace('.png', '')}_image.png", img)

                    # ---- crop gts ----
                    cropped_gts = torch.Tensor().cuda()
                    for j in range(gts.size(0)):
                        crop = gts[j, :, :, : int(param_d[j])]
                        crop = F.interpolate(
                            crop.unsqueeze(0),
                            size=(opt.trainsize, opt.trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                        cropped_gts = torch.cat((cropped_gts, crop), dim=0)

                    # # ---- binarization gt ----
                    # lower = np.array([0, 0, 0], dtype="uint8")
                    # upper = np.array([255, 50, 255], dtype="uint8")
                    # for j in range(cropped_gts.size(0)):
                    #     img = cropped_gts[j].mul(255).byte()
                    #     img = img.permute(1, 2, 0).cpu().numpy()
                    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    #     img = cv2.inRange(img, lower, upper)
                    #     img = cv2.bitwise_not(img)
                    #     # cv2.imwrite(
                    #     #     f"./fig/train_Epoch{epoch}_{name[j].replace('.png', '')}_mask.png",
                    #     #     img,
                    #     # )
                    #     img_tensor = torch.from_numpy(img).unsqueeze(0).float()
                    #     cropped_gts[j] = img_tensor

                    # ---- forward ----
                    lateral_map_4, lateral_map_3, lateral_map_2 = models["Transfuse"](
                        cropped_images
                    )

                    # ---- loss function ----
                    loss4 = structure_loss(lateral_map_4, cropped_gts)
                    loss3 = structure_loss(lateral_map_3, cropped_gts)
                    loss2 = structure_loss(lateral_map_2, cropped_gts)
                    loss = (
                        0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4
                    )  # TODO: try different weights for loss

                    lateral_map_2 = lateral_map_2.repeat(1, 3, 1, 1)

                    d_out = models["Discriminator"](lateral_map_2)
                    d_loss = criterion(d_out, labels)
                    losses = torch.stack((loss, d_loss))
                    # ---- backward ----
                    if phase == "train":
                        loss, extra_outputs = weight_method.backward(
                            losses=losses,
                            shared_parameters=shared_parameters,
                            task_specific_parameters=task_specific_parameters,
                            # last_shared_parameters=list(model.last_shared_parameters()),
                            # representation=features,
                        )
                        clip_gradient(optimizer, opt.clip)
                        optimizer.step()

                        # ---- recording loss ----
                if rate == 1:
                    d_loss_record.update(d_loss.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
            if (i % opt.log_iter == 0 or i == total_step) and phase == "train":
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], "
                    "[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, d_loss: {:0.4f}".format(
                        datetime.now(),
                        epoch,
                        opt.epoch,
                        i,
                        total_step,
                        loss_record2.show(),
                        loss_record3.show(),
                        loss_record4.show(),
                        d_loss_record.show(),
                    )
                )
        if phase == "train":
            train_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()
            train_d_loss = d_loss_record.show()
        elif phase == "val":
            val_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()
            val_d_loss = d_loss_record.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = "snapshots/{}/".format(opt.train_save)
                torch.save(
                    models["Preprocess"].state_dict(), save_path + "Preprocess-best.pth"
                )
                torch.save(
                    models["Transfuse"].state_dict(), save_path + "Transfuse-best.pth"
                )
                torch.save(
                    models["Discriminator"].state_dict(),
                    save_path + "Discriminator-best.pth",
                )
                print("[Saving best Snapshot:]", save_path + "TransFuse-best.pth")
            if val_d_loss < best2_loss:
                best2_loss = val_d_loss
                save_path = "snapshots/{}/".format(opt.train_save)
                torch.save(
                    models["Preprocess"].state_dict(),
                    save_path + "Preprocess-best2.pth",
                )
                torch.save(
                    models["Transfuse"].state_dict(), save_path + "Transfuse-best2.pth"
                )
                torch.save(
                    models["Discriminator"].state_dict(),
                    save_path + "Discriminator-best2.pth",
                )
                print("[Saving best Snapshot:]", save_path + "Discriminator-best2.pth")

    save_path = "snapshots/{}/".format(opt.train_save)
    if (epoch + 1) % opt.save_epoch == 0:
        torch.save(
            models["Preprocess"].state_dict(), save_path + "Preprocess-%d.pth" % epoch
        )
        torch.save(
            models["Transfuse"].state_dict(), save_path + "Transfuse-%d.pth" % epoch
        )
        torch.save(
            models["Discriminator"].state_dict(),
            save_path + "Discriminator-%d.pth" % epoch,
        )

        print("[Saving Snapshot:]", save_path + "Transfuse-%d.pth" % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))

    return epoch, train_loss, val_loss, train_d_loss, val_d_loss, best_loss, best2_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=352, help="training dataset size"
    )
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
    parser.add_argument(
        "--loss_dir",
        type=str,
        default="./fig",
        help="img dir",
    )
    parser.add_argument(
        "--seg_loss_path",
        type=str,
        default="./fig/seg_loss_img.png",
        help="path to segmentation loss img",
    )
    parser.add_argument(
        "--dis_loss_path",
        type=str,
        default="./fig/dis_loss_img.png",
        help="path to Discriminator loss img",
    )
    parser.add_argument("--train_save", type=str, default="Transfuse_S")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 of adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 of adam optimizer"
    )

    parser.add_argument(
        "--tuning",
        action="store_true",
        help="セグメンテーションチューニングを適用するかどうか",
    )

    parser.add_argument("--mtl", type=str, default="nashmtl")

    parser.add_argument(
        "--discriminator",
        type=str,
        default="resnet10",
        help="Discriminatorに使うモデル",
    )
    parser.add_argument(
        "--log_iter", type=int, default=20, help="ログ出力するエポックの単位"
    )
    parser.add_argument(
        "--save_epoch", type=int, default=5, help="重み保存するエポックの単位"
    )
    parser.add_argument(
        "--mae_weight",
        type=str,
        default="./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth",
        help="MAEの重み(TransFuse)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="resnet10",
        help="前処理に使用するモデル",
    )

    opt = parser.parse_args()

    save_parameter(opt)

    # ---- build discriminator ----
    if opt.discriminator == "resnet10":
        from lib.ResNet_10 import ResNet10

        model2 = ResNet10()
    elif opt.discriminator == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        model2 = Discriminator()
    else:
        sys.exit("Invalid discriminator model")

    # ---- build preprocess model ----
    if opt.discriminator == "resnet10":
        from lib.ResNet_10_preprocess import ResNet10

        model0 = ResNet10()
    elif opt.discriminator == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        model0 = Discriminator()
    else:
        sys.exit("Invalid preprocess model")

    model1 = TransFuse_L(pretrained=True)
    if opt.tuning:
        model1.load_state_dict(torch.load(opt.mae_weight))

    if opt.tuning and opt.mtl == "stl":
        for param in model1.parameters():
            param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model0.to(device)
    model1.to(device)
    model2.to(device)

    models = {"Preprocess": model0, "Transfuse": model1, "Discriminator": model2}

    weight_methods_parameters = extract_weight_method_parameters_from_args()
    weight_method = WeightMethods(
        opt.mtl, n_tasks=2, device=device, **weight_methods_parameters[opt.mtl]
    )

    params = [p for v in models.values() for p in list(v.parameters())]

    shared_parameters = [
        p
        for n, p in model1.named_parameters()
        if "final_x" not in n and "final_1" not in n and "cls" not in n
    ]
    task_specific_parameters = [
        p
        for n, p in model1.named_parameters()
        if "final_x" in n or "final_1" in n or "cls" in n
    ]

    for n, p in model0.named_parameters():
        task_specific_parameters.append(p)
    for n, p in model2.named_parameters():
        task_specific_parameters.append(p)

    optimizer = torch.optim.Adam(
        [
            dict(params=params, lr=opt.lr, betas=(opt.beta1, opt.beta2)),
            dict(params=weight_method.parameters(), lr=0.025),
        ],
    )

    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = SmoothCrossEntropy()

    image_root = "{}/images/".format(opt.train_path)
    gt_root = "{}/masks/".format(opt.train_path)

    image_root_val = "{}/images/".format(opt.val_path)
    gt_root_val = "{}/masks/".format(opt.val_path)

    train_loader = get_loader(
        image_root,
        gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        dataset="preprocess",
    )
    total_step = len(train_loader)

    val_loader = get_loader(
        image_root_val,
        gt_root_val,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        phase="val",
        dataset="preprocess",
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    train_d_loss_list = []
    val_d_loss_list = []
    best_loss = 100000
    best2_loss = 100000

    # 学習開始時間の記録
    start_time = time.time()

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        (
            epoch,
            train_loss,
            val_loss,
            train_d_loss,
            val_d_loss,
            best_loss,
            best2_loss,
        ) = train(
            dataloaders_dict,
            models,
            optimizer,
            criterion,
            epoch,
            best_loss,
            best2_loss,
            device,
        )
        epoch_list.append(epoch)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        train_d_loss = train_d_loss.cpu().data.numpy()
        train_d_loss_list.append(train_d_loss)
        val_d_loss = val_d_loss.cpu().data.numpy()
        val_d_loss_list.append(val_d_loss)

    # 学習終了時間の記録
    end_time = time.time()

    # 学習にかかった時間の計算
    training_time = end_time - start_time
    print(f"学習にかかった時間:{training_time}秒")
    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, label="train_loss")
    plt.plot(epoch_list, val_loss_list, label="val_loss", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(left=0)
    plt.legend(loc="upper right")

    fig2 = plt.figure()
    plt.plot(epoch_list, train_d_loss_list, label="train_d_loss", linestyle=":")
    plt.plot(epoch_list, val_d_loss_list, label="val_d_loss", linestyle="-.")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim(left=0)
    plt.legend(loc="upper right")
    # 保存先ディレクトリを作成
    os.makedirs(opt.loss_dir, exist_ok=True)
    fig.savefig(opt.seg_loss_path)
    fig2.savefig(opt.dis_loss_path)
