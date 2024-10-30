import argparse
import os
import pathlib
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
from utils.dataloader import PolypDatasetPostprocess, get_loader
from utils.mtl import extract_weight_method_parameters_from_args
from utils.save_parameter import save_parameter
from utils.set_seed import seed_everything
from utils.smooth_cross_entropy import SmoothCrossEntropy
from utils.utils import AvgMeter, adjust_lr, clip_gradient
from utils.weight_methods import WeightMethods

matplotlib.use("Agg")
seed_everything(42)


# def paramd_d_loss(
#     d_predict: torch.Tensor, d_init: torch.Tensor, lambda_d: float
# ) -> torch.Tensor:
#     return (lambda_d * torch.abs(d_predict - d_init)).mean()


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


def save_crop_image(cropped_images: torch.Tensor, name: list, epoch: int) -> None:
    for j in range(cropped_images.size(0)):
        img = cropped_images[j].mul(255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(
            f"./fig/train_Epoch{epoch}_{name[j].replace('.png', '')}_image.png", img
        )


def binary_mask(cropped_gts: torch.Tensor, name: list, epoch: int) -> torch.Tensor:
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([255, 50, 255], dtype="uint8")

    for j in range(cropped_gts.size(0)):
        img = cropped_gts[j].mul(255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.inRange(img, lower, upper)
        img = cv2.bitwise_not(img)
        cv2.imwrite(
            f"./fig/train_Epoch{epoch}_{name[j].replace('.png', '')}_mask.png",
            img,
        )
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        cropped_gts[j] = img_tensor

    return cropped_gts


def train(**kwargs):
    dataloaders_dict = kwargs["dataloaders_dict"]
    models = kwargs["models"]
    optimizer = kwargs["optimizer"]
    criterion = kwargs["criterion"]
    epoch = kwargs["epoch"]
    best_loss = kwargs["best_loss"]
    best2_loss = kwargs["best2_loss"]
    device = kwargs["device"]
    opt = kwargs["opt"]
    total_step = kwargs["total_step"]
    weight_method = kwargs["weight_method"]
    shared_parameters = kwargs["shared_parameters"]
    task_specific_parameters = kwargs["task_specific_parameters"]
    postprocess = kwargs["postprocess"]
    del kwargs

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
        loss_record2, loss_record3, loss_record4, d_loss_record, p_loss_record = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )

        for i, pack in enumerate(dataloaders_dict[phase], start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                images, gts, name, image_width = pack
                labels = torch.einsum("ijkl->i", gts) > 0

                labels = torch.where(labels > 0, torch.tensor(1), torch.tensor(0))

                images = Variable(images).to(device)
                gts = Variable(gts).to(device)
                labels = Variable(labels).to(device)
                image_width = image_width.to(device)

                # ---- rescale ----
                # trainsize = int(round(opt.trainsize * rate / 32) * 32)
                # if rate != 1:
                #     images = F.upsample(
                #         images,
                #         size=(trainsize, trainsize),
                #         mode="bilinear",
                #         align_corners=True,
                #     )
                #     gts = F.upsample(
                #         gts,
                #         size=(trainsize, trainsize),
                #         mode="bilinear",
                #         align_corners=True,
                #     )

                with torch.set_grad_enabled(phase == "train"):
                    d_predict = models["Preprocess"](images)

                    # p_loss = paramd_d_loss(
                    #     d_predict=d_predict, d_init=opt.d_init, lambda_d=opt.lambda_d
                    # )

                    height, width = images.shape[2:]
                    crop_d = (
                        d_predict.unsqueeze(1) * width
                        + (1 - d_predict.unsqueeze(1)) * height
                    ).long()

                    # ---- crop images ----
                    cropped_images = torch.Tensor().cuda()
                    for j in range(images.size(0)):
                        crop = images[j, :, :, : int(crop_d[j])]
                        crop = F.interpolate(
                            crop.unsqueeze(0),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=True,
                        )
                        cropped_images = torch.cat((cropped_images, crop), dim=0)

                    # save_crop_image(cropped_images, name, epoch)

                    # ---- crop gts ----
                    cropped_gts = torch.Tensor().cuda()
                    for j in range(gts.size(0)):
                        crop = gts[j, :, :, : int(crop_d[j])]
                        crop = F.interpolate(
                            crop.unsqueeze(0),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=True,
                        )
                        cropped_gts = torch.cat((cropped_gts, crop), dim=0)

                    # cropped_gts = binary_mask(cropped_gts, name, epoch)

                    cropped_images, cropped_gts = postprocess(
                        cropped_images, cropped_gts
                    )

                    # # ---- save image ----
                    # os.makedirs("./fig/train_start", exist_ok=True)
                    # for j in range(images.size(0)):
                    #     img = images[j].mul(255).byte()
                    #     img = img.permute(1, 2, 0).cpu().numpy()
                    #     cv2.imwrite(
                    #         f"./fig/train_start/{name[j].replace('.png', '')}_image.png",
                    #         img,
                    #     )

                    # # ---- save image ----
                    # os.makedirs("./fig/train_end", exist_ok=True)
                    # for j in range(cropped_images.size(0)):
                    #     img = cropped_images[j].mul(255).byte()
                    #     img = img.permute(1, 2, 0).cpu().numpy()
                    #     cv2.imwrite(
                    #         f"./fig/train_end/{name[j].replace('.png', '')}_image.png",
                    #         img,
                    #     )

                    # ---- forward ----
                    lateral_map_4, lateral_map_3, lateral_map_2 = models["TransFuse"](
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
                    # p_loss_record.update(p_loss.data, opt.batchsize)
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
                        # p_loss_record.show(),
                    )
                )
                print("mean d: {:.4f}".format(d_predict.mean().cpu().data.numpy()))
        if phase == "train":
            train_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()
            train_d_loss = d_loss_record.show()
            # train_p_loss = p_loss_record.show()
        elif phase == "val":
            val_loss = loss_record2.show() + loss_record3.show() + loss_record4.show()
            val_d_loss = d_loss_record.show()
            # val_p_loss = p_loss_record.show()
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = "snapshots/{}/".format(opt.train_save)
                torch.save(
                    models["Preprocess"].state_dict(), save_path + "Preprocess-best.pth"
                )
                torch.save(
                    models["TransFuse"].state_dict(), save_path + "TransFuse-best.pth"
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
                    models["TransFuse"].state_dict(), save_path + "TransFuse-best2.pth"
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
            models["TransFuse"].state_dict(), save_path + "TransFuse-%d.pth" % epoch
        )
        torch.save(
            models["Discriminator"].state_dict(),
            save_path + "Discriminator-%d.pth" % epoch,
        )

        print("[Saving Snapshot:]", save_path + "TransFuse-%d.pth" % epoch)
    print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(train_loss, val_loss))
    print("train_d_loss: {0:.4f}, val_d_loss: {1:.4f}".format(train_d_loss, val_d_loss))
    # print("train_p_loss: {0:.4f}, val_p_loss: {1:.4f}".format(train_p_loss, val_p_loss))

    output_dict = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_d_loss": train_d_loss,
        "val_d_loss": val_d_loss,
        # "train_p_loss": train_p_loss,
        # "val_p_loss": val_p_loss,
        "best_loss": best_loss,
        "best2_loss": best2_loss,
    }

    return output_dict


def get_discriminator(opt: argparse.Namespace) -> torch.nn.Module:
    if opt.discriminator == "resnet10":
        from lib.ResNet_10 import ResNet10

        return ResNet10()
    elif opt.discriminator == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        return Discriminator()
    else:
        sys.exit("Invalid discriminator model")


def get_preprocess_model(opt: argparse.Namespace) -> torch.nn.Module:
    if opt.preprocess == "resnet10":
        from lib.ResNet_10_preprocess import ResNet10

        return ResNet10(
            d_init=opt.d_init, mean=opt.mean, std=opt.std, init_type=opt.init_type
        )
    elif opt.preprocess == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        return Discriminator()
    else:
        sys.exit("Invalid preprocess model")


def arg_parser() -> argparse.Namespace:
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
    parser.add_argument(
        "--pre_loss_path",
        type=str,
        default="./fig/pre_loss_img.png",
        help="path to Preprocessing loss img",
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
    parser.add_argument(
        "--lambda_d", type=float, default=0.01, help="パラメータdの損失の係数"
    )
    parser.add_argument(
        "--d_init", type=float, default=0.625, help="パラメータdの初期値"
    )
    parser.add_argument(
        "--mean", type=float, default=0, help="ResNet10_preprocessの出力層の重みの平均"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.001,
        help="ResNet10_preprocessの出力層の重みの標準偏差",
    )
    parser.add_argument(
        "--init_type", type=str, default="default", help="出力層の初期化の種類"
    )
    return parser.parse_args()


def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_models(opt: argparse.Namespace, device: torch.device) -> dict:
    model0 = get_preprocess_model(opt)
    model1 = TransFuse_L(pretrained=True)
    model2 = get_discriminator(opt)

    if opt.tuning:
        model1.load_state_dict(torch.load(opt.mae_weight))

    if opt.tuning and opt.mtl == "stl":
        for param in model1.parameters():
            param.requires_grad = False

    model0.to(device)
    model1.to(device)
    model2.to(device)

    models = {"Preprocess": model0, "TransFuse": model1, "Discriminator": model2}

    return models


def set_mtl_parameters(
    models: dict, opt: argparse.Namespace, device: torch.device
) -> dict:
    weight_methods_parameters = extract_weight_method_parameters_from_args()
    weight_method = WeightMethods(
        opt.mtl, n_tasks=2, device=device, **weight_methods_parameters[opt.mtl]
    )

    params = [p for v in models.values() for p in list(v.parameters())]

    shared_parameters = [
        p
        for n, p in models["TransFuse"].named_parameters()
        if "final_x" not in n and "final_1" not in n and "cls" not in n
    ]
    task_specific_parameters = [
        p
        for n, p in models["TransFuse"].named_parameters()
        if "final_x" in n or "final_1" in n or "cls" in n
    ]

    # for _, p in models["Preprocess"].named_parameters():
    #     task_specific_parameters.append(p)
    for _, p in models["Discriminator"].named_parameters():
        task_specific_parameters.append(p)

    output_dict = {
        "weight_method": weight_method,
        "params": params,
        "shared_parameters": shared_parameters,
        "task_specific_parameters": task_specific_parameters,
    }
    return output_dict


def plot_loss(
    epoch_list,
    train_loss_list,
    val_loss_list,
    train_d_loss_list,
    val_d_loss_list,
    opt: argparse.Namespace,
):
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


def main():
    opt = arg_parser()
    save_parameter(opt)

    device = set_device()
    models = get_models(opt, device)

    mtl_parameter = set_mtl_parameters(models, opt, device)
    weight_method = mtl_parameter["weight_method"]
    params = mtl_parameter["params"]
    shared_parameters = mtl_parameter["shared_parameters"]
    task_specific_parameters = mtl_parameter["task_specific_parameters"]

    optimizer = torch.optim.Adam(
        [
            dict(params=params, lr=opt.lr, betas=(opt.beta1, opt.beta2)),
            dict(params=weight_method.parameters(), lr=0.025),
        ],
    )

    criterion = SmoothCrossEntropy()

    train_loader_preprocess = get_loader(
        pathlib.Path(opt.train_path) / "images",
        pathlib.Path(opt.train_path) / "masks",
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        phase="train",
        dataset="preprocess",
    )
    total_step = len(train_loader_preprocess)

    val_loader_preprocess = get_loader(
        pathlib.Path(opt.val_path) / "images",
        pathlib.Path(opt.val_path) / "masks",
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        phase="val",
        dataset="preprocess",
    )

    dataloaders_dict = {"train": train_loader_preprocess, "val": val_loader_preprocess}

    postprocess = PolypDatasetPostprocess(trainsize=opt.trainsize, phase="train")

    print("#" * 20, "Start Training", "#" * 20)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    train_d_loss_list = []
    val_d_loss_list = []
    train_p_loss_list = []
    val_p_loss_list = []
    best_loss = 100000
    best2_loss = 100000

    # 学習開始時間の記録
    start_time = time.time()

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        input_dict = {
            "dataloaders_dict": dataloaders_dict,
            "models": models,
            "optimizer": optimizer,
            "criterion": criterion,
            "epoch": epoch,
            "best_loss": best_loss,
            "best2_loss": best2_loss,
            "device": device,
            "opt": opt,
            "total_step": total_step,
            "weight_method": weight_method,
            "shared_parameters": shared_parameters,
            "task_specific_parameters": task_specific_parameters,
            "postprocess": postprocess,
        }
        output = train(**input_dict)
        best_loss = output["best_loss"]
        best2_loss = output["best2_loss"]
        train_loss = output["train_loss"]
        val_loss = output["val_loss"]
        train_d_loss = output["train_d_loss"]
        val_d_loss = output["val_d_loss"]
        # train_p_loss = output["train_p_loss"]
        # val_p_loss = output["val_p_loss"]

        epoch_list.append(epoch)
        train_loss = train_loss.cpu().data.numpy()
        train_loss_list.append(train_loss)
        val_loss = val_loss.cpu().data.numpy()
        val_loss_list.append(val_loss)
        train_d_loss = train_d_loss.cpu().data.numpy()
        train_d_loss_list.append(train_d_loss)
        val_d_loss = val_d_loss.cpu().data.numpy()
        val_d_loss_list.append(val_d_loss)

        if epoch % 10 == 0:
            try:
                plot_loss(
                    epoch_list,
                    train_loss_list,
                    val_loss_list,
                    train_d_loss_list,
                    val_d_loss_list,
                    opt,
                )
            except Exception as e:
                print(f"Error in plot_loss: {e}")

    # 学習終了時間の記録
    end_time = time.time()

    # 学習にかかった時間の計算
    training_time = end_time - start_time
    print(f"学習にかかった時間:{training_time}秒")


if __name__ == "__main__":
    main()
