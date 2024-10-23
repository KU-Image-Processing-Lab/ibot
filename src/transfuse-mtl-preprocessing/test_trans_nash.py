import argparse
import datetime
import glob
import os
import sys
from pathlib import Path

import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from lib.TransFuse_l import TransFuse_L
from utils.dataloader import test_dataset, test_dataset_preprocess
from utils.set_seed import seed_everything

matplotlib.use("Agg")
seed_everything(42)

#################################################
# epoch1~20の重みを全部テストする。

k = 5
thresh = 160
lower_size = 300
upper_size = 43264
total = 0
correct = 0


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = 0.001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=352, help="testing size")
    parser.add_argument(
        "--pth_path0", type=str, default="./snapshots/Transfuse_S/Preprocess-99.pth"
    )
    parser.add_argument(
        "--pth_path", type=str, default="./snapshots/Transfuse_S/Transfuse-99.pth"
    )
    parser.add_argument(
        "--pth_path2", type=str, default="./snapshots/Transfuse_S/Discriminator-99.pth"
    )
    parser.add_argument(
        "--discriminator",
        type=str,
        default="resnet10",
        help="Discriminatorに使うモデル",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="resnet10",
        help="前処理に使用するモデル",
    )
    parser.add_argument(
        "--test_data_path1",
        type=str,
        default="./dataset/original_dataset_only_left/TestDataset/",
    )
    parser.add_argument(
        "--test_data_path2",
        type=str,
        default="./dataset/original_dataset_only_left/sekkai_TestDataset/",
    )
    parser.add_argument("--save_path", type=str, default="./results/Transfuse_S/")

    return parser.parse_args()


def get_preprocess_model(opt: argparse.Namespace) -> torch.nn.Module:
    if opt.preprocess == "resnet10":
        from lib.ResNet_10_preprocess import ResNet10

        return ResNet10()
    elif opt.preprocess == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        return Discriminator()
    else:
        sys.exit("Invalid preprocess model")


def get_discriminator(opt: argparse.Namespace) -> torch.nn.Module:
    if opt.discriminator == "resnet10":
        from lib.ResNet_10 import ResNet10

        return ResNet10()
    elif opt.discriminator == "res2net":
        from lib.Discriminator_ResNet import Discriminator

        return Discriminator()
    else:
        sys.exit("Invalid discriminator model")


def get_models_for_test(opt: argparse.Namespace, device: torch.device) -> dict:
    model0 = get_preprocess_model(opt)
    model1 = TransFuse_L()
    model2 = get_discriminator(opt)

    # if opt.tuning and opt.mtl == "stl":
    #     for param in model1.parameters():
    #         param.requires_grad = False

    model0.load_state_dict(torch.load(opt.pth_path0))
    model0.cuda()
    model0.eval()

    model1.load_state_dict(torch.load(opt.pth_path))
    model1.cuda()
    model1.eval()

    model2.load_state_dict(torch.load(opt.pth_path2))
    model2.cuda()
    model2.eval()

    models = {"Preprocess": model0, "Transfuse": model1, "Discriminator": model2}

    return models


def clean_save_dir(save_path: Path):
    os.makedirs(save_path, exist_ok=True)
    for file in glob.glob(f"{save_path}/*.png"):
        os.remove(file)
    os.makedirs(save_path / "TP", exist_ok=True)
    for file in glob.glob(f"{save_path}/TP/*.png"):
        os.remove(file)
    os.makedirs(save_path / "FN", exist_ok=True)
    for file in glob.glob(f"{save_path}/FN/*.png"):
        os.remove(file)
    os.makedirs(save_path / "FP", exist_ok=True)
    for file in glob.glob(f"{save_path}/FP/*.png"):
        os.remove(file)
    os.makedirs(save_path / "TN", exist_ok=True)
    for file in glob.glob(f"{save_path}/TN/*.png"):
        os.remove(file)


def get_preprocess_loader(
    opt: argparse.Namespace, data_path: Path
) -> test_dataset_preprocess:
    image_root = data_path / "images"
    gt_root = data_path / "masks"
    return test_dataset_preprocess(image_root, gt_root, opt.testsize)


def get_mtl_loader(opt: argparse.Namespace, data_path: Path) -> test_dataset:
    image_root = data_path / "images"
    gt_root = data_path / "masks"
    return test_dataset(image_root, gt_root, opt.testsize)


def main():
    opt = arg_parse()

    data_path1 = Path(opt.test_data_path1)
    data_path2 = Path(opt.test_data_path2)
    save_path = Path(opt.save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path / "TP", exist_ok=True)
    os.makedirs(save_path / "FN", exist_ok=True)
    os.makedirs(save_path / "FP", exist_ok=True)
    os.makedirs(save_path / "TN", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = get_models_for_test(opt, device)

    preprocess_loader1 = get_preprocess_loader(opt, data_path1)
    preprocess_loader2 = get_preprocess_loader(opt, data_path2)

    mtl_loader1 = get_mtl_loader(opt, data_path1)
    mtl_loader2 = get_mtl_loader(opt, data_path2)

    dice_bank = []
    iou_bank = []
    acc_bank = []

    y_true = np.array([])
    y_score = np.array([])
    y_pred = np.array([])

    output_dict = {}
    param_list = []

    for _ in range(preprocess_loader1.size):
        image, gt, name, image_width = preprocess_loader1.load_data()
        label = torch.einsum("ijk->i", gt) > 0
        label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
        gt = np.asarray(gt, np.float32)

        # save image
        os.makedirs("./fig/source", exist_ok=True)
        img = image.mul(255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(f"./fig/source/{name.replace('.png', '')}_image.png", img)

        image = image.cuda()
        # image_width = image_width.cuda()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            d_predict = models["Preprocess"](image)
            param_list.append(float(d_predict))
            print(f"{float(d_predict) = }")

            height, width = image.shape[2:]
            crop_d = int(d_predict * width + (1 - d_predict) * height)

            # ---- crop image ----
            cropped_image = torch.Tensor().cuda()
            for j in range(image.size(0)):
                crop = image[j, :, :, : int(crop_d)]
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                cropped_image = torch.cat((cropped_image, crop), dim=0)

            # ---- save image ----
            os.makedirs("./fig/cropped", exist_ok=True)
            for j in range(cropped_image.size(0)):
                img = cropped_image[j].mul(255).byte()
                img = img.permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(
                    f"./fig/cropped/{name.replace('.png', '')}_image_{j}.png", img
                )

            # ---- crop gt ----
            cropped_gt = torch.Tensor()
            gt = torch.Tensor(gt)
            for j in range(gt.size(0)):
                crop = gt[j, :, : int(crop_d)]
                crop = F.interpolate(
                    crop.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                cropped_gt = torch.cat((cropped_gt, crop), dim=0)

            # # ---- binarization gt ----
            # lower = np.array([0, 0, 0], dtype="uint8")
            # upper = np.array([255, 50, 255], dtype="uint8")
            # for j in range(cropped_gt.size(0)):
            #     img = cropped_gt[j].mul(255).byte()
            #     img = img.permute(1, 2, 0).cpu().numpy()
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #     img = cv2.inRange(img, lower, upper)
            #     img = cv2.bitwise_not(img)
            #     cv2.imwrite(f"./fig/{name.replace('.png', '')}_mask.png", img)
            #     img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            #     cropped_gt[j] = img_tensor

            cropped_image, cropped_gt = mtl_loader1.transform_process(
                cropped_image, cropped_gt
            )
            cropped_image = cropped_image.unsqueeze(0)

            # ---- save image ----
            os.makedirs("./fig/end", exist_ok=True)
            for j in range(cropped_image.size(0)):
                img = cropped_image[j].mul(255).byte()
                img = img.permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(f"./fig/end/{name.replace('.png', '')}_image_{j}.png", img)

            _, _, res = models["Transfuse"](cropped_image)

            res1 = F.upsample(
                res,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            res1 = res1.sigmoid().data.cpu().numpy().squeeze()

            res1 = 1.0 * (res1 > 0.5)  ############################

            imageio.imsave(str(save_path / name), img_as_ubyte(res1))
            res = res.repeat(1, 3, 1, 1)

            out = models["Discriminator"](res)

            _, predicted = torch.max(out, dim=1)
            predicted = torch.Tensor.cpu(predicted).detach().numpy()

            label = torch.Tensor.cpu(label).detach().numpy()
            out = torch.Tensor.cpu(out).detach().numpy()
            y_true = np.append(y_true, label)
            y_score = np.append(y_score, out[0][1] - out[0][0])
            y_pred = np.append(y_pred, predicted)

        if label == 1:
            if predicted == 1:
                imageio.imsave(save_path / "TP" / name, img_as_ubyte(res1))

            else:
                imageio.imsave(save_path / "FN" / name, img_as_ubyte(res1))

        else:
            if predicted == 1:
                imageio.imsave(save_path / "FP" / name, img_as_ubyte(res1))

            else:
                imageio.imsave(save_path / "TN" / name, img_as_ubyte(res1))

    output_dict["param_d_min"] = min(param_list)
    output_dict["param_d_max"] = max(param_list)
    output_dict["param_d_mean"] = sum(param_list) / len(param_list)
    output_dict["param_d_median"] = sorted(param_list)[len(param_list) // 2]

    for _ in range(preprocess_loader2.size):
        image, gt, name, image_width = preprocess_loader2.load_data()
        gt = np.asarray(gt, np.float32)

        gt = 1.0 * (gt > 0.5)  ########################

        image = image.cuda()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            d_predict = models["Preprocess"](image)
            print(f"{float(d_predict) = }")

            height, width = image.shape[2:]
            crop_d = int(d_predict * width + (1 - d_predict) * height)

            # cropped_image = crop_image_func.crop(crop_d, image, name)
            # crop_image_func.save()

            # cropped_gt = crop_gt_func.crop(crop_d, gt, name)
            # crop_gt_func.save()

            # ---- crop image ----
            cropped_image = torch.Tensor().cuda()
            for j in range(image.size(0)):
                crop = image[j, :, :, : int(crop_d)]
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                cropped_image = torch.cat((cropped_image, crop), dim=0)

            # ---- crop gt ----
            cropped_gt = torch.Tensor()
            gt = torch.Tensor(gt)
            for j in range(gt.size(0)):
                crop = gt[j, :, : int(crop_d)]
                crop = F.interpolate(
                    crop.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
                cropped_gt = torch.cat((cropped_gt, crop), dim=0)

            cropped_image, cropped_gt = mtl_loader2.transform_process(
                cropped_image, cropped_gt
            )
            cropped_image = cropped_image.unsqueeze(0)
            _, _, res = models["Transfuse"](cropped_image)

        res = F.upsample(
            res, size=(opt.testsize, opt.testsize), mode="bilinear", align_corners=False
        )
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)  ############################
        res = 1.0 * (res > 0.5)  ############################

        cropped_gt = cropped_gt.cpu().numpy()
        dice = mean_dice_np(cropped_gt, res)
        iou = mean_iou_np(cropped_gt, res)
        acc = np.sum(res == cropped_gt) / (res.shape[0] * res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    # ---- calculate the confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.flatten()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    output_dict["Discriminator"] = opt.discriminator
    output_dict["pth_path"] = str(opt.pth_path)
    output_dict["pth_path2"] = str(opt.pth_path2)

    output_dict["Dice"] = np.mean(dice_bank)
    output_dict["IoU"] = np.mean(iou_bank)
    output_dict["Acc"] = np.mean(acc_bank)

    output_dict["TPR-FPR"] = TPR - FPR
    output_dict["TP"] = TP
    output_dict["FN"] = FN
    output_dict["FP"] = FP
    output_dict["TN"] = TN

    if TP != 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        F_measure = f1_score(y_true, y_pred)
        specificity = TN / (TN + FP)

        output_dict["Accuracy"] = accuracy
        output_dict["F-measure"] = F_measure
        output_dict["precision"] = precision
        output_dict["recall"] = recall
        output_dict["specificity"] = specificity

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR: False positive rate")
    plt.ylabel("TPR: True positive rate")
    plt.grid()
    plt.savefig("./fig/roc_curve.png")

    AUC = roc_auc_score(y_true, y_score)
    output_dict["AUC"] = AUC

    # ---- calculate the confusion matrix with cutoff ----
    Youden_index_candidates = tpr - fpr
    index = np.where(Youden_index_candidates == max(Youden_index_candidates))[0][0]
    cutoff = thresholds[index]
    y_pred_cutoff = (y_score >= cutoff).astype(int)

    cm = confusion_matrix(y_true, y_pred_cutoff)
    TN, FP, FN, TP = cm.flatten()

    output_dict["TPR-FPR_cutoff"] = max(Youden_index_candidates)
    output_dict["Threshold"] = cutoff
    output_dict["TP_cutoff"] = TP
    output_dict["FN_cutoff"] = FN
    output_dict["FP_cutoff"] = FP
    output_dict["TN_cutoff"] = TN

    if TP != 0:
        accuracy = accuracy_score(y_true, y_pred_cutoff)
        precision = precision_score(y_true, y_pred_cutoff)
        recall = recall_score(y_true, y_pred_cutoff)
        F_measure = f1_score(y_true, y_pred_cutoff)
        specificity = TN / (TN + FP)

        output_dict["Accuracy_cutoff"] = accuracy
        output_dict["F-measure_cutoff"] = F_measure
        output_dict["precision_cutoff"] = precision
        output_dict["recall_cutoff"] = recall
        output_dict["specificity_cutoff"] = specificity

    # ---- print the results ----
    for k, v in output_dict.items():
        print(k, v)

    # ---- save the results ----
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    # columns, valuesの2列にしてcsvに保存
    with open(Path(save_path).joinpath(f"result_{now}.csv"), "a") as f:
        f.write("columns,values\n")
        for k, v in output_dict.items():
            f.write(f"{k},{v}\n")


if __name__ == "__main__":
    main()
