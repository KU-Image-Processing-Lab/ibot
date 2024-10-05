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
import torchinfo
from skimage import img_as_ubyte
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torchvision import transforms

# # from lib.models_vit_discriminator import vit_large_patch16 as vit_large
from lib.TransFuse_l import TransFuse_L
from utils.dataloader import test_dataset
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


def get_model_scale(model: torch.nn.Module) -> None:
    torchinfo.summary(model, input_size=(1, 3, 352, 352))


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
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
parser.add_argument("--test_data_path1", type=str, default="./dataset/TestDataset/")
parser.add_argument(
    "--test_data_path2", type=str, default="./dataset/sekkai_TestDataset/"
)
parser.add_argument("--test_save_path", type=str, default="./results/Transfuse_S/")


opt = parser.parse_args()
# for ep in range(opt.epoch):

data_path1 = opt.test_data_path1
data_path2 = opt.test_data_path2
save_path = opt.test_save_path
os.makedirs(save_path, exist_ok=True)

model = TransFuse_L()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

print("model1")
get_model_scale(model)

# model2 = vit_large(pretrained=pretrained)

# model2 = torch_model.vgg16(pretrained=True)
# model2.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# ---- build models ----
if opt.discriminator == "resnet10":
    from lib.ResNet_10 import ResNet10

    model2 = ResNet10()

if opt.discriminator == "res2net":
    from lib.Discriminator_ResNet import Discriminator

    model2 = Discriminator()

model2.load_state_dict(torch.load(opt.pth_path2))

model2.cuda()
model2.eval()

print("model2")
get_model_scale(model2)

os.makedirs(save_path, exist_ok=True)
for file in glob.glob(save_path + "/*.png"):
    os.remove(file)
os.makedirs(save_path + "/TP", exist_ok=True)
for file in glob.glob(save_path + "/TP/*.png"):
    os.remove(file)
os.makedirs(save_path + "/FN", exist_ok=True)
for file in glob.glob(save_path + "/FN/*.png"):
    os.remove(file)
os.makedirs(save_path + "/FP", exist_ok=True)
for file in glob.glob(save_path + "/FP/*.png"):
    os.remove(file)
os.makedirs(save_path + "/TN", exist_ok=True)
for file in glob.glob(save_path + "/TN/*.png"):
    os.remove(file)
image_root1 = "{}/images/".format(data_path1)
gt_root1 = "{}/masks/".format(data_path1)
test_loader1 = test_dataset(image_root1, gt_root1, opt.testsize)

image_root2 = "{}/images/".format(data_path2)
gt_root2 = "{}/masks/".format(data_path2)
test_loader2 = test_dataset(image_root2, gt_root2, opt.testsize)

dice_bank = []
iou_bank = []
acc_bank = []

dice_dict = {}
iou_dict = {}
acc_dict = {}

y_true = np.array([])
y_score = np.array([])
y_pred = np.array([])

output_dict = {}

for i in range(test_loader1.size):
    image, gt, name = test_loader1.load_data()
    label = transforms.functional.to_tensor(gt)
    label = torch.einsum("ijk->i", label) > 0
    label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
    gt = np.asarray(gt, np.float32)

    # gt /= (gt.max() + 1e-8)  ##########################

    image = image.cuda()

    with torch.no_grad():
        _, _, res = model(image)

        res1 = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        # print(torch.max(res),torch.min(res),torch.mean(res))
        # print(res1.max(),res1.min(),res1.mean())

        # res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)  ############################
        res1 = 1.0 * (res1 > 0.5)
        # print(res1.max(),res1.min(),res1.mean())

        imageio.imsave(save_path + name, img_as_ubyte(res1))
        res = res.repeat(1, 3, 1, 1)

        # res = res.sigmoid()  #########################
        # res = 1. * (res > 0.5)  ############################
        # res = res * image

        out = model2(res)
        # out = model2(res, image)

        # predicted = torch.round(out)
        _, predicted = torch.max(out, dim=1)
        predicted = torch.Tensor.cpu(predicted).detach().numpy()

        label = torch.Tensor.cpu(label).detach().numpy()
        out = torch.Tensor.cpu(out).detach().numpy()
        y_true = np.append(y_true, label)
        y_score = np.append(y_score, out[0][1] - out[0][0])
        y_pred = np.append(y_pred, predicted)

    if label == 1:
        if predicted == 1:
            imageio.imsave(save_path + "/TP/" + name, img_as_ubyte(res1))

        else:
            imageio.imsave(save_path + "/FN/" + name, img_as_ubyte(res1))

    else:
        if predicted == 1:
            imageio.imsave(save_path + "/FP/" + name, img_as_ubyte(res1))

        else:
            imageio.imsave(save_path + "/TN/" + name, img_as_ubyte(res1))

for i in range(test_loader2.size):
    image, gt, name = test_loader2.load_data()
    gt = np.asarray(gt, np.float32)

    gt = 1.0 * (gt > 0.5)  ########################

    image = image.cuda()

    with torch.no_grad():
        _, _, res = model(image)

    res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    # res = (res - res.min()) / (res.max() - res.min() + 1e-8)  ############################
    res = 1.0 * (res > 0.5)  ############################

    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)

    acc_dict[name] = acc
    dice_dict[name] = dice
    iou_dict[name] = iou

# ---- calculate the confusion matrix ----
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.flatten()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

output_dict["Discriminator"] = opt.discriminator
output_dict["pth_path"] = opt.pth_path
output_dict["pth_path2"] = opt.pth_path2

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
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
with open(Path(opt.test_save_path) / f"{now_str}_result.csv", "w") as f:
    f.write("name, value\n")
    for k, v in output_dict.items():
        print(k, v)
        f.write(f"{k},{v}\n")

# ---- save the acc, dice, iou as datframe ----
import pandas as pd

df = pd.DataFrame(
    {
        "acc": acc_dict,
        "dice": dice_dict,
        "iou": iou_dict,
    }
)
df.to_csv(Path(opt.test_save_path) / f"{now_str}_evaluation.csv")
