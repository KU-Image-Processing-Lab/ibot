import argparse
import glob
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_model
from scipy import misc
from skimage import img_as_ubyte
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torchvision import transforms

from lib.Discriminator_ResNet import Discriminator
from lib.ResNet_18 import ResNet18
from lib.Trans_CaraNet_nash import Trans_CaraNet
from utils.dataloader import test_dataset

# from lib.TransFuse_l import TransFuse_L
# from lib.models_vit_discriminator import vit_large_patch16 as vit_large

# from lib.Discriminator_ResNet import Discriminator

# from lib.Trans_CaraNet import Trans_CaraNet_L



# import sys
# sys.path.append('../../git/laboratory/python/py/tools/')
# from slack_bot import Slack_bot

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


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/Transfuse-99.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/Transfuse-59.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/Transfuse-24.pth')
parser.add_argument(
    "--pth_path",
    type=str,
    default="./snapshots/NASHRes2/TransCara/re/Transfuse-best2.pth",
)
# parser.add_argument('--pth_path', type=str, default='./snapshots/STLT/Transfuse-4.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/mtl/nash/Tuning/TransCaraNet/Transfuse-best.pth')

# parser.add_argument('--pth_path2', type=str, default='./snapshots/Transfuse_S/Discriminator-99.pth')
# parser.add_argument('--pth_path2', type=str, default='./snapshots/Transfuse_S/Discriminator-59.pth')
# parser.add_argument('--pth_path2', type=str, default='./snapshots/Transfuse_S/Discriminator-24.pth')
parser.add_argument(
    "--pth_path2",
    type=str,
    default="./snapshots/NASHRes2/TransCara/re/Discriminator-best2.pth",
)
# parser.add_argument('--pth_path2', type=str, default='./snapshots/STLT/Discriminator-4.pth')
# parser.add_argument('--pth_path2', type=str, default='./weights/修論/mtl/nash/Tuning/TransCaraNet/Discriminator-best.pth')
parser.add_argument(
    "--test_path1",
    type=str,
    default="./dataset/TestDataset",
    help="path to test dataset",
)
parser.add_argument(
    "--test_path2",
    type=str,
    default="./dataset/sekkai_TestDataset",
    help="path to test dataset",
)
# parser.add_argument('--save_path', type=str, default='./fig/nashB4/TransCara/', help='path to result')
# parser.add_argument('--save_path', type=str, default='./fig/cus/', help='path to result')
# parser.add_argument('--save_path2', type=str, default='./fig/nashB4/TransCara/tensor/', help='path to result')

opt = parser.parse_args()

data_path1 = opt.test_path1
data_path2 = opt.test_path2

# save_path = opt.save_path
# save_path2 = opt.save_path2

model = Trans_CaraNet()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()


# model2 = vit_large(pretrained=pretrained)
# model2 = torch_model.vgg16(pretrained=True)
# model2.classifier[6] = nn.Linear(in_features=4096, out_features=2)

model2 = Discriminator()
model2.load_state_dict(torch.load(opt.pth_path2))
model2.cuda()
model2.eval()

# os.makedirs(save_path, exist_ok=True)
for file in glob.glob("./fig/*.png"):
    os.remove(file)
os.makedirs("./fig/TP", exist_ok=True)
for file in glob.glob("./fig/TP/*.png"):
    os.remove(file)
os.makedirs("./fig/FN", exist_ok=True)
for file in glob.glob("./fig/FN/*.png"):
    os.remove(file)
os.makedirs("./fig/FP", exist_ok=True)
for file in glob.glob("./fig/FP/*.png"):
    os.remove(file)
os.makedirs("./fig/TN", exist_ok=True)
for file in glob.glob("./fig/TN/*.png"):
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

y_true = np.array([])
y_score = np.array([])
y_pred = np.array([])

for i in range(test_loader1.size):
    image, gt, name = test_loader1.load_data()
    label = transforms.functional.to_tensor(gt)
    label = torch.einsum("ijk->i", label) > 0
    label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
    gt = np.asarray(gt, np.float32)

    # gt /= (gt.max() + 1e-8)  ##########################

    image = image.cuda()

    with torch.no_grad():
        _, _, _, res = model(image)

        # res1 = F.upsample(res, size=gt.shape,
        #                   mode='bilinear', align_corners=False) #しきい値をまたぐ処理の場合必要
        # res1 = res1.sigmoid().data.cpu().numpy().squeeze()#しきい値をまたぐ処理の場合必要
        # print(torch.max(res),torch.min(res),torch.mean(res))
        # print(res1.max(),res1.min(),res1.mean())

        # res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)  ############################
        # res1 = 1. * (res1 > 0.5)#しきい値をまたぐ処理の場合必要
        # print(res1.max(),res1.min(),res1.mean())

        # imageio.imsave(save_path + name, img_as_ubyte(res1))#しきい値をまたぐ処理の場合必要
        res_cpu = res.cpu()  #
        res_cpu_clipped = torch.clamp(res_cpu, -1.0, 1.0)  #
        image_2d = res_cpu_clipped.squeeze()  #
        # print(image_2d.shape)
        # imageio.imsave(save_path + name, img_as_ubyte(image_2d))
        # print(image_2d.shape)
        res = res.repeat(1, 3, 1, 1)

        # torch.save(res, save_path2 + name + ".pt")
        # print(res.shape)

        # res = res.sigmoid()  #########################
        # res = 1. * (res > 0.5)  ############################
        # res = res * image

        out = model2(res)  #
        # out = model2(res1)
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
            res_cpu = res.cpu()  #
            res_cpu_clipped = torch.clamp(res_cpu, -1.0, 1.0)  #
            image_2d = res_cpu_clipped.squeeze()  #
            image_2d_permuted = image_2d.permute(1, 2, 0)  #
            # imageio.imsave(save_path + '/TP/' + name, img_as_ubyte(res1))
            # imageio.imsave(save_path + '/TP/' + name, img_as_ubyte(image_2d_permuted))

        else:
            res_cpu = res.cpu()  #
            res_cpu_clipped = torch.clamp(res_cpu, -1.0, 1.0)  #
            image_2d = res_cpu_clipped.squeeze()  #
            image_2d_permuted = image_2d.permute(1, 2, 0)  #
            # imageio.imsave(save_path + '/FN/' + name, img_as_ubyte(res1))
            # imageio.imsave(save_path + '/FN/' + name, img_as_ubyte(image_2d_permuted))

    else:
        if predicted == 1:
            res_cpu = res.cpu()  #
            res_cpu_clipped = torch.clamp(res_cpu, -1.0, 1.0)  #
            image_2d = res_cpu_clipped.squeeze()  #
            image_2d_permuted = image_2d.permute(1, 2, 0)  #
            # imageio.imsave(save_path + '/FP/' + name, img_as_ubyte(res1))
            # imageio.imsave(save_path + '/FP/' + name, img_as_ubyte(image_2d_permuted))

        else:
            res_cpu = res.cpu()  #
            res_cpu_clipped = torch.clamp(res_cpu, -1.0, 1.0)  #
            image_2d = res_cpu_clipped.squeeze()  #
            image_2d_permuted = image_2d.permute(1, 2, 0)  #
            # print(image_2d.shape)
            # imageio.imsave(save_path + '/TN/' + name, img_as_ubyte(res1))
            # imageio.imsave(save_path + '/TN/' + name, img_as_ubyte(image_2d_permuted))


for i in range(test_loader2.size):
    image, gt, name = test_loader2.load_data()
    gt = np.asarray(gt, np.float32)

    gt = 1.0 * (gt > 0.5)

    image = image.cuda()

    with torch.no_grad():
        _, _, _, res = model(image)

    res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    # res = (res - res.min()) / (res.max() - res.min() + 1e-8)  ############################
    res = 1.0 * (res > 0.5)

    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)


print(
    "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
        np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
    )
)

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.flatten()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print("----------discriminator-----------")
print("TPR-FPR:", TPR - FPR)
print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
print("TN:", TN)

# slack = Slack_bot()

if TP != 0:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F_measure = f1_score(y_true, y_pred)
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("F-measure:", F_measure)
    print("precision:", precision)
    print("recall:", recall)
    print("specificity:", specificity)

    output_text_1 = f"----------discriminator-----------\nAccuracy: {accuracy}\nF-measuer: {F_measure}\n\n"

# fpr, tpr, thresholds = roc_curve(y_true, y_score)

# plt.plot(fpr, tpr)

# plt.xlabel('FPR: False positive rate')
# plt.ylabel('TPR: True positive rate')
# plt.grid()
# plt.savefig('./fig/roc_curve.png')

# AUC = roc_auc_score(y_true, y_score)
# print("AUC:", AUC)

# print("-------------cutoff--------------")
# Youden_index_candidates = tpr - fpr
# index = np.where(Youden_index_candidates == max(Youden_index_candidates))[0][0]
# cutoff = thresholds[index]
# y_pred_cutoff = (y_score >= cutoff).astype(int)

# cm = confusion_matrix(y_true, y_pred_cutoff)
# TN, FP, FN, TP = cm.flatten()
# print("TPR-FPR:", max(Youden_index_candidates))
# print("Threshold:", cutoff)
# print("TP:", TP)
# print("FN:", FN)
# print("FP:", FP)
# print("TN:", TN)

# if TP != 0:
#     accuracy = accuracy_score(y_true, y_pred_cutoff)
#     precision = precision_score(y_true, y_pred_cutoff)
#     recall = recall_score(y_true, y_pred_cutoff)
#     F_measure = f1_score(y_true, y_pred_cutoff)
#     specificity = TN / (TN + FP)

#     print("Accuracy:", accuracy)
#     print("F-measure:", F_measure)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("specificity:", specificity)

#     # slack(output_text_1 +
#     #       "----------cutoff-----------\n"
#     #       f"Accuracy: {accuracy}\n"
#     #       f"F-measuer: {F_measure}\n")
