import argparse
import glob
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import misc
from skimage import img_as_ubyte

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from lib.Trans_CaraNet import Trans_CaraNet_L
from utils.dataloader import test_dataset


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


k = 5
# thresh = 160
thresh = 128
# lower_size = 300
lower_size = 107
upper_size = 352 * 352 + 1


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
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument(
    "--test_path",
    type=str,
    default="./dataset/TestDataset/",
    help="path to test dataset",
)
parser.add_argument("--normalization", type=bool, default=False)

opt = parser.parse_args()

# data_path = opt.test_path
# data_path = './dataset/TestDataset/'
# data_path = './dataset/ValDataset/'
data_path = "./dataset/sekkai_TestDataset/"
# data_path = './dataset/sekkai_ValDataset/'

norm = opt.normalization
# norm = True

save_path = "./results/Transfuse_S/"

model = Trans_CaraNet_L()

# model.load_state_dict(torch.load('./snapshots/Transfuse_S/Transfuse-99.pth'))
# model.load_state_dict(torch.load('./snapshots/Transfuse_S/Transfuse-59.pth'))
# model.load_state_dict(torch.load('./snapshots/Transfuse_S/Transfuse-best.pth'))
model.load_state_dict(
    torch.load(
        "./weights/修論/segmentation/TransCaraNet+MAE_calsification/石灰化ありのみ/Transfuse-best.pth"
    )
)
# model.load_state_dict(torch.load('./weights/修論/segmentation/TransCaraNet+MAE_calsification/石灰化なし含む/Transfuse-best.pth'))
# model.load_state_dict(torch.load('./weights/修論/discriminator_nash/TransCaraNet_discriminator/ResNet/Transfuse-best.pth'))

model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
for file in glob.glob("./results/Transfuse_S/*.png"):
    os.remove(file)
image_root = "{}/images/".format(data_path)
gt_root = "{}/masks/".format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

dice_bank = []
iou_bank = []
acc_bank = []

no = 0
time_start = time.time()
for i in range(test_loader.size):
    no += 1
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)

    if norm:
        gt /= gt.max() + 1e-8  ##########################
    else:
        gt = 1.0 * (gt > 0.5)  ########################

    image = image.cuda()
    with torch.no_grad():
        res5, res4, res3, res2 = model(image)
    res = res2
    # res = res5
    res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()

    if norm:
        res = (res - res.min()) / (
            res.max() - res.min() + 1e-8
        )  ############################
    else:
        res = 1.0 * (res > 0.5)  ############################

    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)
    imageio.imsave(save_path + name, img_as_ubyte(res))

time_finish = time.time()
print("timer: {:.4f} sec.".format((time_finish - time_start) / no))
print(
    "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
        np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
    )
)
