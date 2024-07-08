import argparse
import glob
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from lib.TransFuse_l import TransFuse_L
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


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")


# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/Transfuse-99.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/Transfuse-59.pth')
# parser.add_argument('--pth_path', type=str, default='./snapshots/Transfuse_S/TransFuse-best.pth')
parser.add_argument(
    "--pth_path",
    type=str,
    default="./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化ありのみ/Transfuse-best.pth",
)
# parser.add_argument('--pth_path', type=str, default='./weights/修論/segmentation/TransFuse-L+MAE/vit-l_352/石灰化なし含む/Transfuse-best.pth')
# parser.add_argument('--pth_path', type=str, default='./weights/修論/discriminator_nash/TransFuse_discriminator/ResNet/Transfuse-best.pth')

parser.add_argument("--normalization", type=bool, default=False)

opt = parser.parse_args()

# data_path = './dataset/TestDataset/'
# data_path = './dataset/ValDataset/'
# data_path = './dataset/TrainDataset/'

data_path = "./dataset/sekkai_TestDataset/"
# data_path = './dataset/sekkai_ValDataset/'

norm = opt.normalization
# norm = True

save_path = "./results/Transfuse_S/"

# model = TransFuse_S()
model = TransFuse_L()
# model = TransFuse_H()
# model = beta_TransFuse_S()
# model = enhanced_TransFuse()
# model = enhanced_beta_TransFuse()
model.load_state_dict(torch.load(opt.pth_path))
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

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)

    if norm:
        gt /= gt.max() + 1e-8  ##########################
    else:
        gt = 1.0 * (gt > 0.5)  ########################

    image = image.cuda()

    with torch.no_grad():
        _, _, res = model(image)

    res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    # imageio.imsave(save_path + name, img_as_ubyte(res))

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

print(
    "Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}".format(
        np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)
    )
)
