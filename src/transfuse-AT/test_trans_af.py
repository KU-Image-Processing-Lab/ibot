import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
# from lib.TransFuse_s import TransFuse_S
from lib.se_TransFuse_s import SE_TransFuse_S
from lib.Bi_TransFuse_s import Bi_TransFuse_S
from lib.enhanced_TransFuse import enhanced_TransFuse
from utils.dataloaderT import test_dataset
from skimage import img_as_ubyte
import glob
from lib.TransFuse_s_split import TransFuse_S


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
# parser.add_argument('--pth_path', type=str, default='./snapshots/TransFuse_S/TransFuse-59.pth')
parser.add_argument('--pth_path', type=str, default='./snapshots/Weight_discri/TransFuse_af528unified_max-59.pth')
parser.add_argument('--data_path', type=str, default='./dataset/sekkai_TestDataset_unity')
parser.add_argument('--dice', type=int, default=1)

opt = parser.parse_args()

#data_path = './dataset/TestDataset/'
#data_path = './dataset/ValDataset/'
#data_path = './dataset/sekkai_TestDataset/'
# data_path = './dataset/sekkai_ValDataset/'
data_path = opt.data_path

save_path = './results/TransFuse_S/'

model = TransFuse_S()
# model = SE_TransFuse_S()
# model = Bi_TransFuse_S()
# model = enhanced_TransFuse()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
for file in glob.glob('./results/TransFuse_S/*.png'):
    os.remove(file)
image_root = '{}/images/'.format(data_path)
gt_root = '{}/masks/'.format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

dice_bank = []
iou_bank = []
acc_bank = []

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    if opt.dice:
        gt = 1. * (gt > 0.5)
    else:
        gt /= (gt.max() + 1e-8)


    image = image.cuda()

    with torch.no_grad():
        _, _, res = model(image)

    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    if opt.dice:
        res = 1. * (res > 0.5)
    else:
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)


    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)
    imageio.imsave(save_path + name, img_as_ubyte(res))

print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
      format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

# number of parameters
num_parameters = count_parameters(model)
print(num_parameters)
# number of trainable parameters
num_parameters = count_trainable_parameters(model)
print(num_parameters)
