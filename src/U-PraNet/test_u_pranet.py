import argparse
import glob
import os
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import misc
from skimage import img_as_ubyte

from lib.U_PraNet_Res2Net import U_PraNet
from utils.dataloader import test_dataset

# from lib.ResUPraNet import ResUPraNet


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
thresh = 250
# lower_size = 300
lower_size = 121
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
parser.add_argument("--normalization", type=bool, default=True)

opt = parser.parse_args()

# data_path = opt.test_path
# data_path = './dataset/TestDataset/'
# data_path = './dataset/ValDataset/'
data_path = "./dataset/sekkai_TestDataset/"
# data_path = './dataset/sekkai_ValDataset/'

# norm = opt.normalization
norm = False


save_path = "./results/PraNet/"
# for ep in range(opt.epoch):
for ep in range(99, 100):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    TP2 = 0
    FN2 = 0
    FP2 = 0
    TN2 = 0
    model = U_PraNet()
    # model = ResUPraNet()
    # model.load_state_dict(torch.load('./snapshots/PraNet_Res2Net/PraNet-' + str(ep + 1) + '.pth'))
    # model.load_state_dict(torch.load('./snapshots/PraNet_Res2Net/PraNet-best.pth'))
    model.load_state_dict(torch.load("./snapshots/PraNet_vat/PraNet-best.pth"))

    # model.load_state_dict(torch.load('./snapshots/PraNet_Res2Net/PraNet-100.pth'))

    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    for file in glob.glob("./results/PraNet/*.png"):
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

    files = glob.glob(data_path + "masks/*.png")  ###正解画像

    os.makedirs("./bbox_result/", exist_ok=True)
    for file in glob.glob("./bbox_result/*.png"):
        os.remove(file)
    os.makedirs("./bbox_result/TP", exist_ok=True)
    for file in glob.glob("./bbox_result/TP/*.png"):
        os.remove(file)
    os.makedirs("./bbox_result/FN", exist_ok=True)
    for file in glob.glob("./bbox_result/FN/*.png"):
        os.remove(file)
    os.makedirs("./bbox_result/FP", exist_ok=True)
    for file in glob.glob("./bbox_result/FP/*.png"):
        os.remove(file)
    os.makedirs("./bbox_result/TN", exist_ok=True)
    for file in glob.glob("./bbox_result/TN/*.png"):
        os.remove(file)
    for i in files:
        basename = os.path.basename(i)
        img = imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)

        img2 = imread(r"./results/PraNet/" + basename)
        t2, img2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        nlabels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(gray2)

        top2_idx2 = []
        for j in range(1, nlabels2):
            area = np.sum(labels2 == j)
            if (area >= lower_size) and (area <= upper_size):
                top2_idx2.append(j)

        output = np.zeros((height, width))
        for s in top2_idx2:
            output += np.where(labels2 == s, 255, 0)
        output = output.astype("uint8")
        cv2.imwrite("./re/vat_test/" + basename, output)

        #################################################################
        #########kai_pra_test######################################
        for k in range(1, nlabels1):
            count = 0
            for l in top2_idx2:
                l1 = labels1 == k
                l2 = labels2 == l
                a = l1 * l2
                if np.any(a):
                    TP += 1
                    count = 1
                    break
            if count == 0:
                FN += 1

        for x in top2_idx2:
            count = 0
            for y in range(1, nlabels1):
                l1 = labels1 == y
                l2 = labels2 == x
                a = l1 * l2

                if np.any(a):
                    count = 1
                    break
            if count == 0:
                FP += 1
        ##############################################################
        ##############一つでも重なったらTPとする#####################
        #########################sin_pra_test########################
        flag = False
        count = 0
        if nlabels1 > 1:
            for k in range(1, nlabels1):
                for l in top2_idx2:
                    l1 = labels1 == k
                    l2 = labels2 == l
                    a = l1 * l2
                    if np.any(a):
                        TP2 += 1
                        count = 1
                        flag = True
                        cv2.imwrite("./bbox_result/TP/" + basename, output)
                        break
                if flag:
                    break
            if count == 0:
                FN2 += 1
                cv2.imwrite("./bbox_result/FN/" + basename, output)
        if nlabels1 == 1:
            if len(top2_idx2) != 0:
                FP2 += 1
                cv2.imwrite("./bbox_result/FP/" + basename, output)

            else:
                TN2 += 1
                cv2.imwrite("./bbox_result/TN/" + basename, output)
    #########################################################################
    #############重なっていなくてもTPとする##################################
    ##############new_pra_test########################################
    # if len(top2_idx2) != 0:
    #     if nlabels1 > 1:
    #         TP2 += 1
    #         cv2.imwrite('./bbox_result/TP/' + basename, output)
    #     else:
    #         FP2 += 1
    #         cv2.imwrite('./bbox_result/FP/' + basename, output)
    # else:
    #     if nlabels1 > 1:
    #         FN2 += 1
    #         cv2.imwrite('./bbox_result/FN/' + basename, output)
    #     else:
    #         TN2 += 1
    #         cv2.imwrite('./bbox_result/TN/' + basename, output)

    print("-----------------------------")
    print("------------old--------------")
    print("epoch:", ep + 1)
    print("TP:", TP)
    print("FN:", FN)
    print("FP:", FP)
    if TP != 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F_measure = (2 * precision * recall) / (precision + recall)
        print("F-measure:", F_measure)
        print("precision:", precision)
        print("recall:", recall)

    print("------------sin--------------")
    print("epoch:", ep + 1)
    print("TP:", TP2)
    print("FN:", FN2)
    print("FP:", FP2)
    print("TN:", TN2)
    if TP2 != 0 and TN2 != 0:
        accuracy = (TP2 + TN2) / (TP2 + FN2 + FP2 + TN2)
        precision = TP2 / (TP2 + FP2)
        recall = TP2 / (TP2 + FN2)
        F_measure = (2 * precision * recall) / (precision + recall)
        specificity = TN2 / (TN2 + FP2)
        print("Accuracy:", accuracy)
        print("F-measure:", F_measure)
        print("precision:", precision)
        print("recall:", recall)
        print("specificity:", specificity)
