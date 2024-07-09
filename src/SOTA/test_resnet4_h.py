import argparse
import glob
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch

from torchvision import transforms

from torchinfo import summary

from lib.ResNet_4 import ResNet4

from utils.dataloader2 import test_dataset

import glob


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
    mask_sum = np.sum(np.abs(y_true), axis=axes) + \
        np.sum(np.abs(y_pred), axis=axes)
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
    mask_sum = np.sum(np.abs(y_true), axis=axes) + \
        np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
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
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./snapshots/ResNet4/3/Discriminator-best2.pth')
parser.add_argument('--test_path1', type=str, default='./discri_dataset/Test', help='path to test dataset')
parser.add_argument('--test_path2', type=str, default='./discri_dataset/Test/tensor/', help='path to test dataset')
# parser.add_argument('--test_path3', type=str, default='./discri_dataset/Train', help='path to test dataset')#
# parser.add_argument('--test_path4', type=str, default='./discri_dataset/Train/tensor/', help='path to test dataset')#
# parser.add_argument('--test_path5', type=str, default='./discri_dataset/Val', help='path to test dataset')#
# parser.add_argument('--test_path6', type=str, default='./discri_dataset/Val/tensor/', help='path to test dataset')#


opt = parser.parse_args()

data_path1 = opt.test_path1
data_path2 = opt.test_path2
# data_path3 = opt.test_path3#
# data_path4 = opt.test_path4#
# data_path5 = opt.test_path5#
# data_path6 = opt.test_path6#

model2 = ResNet4()

# summary(model2,input_size=(16, 3, 352, 352),col_names=["output_size", "num_params"],)

model2.load_state_dict(torch.load(opt.pth_path))#modelのbestを読み込み

model2.cuda()
model2.eval()

# ###################Train########################################
# gt_root1 = '{}/masks/'.format(data_path3)
# res_root1='{}'.format(data_path4)
# res_loader1=test_dataset(res_root1,gt_root1,opt.testsize)

# y_true1 = np.array([])
# y_score1= np.array([])
# y_pred1= np.array([])

# for i in range(res_loader1.size):
#     res1, gt1, name1 = res_loader1.load_data()
#     label1 = transforms.functional.to_tensor(gt1)
#     label1 = torch.einsum("ijk->i", label1) > 0
#     label1 = torch.where(label1 > 0, torch.tensor(1), torch.tensor(0))
#     gt1 = np.asarray(gt1, np.float32)

#     # gt /= (gt.max() + 1e-8)  ##########################

#     res1=res1.cuda()
#     with torch.no_grad():
    
#         out1 = model2(res1)#

#         _, predicted1= torch.max(out1, dim=1)

#         predicted1 = torch.Tensor.cpu(predicted1).detach().numpy()

#         label1 = torch.Tensor.cpu(label1).detach().numpy()
#         out1 = torch.Tensor.cpu(out1).detach().numpy()
#         y_true1 = np.append(y_true1, label1)
#         y_score1 = np.append(y_score1, out1[0][1] - out1[0][0])
#         y_pred1 = np.append(y_pred1, predicted1)

# cm1 = confusion_matrix(y_true1, y_pred1)
# TN1, FP1, FN1, TP1 = cm1.flatten()
# TPR1 = TP1 / (TP1 + FN1)
# FPR1 = FP1 / (FP1 + TN1)
# print("----------discriminatorTrain-----------")
# # print("TPR-FPR:", TPR-FPR)
# print("TP:", TP1)
# print("FN:", FN1)
# print("FP:", FP1)
# print("TN:", TN1)

# # slack = Slack_bot()

# if TP1 != 0:
#     accuracy1 = accuracy_score(y_true1, y_pred1)
#     precision1 = precision_score(y_true1, y_pred1)
#     recall1 = recall_score(y_true1, y_pred1)
#     F_measure1= f1_score(y_true1, y_pred1)
#     specificity1 = TN1 / (TN1 + FP1)

#     print("Accuracy:", accuracy1)
#     print("F-measure:", F_measure1)
#     print("precision:", precision1)
#     print("recall:", recall1)
#     print("specificity:", specificity1)

#     output_text_1 = f"----------discriminatorTrain-----------\nAccuracy: {accuracy1}\nF-measuer: {F_measure1}\n\n"

# fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_score1)

# # plt.plot(fpr1, tpr1)

# # plt.xlabel('FPR: False positive rate')
# # plt.ylabel('TPR: True positive rate')
# # plt.grid()
# # plt.savefig('./fig/roc_curve.png')

# AUC1 = roc_auc_score(y_true1, y_score1)
# print("AUC:", AUC1)

# print("-------------cutoffTrain--------------")
# Youden_index_candidates_Train = tpr1 - fpr1
# index1 = np.where(Youden_index_candidates_Train == max(Youden_index_candidates_Train))[0][0]
# cutoff1 = thresholds1[index1]
# y_pred_cutoff1 = (y_score1 >= cutoff1).astype(int)

# cm1 = confusion_matrix(y_true1, y_pred_cutoff1)
# TN1, FP1, FN1, TP1 = cm1.flatten()
# print("TPR-FPR:", max(Youden_index_candidates_Train))
# print("Threshold:", cutoff1)
# print("TP:", TP1)
# print("FN:", FN1)
# print("FP:", FP1)
# print("TN:", TN1)

# if TP1 != 0:
#     accuracy1 = accuracy_score(y_true1, y_pred_cutoff1)
#     precision1 = precision_score(y_true1, y_pred_cutoff1)
#     recall1 = recall_score(y_true1, y_pred_cutoff1)
#     F_measure1 = f1_score(y_true1, y_pred_cutoff1)
#     specificity1 = TN1 / (TN1 + FP1)

#     print("Accuracy:", accuracy1)
#     print("F-measure:", F_measure1)
#     print("precision:", precision1)
#     print("recall:", recall1)
#     print("specificity:", specificity1)

# ##################Val#######################################
# gt_root2 = '{}/masks/'.format(data_path5)
# res_root2='{}'.format(data_path6)
# res_loader2=test_dataset(res_root2,gt_root2,opt.testsize)

# y_true2 = np.array([])
# y_score2= np.array([])
# y_pred2= np.array([])

# for i in range(res_loader2.size):
#     res2, gt2, name2 = res_loader2.load_data()
#     label2 = transforms.functional.to_tensor(gt2)
#     label2 = torch.einsum("ijk->i", label2) > 0
#     label2 = torch.where(label2 > 0, torch.tensor(1), torch.tensor(0))
#     gt2 = np.asarray(gt2, np.float32)

#     # gt /= (gt.max() + 1e-8)  ##########################

#     res2=res2.cuda()
#     with torch.no_grad():
    
#         out2 = model2(res2)#

#         _, predicted2= torch.max(out2, dim=1)

#         predicted2 = torch.Tensor.cpu(predicted2).detach().numpy()

#         label2 = torch.Tensor.cpu(label2).detach().numpy()
#         out2 = torch.Tensor.cpu(out2).detach().numpy()
#         y_true2 = np.append(y_true2, label2)
#         y_score2 = np.append(y_score2, out2[0][1] - out2[0][0])
#         y_pred2 = np.append(y_pred2, predicted2)

# cm2 = confusion_matrix(y_true2, y_pred2)
# TN2, FP2, FN2, TP2 = cm2.flatten()
# TPR2 = TP2 / (TP2 + FN2)
# FPR2 = FP2 / (FP2 + TN2)
# print("----------discriminatorVal-----------")
# # print("TPR-FPR:", TPR-FPR)
# print("TP:", TP2)
# print("FN:", FN2)
# print("FP:", FP2)
# print("TN:", TN2)

# # slack = Slack_bot()

# if TP1 != 0:
#     accuracy2 = accuracy_score(y_true2, y_pred2)
#     precision2 = precision_score(y_true2, y_pred2)
#     recall2 = recall_score(y_true2, y_pred2)
#     F_measure2= f1_score(y_true2, y_pred2)
#     specificity2 = TN2 / (TN2 + FP2)

#     print("Accuracy:", accuracy2)
#     print("F-measure:", F_measure2)
#     print("precision:", precision2)
#     print("recall:", recall2)
#     print("specificity:", specificity2)

#     output_text_2 = f"----------discriminatorVal-----------\nAccuracy: {accuracy2}\nF-measuer: {F_measure2}\n\n"

# fpr2, tpr2, thresholds2 = roc_curve(y_true2, y_score2)

# # plt.plot(fpr2, tpr2)

# # plt.xlabel('FPR: False positive rate')
# # plt.ylabel('TPR: True positive rate')
# # plt.grid()
# # plt.savefig('./fig/roc_curve.png')

# AUC2 = roc_auc_score(y_true2, y_score2)
# print("AUC:", AUC2)

# print("-------------cutoffVal--------------")
# Youden_index_candidates_Val = tpr2 - fpr2
# index2 = np.where(Youden_index_candidates_Val == max(Youden_index_candidates_Val))[0][0]
# cutoff2 = thresholds2[index2]
# y_pred_cutoff2 = (y_score2 >= cutoff2).astype(int)

# cm2 = confusion_matrix(y_true2, y_pred_cutoff2)
# TN2, FP2, FN2, TP2 = cm2.flatten()
# print("TPR-FPR:", max(Youden_index_candidates_Val))
# print("Threshold:", cutoff2)
# print("TP:", TP2)
# print("FN:", FN2)
# print("FP:", FP2)
# print("TN:", TN2)

# if TP2 != 0:
#     accuracy2 = accuracy_score(y_true2, y_pred_cutoff2)
#     precision2 = precision_score(y_true2, y_pred_cutoff2)
#     recall2 = recall_score(y_true2, y_pred_cutoff2)
#     F_measure2 = f1_score(y_true2, y_pred_cutoff2)
#     specificity2 = TN2 / (TN2 + FP2)

#     print("Accuracy:", accuracy2)
#     print("F-measure:", F_measure2)
#     print("precision:", precision2)
#     print("recall:", recall2)
#     print("specificity:", specificity2)
##################Test#########################################
gt_root = '{}/masks/'.format(data_path1)
res_root='{}'.format(data_path2)
res_loader=test_dataset(res_root,gt_root,opt.testsize)

y_true = np.array([])
y_score = np.array([])
y_pred = np.array([])

for i in range(res_loader.size):
    res, gt, name = res_loader.load_data()
    label = transforms.functional.to_tensor(gt)
    label = torch.einsum("ijk->i", label) > 0
    label = torch.where(label > 0, torch.tensor(1), torch.tensor(0))
    gt = np.asarray(gt, np.float32)

    # gt /= (gt.max() + 1e-8)  ##########################

    res=res.cuda()
    with torch.no_grad():
    
        out = model2(res)

        _, predicted = torch.max(out, dim=1)

        predicted = torch.Tensor.cpu(predicted).detach().numpy()

        label = torch.Tensor.cpu(label).detach().numpy()
        out = torch.Tensor.cpu(out).detach().numpy()
        y_true = np.append(y_true, label)
        y_score = np.append(y_score, out[0][1] - out[0][0])
        y_pred = np.append(y_pred, predicted)

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.flatten()
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print("----------discriminator-----------")
# print("TPR-FPR:", TPR-FPR)
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

AUC = roc_auc_score(y_true, y_score)
print("AUC:", AUC)

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

# print("-------------useTraincutoff--------------")
# y_pred_cutoff1 = (y_score >= cutoff1).astype(int)

# cm = confusion_matrix(y_true, y_pred_cutoff1)
# TN, FP, FN, TP = cm.flatten()
# print("TPR-FPR:", max(Youden_index_candidates_Train))
# print("Threshold:", cutoff1)
# print("TP:", TP)
# print("FN:", FN)
# print("FP:", FP)
# print("TN:", TN)

# if TP != 0:
#     accuracy = accuracy_score(y_true, y_pred_cutoff1)
#     precision = precision_score(y_true, y_pred_cutoff1)
#     recall = recall_score(y_true, y_pred_cutoff1)
#     F_measure = f1_score(y_true, y_pred_cutoff1)
#     specificity = TN / (TN + FP)

#     print("Accuracy:", accuracy)
#     print("F-measure:", F_measure)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("specificity:", specificity)

# print("-------------useValcutoff--------------")
# y_pred_cutoff2 = (y_score >= cutoff2).astype(int)

# cm = confusion_matrix(y_true, y_pred_cutoff2)
# TN, FP, FN, TP = cm.flatten()
# print("TPR-FPR:", max(Youden_index_candidates_Val))
# print("Threshold:", cutoff2)
# print("TP:", TP)
# print("FN:", FN)
# print("FP:", FP)
# print("TN:", TN)

# if TP != 0:
#     accuracy = accuracy_score(y_true, y_pred_cutoff2)
#     precision = precision_score(y_true, y_pred_cutoff2)
#     recall = recall_score(y_true, y_pred_cutoff2)
#     F_measure = f1_score(y_true, y_pred_cutoff2)
#     specificity = TN / (TN + FP)

#     print("Accuracy:", accuracy)
#     print("F-measure:", F_measure)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("specificity:", specificity)
