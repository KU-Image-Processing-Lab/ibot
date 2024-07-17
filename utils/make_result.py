"""
ex.sh no jikkouji ni result.txt kara 10 kai no kekka no heikin ya saidai wo motomete csv ni syuturyoku

"""
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='test224_vat_semi-supervised933ss.txt', help='result txt file name')
parser.add_argument('--read_line', type=int, default=5, help='read experimental result')
opt = parser.parse_args()

file_name = opt.file_name

with open(file_name) as f:
    lines = f.readlines()

# print(lines)

dice_L  = []
iou_L = []
acc_L = []
dice_B  = []
iou_B = []
acc_B = []
for i in range(opt.read_line):
    last = lines[3 + i * 8]
    best = lines[6 + i * 8]#lines[6 + i * 8]

    dice_last = float(last.split(',')[0].split(': ')[1].strip('\n'))
    dice_best = float(best.split(',')[0].split(': ')[1].strip('\n'))
    iou_last = float(last.split(',')[1].split(': ')[1].strip('\n'))
    iou_best = float(best.split(',')[1].split(': ')[1].strip('\n'))
    acc_last = float(last.split(',')[2].split(': ')[1].strip('\n'))
    acc_best = float(best.split(',')[2].split(': ')[1].strip('\n'))

    # if dice_last > dice_best:
    # # if iou_last > iou_best: # iouで評価
    #     dice_ = float(last.split(',')[0].split(': ')[1].strip('\n'))
    #     iou_ = float(last.split(',')[1].split(': ')[1].strip('\n'))
    #     acc_ = float(last.split(',')[2].split(': ')[1].strip('\n'))
    # elif dice_last <= dice_best:
    # # elif iou_last <= iou_best: # iouで評価
    #     dice_ = float(best.split(',')[0].split(': ')[1].strip('\n'))
    #     iou_ = float(best.split(',')[1].split(': ')[1].strip('\n'))
    #     acc_ = float(best.split(',')[2].split(': ')[1].strip('\n'))
    
    
    dice_L.append(dice_last)
    iou_L.append(iou_last)
    acc_L.append(acc_last)
    dice_B.append(dice_best)
    iou_B.append(iou_best)
    acc_B.append(acc_best)
    
    

dice_mean1 = np.array(dice_L).mean()
iou_mean1 = np.array(iou_L).mean()
acc_mean1 = np.array(acc_L).mean()

dice_max1 = np.array(dice_L).max()
iou_max1 = np.array(iou_L).max()
acc_max1 = np.array(acc_L).max()

dice_var1 = np.array(dice_L).std()
iou_var1 = np.array(iou_L).std()
acc_var1 = np.array(acc_L).std()

dice_mean2 = np.array(dice_B).mean()
iou_mean2 = np.array(iou_B).mean()
acc_mean2 = np.array(acc_B).mean()

dice_max2 = np.array(dice_B).max()
iou_max2 = np.array(iou_B).max()
acc_max2 = np.array(acc_B).max()

dice_var2 = np.array(dice_B).std()
iou_var2 = np.array(iou_B).std()
acc_var2 = np.array(acc_B).std()

dice_L.append(dice_mean1)
dice_L.append(dice_max1)
dice_L.append(dice_var1)

iou_L.append(iou_mean1)
iou_L.append(iou_max1)
iou_L.append(iou_var1)

acc_L.append(acc_mean1)
acc_L.append(acc_max1)
acc_L.append(acc_var1)

dice_B.append(dice_mean2)
dice_B.append(dice_max2)
dice_B.append(dice_var2)

iou_B.append(iou_mean2)
iou_B.append(iou_max2)
iou_B.append(iou_var2)

acc_B.append(acc_mean2)
acc_B.append(acc_max2)
acc_B.append(acc_var2)

df = pd.DataFrame()

df['Dice_last'] = dice_L
df['IOU_last'] = iou_L
df['ACC_last'] = acc_L
df['Dice_best'] = dice_B
df['IOU_best'] = iou_B
df['ACC_best'] = acc_B

df.head(3)

df.to_csv(file_name.split('.')[0] + '.csv')

