import numpy as np
import cv2
import glob
import os


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



for file in glob.glob('./bbox_result/*.png'):
    os.remove(file)
os.makedirs('./bbox_result/TP', exist_ok=True)
for file in glob.glob('./bbox_result/TP/*.png'):
    os.remove(file)
os.makedirs('./bbox_result/FN', exist_ok=True)
for file in glob.glob('./bbox_result/FN/*.png'):
    os.remove(file)
os.makedirs('./bbox_result/FP', exist_ok=True)
for file in glob.glob('./bbox_result/FP/*.png'):
    os.remove(file)
os.makedirs('./bbox_result/TN', exist_ok=True)
for file in glob.glob('./bbox_result/TN/*.png'):
    os.remove(file)


k = 5
lower_size = 818  ###########################検証データで求めたしきい値#################################
upper_size = 352 * 352 + 1

TP = 0
FN = 0
FP = 0
TN = 0

files = glob.glob(r"./dataset/TestDataset/masks/*.png")  ###正解画像

for i in files:
    basename = os.path.basename(i)
    img = imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    nlabels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(gray)

    img2 = imread(r"./results/Transfuse_S/" + basename)
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
    output = output.astype('uint8')

    ##################この間をコメントアウトすると、bboxなしのlower,threshで加工した出力が得られる#########################

    # nlabels3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(output)
    # bbox1 = []
    # for t in range(1, nlabels3):
    #     x0 = max(stats3[t][0] - k, 0)
    #     y0 = max(stats3[t][1] - k, 0)
    #     x1 = min(stats3[t][0] + stats3[t][2] + k, 415)
    #     y1 = min(stats3[t][1] + stats3[t][3] + k, 415)
    #     bbox1.append([x0, y0, x1, y1])
    # output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    # for u in bbox1:
    #     cv2.rectangle(output, (u[0], u[1]), (u[2], u[3]), (0, 0, 255), thickness=1)
    ##################################################################################################################
    cv2.imwrite('./bbox_result/' + basename, output)

    if len(top2_idx2) != 0:
        if nlabels1 > 1:
            TP += 1
            cv2.imwrite('./bbox_result/TP/' + basename, output)
        else:
            FP += 1
            cv2.imwrite('./bbox_result/FP/' + basename, output)
    else:
        if nlabels1 > 1:
            FN += 1
            cv2.imwrite('./bbox_result/FN/' + basename, output)
        else:
            TN += 1
            cv2.imwrite('./bbox_result/TN/' + basename, output)

TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
print("TPR-FPR:", TPR - FPR)
print("TP:", TP)
print("FN:", FN)
print("FP:", FP)
print("TN:", TN)
if TP != 0:
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_measure = (2 * precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    specificity = TN / (TN + FP)
    print("Accuracy:", accuracy)
    print("F-measure:", F_measure)
    print("precision:", precision)
    print("recall:", recall)
    print("specificity:", specificity)
