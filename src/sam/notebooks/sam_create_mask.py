import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))  
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
##################    
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
##################



###画像の読み込み###
##単一画像の場合##
image_path = "left_CD13_070141.png"
image = cv2.imread(f"./images/{image_path}")
##複数画像の場合##
# image_dir = glob.glob('../../dataset/sekkai_TrainDataset528_unity/images/*.png')



##単一画像の場合##
###座標をプロンプトとしてsegmentationする#########
predictor = SamPredictor(sam)

predictor.set_image(image) 

# 座標をプロンプトとして指定する
input_point = np.array([[351, 70]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
) 

# マスクの一つを選択（例としてスコアが最高のものを選択）
best_mask_index = np.argmax(scores)  # スコアが最も高いマスクのインデックス
best_mask = masks[best_mask_index]  # 最適なマスクを取得

# 2値画像に変換
binary_mask = (best_mask * 255).astype(np.uint8)  # 0または1のマスクを0または255の2値画像に変換

# 2値画像を保存
cv2.imwrite(f"./images/{image_path}", binary_mask)
#####################################################





##複数画像の場合##
###ディレクトリ内の画像を一枚ずつsegmentationする(白黒画像出力)###
# a=0
# for image in image_dir:
#     a+=1
#     print(a)    
#     basename = os.path.basename(image)
#     image = cv2.imread(f"{image}")
#     predictor = SamPredictor(sam)
#     predictor.set_image(image)
    
#     # 座標をプロンプトとして指定する
#     input_point = np.array([[335, 50]])
#     input_label = np.array([1])
    
#     masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
#     )
    
#     # マスクの一つを選択（例としてスコアが最高のものを選択）
#     best_mask_index = np.argmax(scores)  # スコアが最も高いマスクのインデックス
#     best_mask = masks[best_mask_index]
#     # best_mask1 = masks[0]  # 一つ目のマスクを取得
#     # best_mask2 = masks[1]  # 二つ目のマスクを取得
    

#     # 2値画像に変換
#     binary_mask = (best_mask * 255).astype(np.uint8)
#     # binary_mask1 = (best_mask1 * 255).astype(np.uint8)  # 0または1のマスクを0または255の2値画像に変換
#     # binary_mask2 = (best_mask2 * 255).astype(np.uint8)  # 0または1のマスクを0または255の2値画像に変換

#     # 2値画像を保存
#     cv2.imwrite(f"./images/masks/{basename}", binary_mask)
#     # cv2.imwrite(f"./images/masks1/{basename}", binary_mask1)
#     # cv2.imwrite(f"./images/masks2/{basename}", binary_mask2)
########################################################################
