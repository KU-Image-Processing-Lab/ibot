import glob
import os
import shutil

import cv2

base = "original_dataset_only_left"
target = "original_dataset_only_left_"

# base_dataset = f"./dataset/{base}/TrainDataset"
# target_dataset = f"./dataset/{target}/TrainDataset"
# base_dataset = f"./dataset/{base}/ValDataset"
# target_dataset = f"./dataset/{target}/ValDataset"
# base_dataset = f"./dataset/{base}/TestDataset"
# target_dataset = f"./dataset/{target}/TestDataset"

# base_dataset = f"./dataset/{base}/sekkai_TrainDataset"
# target_dataset = f"./dataset/{target}/sekkai_TrainDataset"
# base_dataset = f"./dataset/{base}/sekkai_ValDataset"
# target_dataset = f"./dataset/{target}/sekkai_ValDataset"
base_dataset = f"./dataset/{base}/sekkai_TestDataset"
target_dataset = f"./dataset/{target}/sekkai_TestDataset"

varbose = True

if not os.path.exists(target_dataset):
    os.makedirs(target_dataset)
    os.makedirs(target_dataset + "/images")
    os.makedirs(target_dataset + "/masks")

right_list = glob.glob(base_dataset + "/images/right*.png")
mright_list = glob.glob(base_dataset + "/masks/right*.png")
file_list = glob.glob(base_dataset + "/images/left*.png")


for f in right_list:
    img = cv2.imread(f)
    rev = cv2.flip(img, 1)
    new_file_path = f.replace(base_dataset + "/", target_dataset + "/")
    if varbose:
        print(new_file_path)
    shutil.copy(f, new_file_path)
    cv2.imwrite(new_file_path, rev)

for f in mright_list:
    img = cv2.imread(f)
    rev = cv2.flip(img, 1)
    new_file_path = f.replace(base_dataset + "/", target_dataset + "/")
    if varbose:
        print(new_file_path)
    shutil.copy(f, new_file_path)
    cv2.imwrite(new_file_path, rev)

for file in file_list:
    new_file_path = file.replace(base_dataset + "/", target_dataset + "/")
    if varbose:
        print(new_file_path)
    shutil.copy(file, new_file_path)

    mask_file = file.replace("/images/", "/masks/")
    new_mask_file = mask_file.replace(base_dataset, target_dataset)
    shutil.copy(mask_file, new_mask_file)
