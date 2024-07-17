import argparse
import glob
import os
import random
import shutil

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import misc
from skimage import img_as_ubyte
from torch.autograd import Variable

from lib.Bi_TransFuse_s import Bi_TransFuse_S
from lib.enhanced_TransFuse import enhanced_TransFuse
from lib.se_TransFuse_s import SE_TransFuse_S
from lib.TransFuse_s import TransFuse_S
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="___")
parser.add_argument("--attempt", type=int, default=9800)
parser.add_argument(
    "--train_path",
    type=str,
    default="./dataset/sekkai_TrainDataset",
    help="path to train dataset",
)

opt = parser.parse_args()
train_file_list = glob.glob(opt.train_path + "/images/*.png")
dataset_count = len(train_file_list)

pth_folder = "./snapshots/TransFuse_S/"
pth_name = "TransFuse-59.pth"
pth_name2 = "TransFuse-best.pth"

new_pth_folder = f"./snapshots/{opt.file_name}/"
if os.path.exists(new_pth_folder):
    print("exists")
    pass
else:
    print("mkdir...")
    os.mkdir(new_pth_folder)

original_file_name = pth_folder + pth_name
new_file_name = (
    new_pth_folder + opt.file_name + f"-data{dataset_count}-attempt{opt.attempt}.pth"
)

original_file_name2 = pth_folder + pth_name2
new_file_name2 = (
    new_pth_folder
    + opt.file_name
    + f"_best-data{dataset_count}-attempt{opt.attempt}.pth"
)

print("copy")
print(original_file_name)
print(new_file_name)
shutil.copy2(original_file_name, new_file_name)

print(original_file_name2)
print(new_file_name2)
shutil.copy2(original_file_name2, new_file_name2)
print("completed!")
