import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import axes3d
from PIL import Image
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import save_image
# from vae_medical224 import VAE2
from vae_medical352 import VAE2

from utils.dataloader import get_loader, test_dataset

########ハイパーパラメータ等#######
epoch = 300
# batch_size = 1000
learning_rate = 1e-3
z_dim = 128  ##medicalで400, 40, 30,batch16  medical2で100 batch24   ##
best_loss = 100000000
train_losses = []
val_losses = []
epoch_list = []
#################################


# #画像データを前処理する関数
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(32, scale=(1.0, 1.0), ratio=(1., 1.)),
#     transforms.ToTensor(),
#     # transforms.Lambda(lambda x: x.view(-1))        ####vae.pyを使う場合###

#     ])

# image_root = 'dataset/HealthyDataset650+sekkai_radom100_unity/images/'
# gt_root = 'dataset/HealthyDataset650+sekkai_radom100_unity/images/'
image_root = "dataset/sekkai_TrainDataset528/images/"
gt_root = "dataset/sekkai_TrainDataset528/masks/"
image_root_val = "dataset/sekkai_ValDataset_unity/images/"
gt_root_val = "dataset/sekkai_ValDataset_unity/masks/"
# train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, droplast=True)
train_loader = get_loader(image_root, gt_root, batchsize=24, trainsize=352)
total_step = len(train_loader)
val_loader = get_loader(
    image_root_val, gt_root_val, batchsize=24, trainsize=352, phase="val"
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VAE2(z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

for i in range(1, epoch + 1):
    epoch_list.append(i)
    k = 0
    ###train####
    model.train()
    for x, t in train_loader:
        k += 1
        x = x.to(device)

        model.zero_grad()
        y = model(x)
        train_loss, kl, recon = model.loss(x)
        train_loss.backward()
        optimizer.step()
        if (i > 80) and (i % 2 == 0):
            scheduler.step()

    ###val###
    model.eval()
    with torch.no_grad():
        for x, t in val_loader:
            x = x.to(device)

            y = model(x)
            val_loss, _, _ = model.loss(x)

    ###最後のバッチにおける損失をリストに追加###
    train_losses.append(train_loss.cpu().detach().numpy())
    val_losses.append(val_loss.cpu().detach().numpy())

    print(
        "EPOCH: {}/{} total_step: {} train: {:.4f} KL: {:.4f} recon: {:.4f}".format(
            i, epoch, k, train_loss, kl, recon
        )
    )
    print("train: {:.4f} val: {:.4f}".format(train_loss, val_loss))

    ##val_lossがそれ以前のより減少した場合, 重みを更新###
    if best_loss > val_loss:
        best_loss = val_loss
        i_m = i
        torch.save(model.state_dict(), "./weight/weight_VAE{}.pth".format(i))
        print("Saving weight: weight_VAE{}".format(i))

    print("\n")
print("Best weight: weight_VAE{}".format(i_m))
torch.save(model.state_dict(), "./weight/weight_sekkai528+650unified_352_last.pth")

###損失のグラフを描写###
fig = plt.figure()
plt.plot(epoch_list, train_losses, label="train_loss")
plt.plot(epoch_list, val_losses, label="val_loss", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.xlim(left=1)
plt.legend(loc="upper right")

fig.savefig("fig/sekkai528+650unified_352.png")
