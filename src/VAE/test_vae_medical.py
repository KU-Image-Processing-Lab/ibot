import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
from VAE_linear import VAE
from VAE_linear import criterion
from vae_medical224 import VAE2 
import random
from utils.dataloader import test_dataset
from PIL import Image
from utils.dataloader import get_loader, PolypDataset



######ハイパーパラメータ#######
# P = '18'
batch_size = 24
z_dim = 128
learning_rate = 1e-3
###############################

image_root = 'dataset/sekkai_TestDataset_unity/images/'
gt_root = 'dataset/sekkai_TestDataset_unity/masks/'
# image_root = 'dataset/test' + P + '/images/'
# gt_root = 'dataset/test' + P + '/masks/'
testdataset = test_dataset(image_root, gt_root, testsize=224)     

# test_loader = get_loader(image_root, gt_root, batchsize=24, trainsize=32)


epoch_list = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = VAE2(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



model.load_state_dict(torch.load('weight/weight_VAE1178unified_last.pth'))  ######学習によって重みを変更#######




model.eval()
za_0 = []####
za_1 = []####
# os.makedirs('fig/linear/in_plot'+P)
# os.makedirs('fig/linear/re_plot'+P)
os.makedirs('fig/input1178unified')
os.makedirs('fig/result1178unified')
for i in range(testdataset.size):   #医用画像のとき

    image, gt, name = testdataset.load_data()
    

    image = image.cuda()
    with torch.no_grad():
     
     output = model(image) ########
    #  z = model.Encoder(image)
    #  za = z.to('cpu').detach().numpy().copy()
    #  za_0.append(za[0][0]) ##通常の時##
    #  za_1.append(za[0][1])
    
    #  if((za[0][1]<0.5*za[0][0]-6) & (za[0][1]>0.5*za[0][0]-7)):####z空間に制限をかける時##
    #     za_0.append(za[0][0])
    #     za_1.append(za[0][1])
    #     print(name)
    #     print(za[0])
    #     image = image[0] ##
    #     output = output[0] ########
    #     #  print(image.shape)
    #     tmp_f = image[0,:,:] ##
    #     tmp = output[0,:,:,:]###########
    #     #  tmp = unnorm(tmp)
    #     #[横幅,縦幅,色]にする
    #     tmp_f = image.permute(1, 2, 0) ##
    #     tmp = tmp.permute(1, 2, 0)  #######
    #     #  output = F.upsample(output, size=gt.shape, mode='bilinear', align_corners=False)
    #     tmp_f = tmp_f.data.cpu().numpy().squeeze() ##
    #     tmp = tmp.data.cpu().numpy().squeeze()  ##########
    #     img_pil_f = Image.fromarray((tmp_f*255).astype(np.uint8)) ##
    #     img_pil = Image.fromarray((tmp*255).astype(np.uint8)) ##########
        
    #     img_pil_f.save('fig/linear/i1/' + name) ##
    #     img_pil.save('fig/linear/r1/' + name) ###########
    #     fig = plt.figure()####
    #     plt.plot(za[0][0], za[0][1], '.')
    #     plt.xlim(-10, 15)
    #     plt.ylim(-20, 10)
    #     plt.grid
    #     fig.savefig("fig/linear/"+ name)
    #     # fig.savefig("fig/linear/plo.png")
        
        
    #  print(za[0][0])
     image = image[0] ##通常の時##
     output = output[0] ########
     tmp_f = image[0,:,:] ##
     tmp = output[0,:,:,:]###########
    #  tmp = unnorm(tmp)
     #[横幅,縦幅,色]にする
     tmp_f = image.permute(1, 2, 0) ##
     tmp = tmp.permute(1, 2, 0)  #######
    #  output = F.upsample(output, size=gt.shape, mode='bilinear', align_corners=False)
     tmp_f = tmp_f.data.cpu().numpy().squeeze() ##
     tmp = tmp.data.cpu().numpy().squeeze()  ##########
     img_pil_f = Image.fromarray((tmp_f*255).astype(np.uint8)) ##
     img_pil = Image.fromarray((tmp*255).astype(np.uint8)) ##########
     img_pil_f.save('fig/input1178unified/' + name) ##
     img_pil.save('fig/result1178unified/'  + name) ###########
   
# fig = plt.figure()####
# plt.plot(za_0, za_1, '.')
# plt.xlim(-10, 15)
# plt.ylim(-20, 10)
# plt.grid
# # # fig.savefig("fig/linear/plot"+P+".png")
# fig.savefig("fig/linear/r.png")
