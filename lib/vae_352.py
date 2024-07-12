import torchvision
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import pylab
import math
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class VAE(nn.Module):
#     def __init__(self, z_dim):
#       super(VAE, self).__init__()
#       self.dense_enc1 = nn.Linear(28*28, 200)
#       self.dense_enc2 = nn.Linear(200, 200)
#       self.dense_encmean = nn.Linear(200, z_dim)
#       self.dense_encvar = nn.Linear(200, z_dim)
#       self.dense_dec1 = nn.Linear(z_dim, 200)
#       self.dense_dec2 = nn.Linear(200, 200)
#       self.dense_dec3 = nn.Linear(200, 28*28)
    
#     def _encoder(self, x):
#       x = F.relu(self.dense_enc1(x))
#       x = F.relu(self.dense_enc2(x))
#       mean = self.dense_encmean(x)
#       var = F.softplus(self.dense_encvar(x))
#       return mean, var
    
#     def _sample_z(self, mean, var):
#       epsilon = torch.randn(mean.shape).to(device)
#       return mean + torch.sqrt(var) * epsilon
 
#     def _decoder(self, z):
#       x = F.relu(self.dense_dec1(z))
#       x = F.relu(self.dense_dec2(x))
#       x = F.sigmoid(self.dense_dec3(x))
#       return x

#     def forward(self, x):
#       mean, var = self._encoder(x)
#       z = self._sample_z(mean, var)
#       x = self._decoder(z)
#       return x, z
    
#     def loss(self, x):
#       mean, var = self._encoder(x)
#       KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
#       z = self._sample_z(mean, var)
#       y = self._decoder(z)
#       reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
#       lower_bound = [-KL, reconstruction]                                      
#       return -sum(lower_bound), KL, -reconstruction
  

#ネットワーク定義
class VAE2(nn.Module):
    def __init__(self, latent_dim):
        super(VAE2, self).__init__()
        self.conv00 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # b, 8, 352, 352
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 8, 176, 176
        )
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # b, 16, 176, 176
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 16, 88, 88
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # b, 32, 88, 88
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 32, 44, 44
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # b, 64, 44, 44
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 64, 22, 22
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 22, 22
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),          
            nn.MaxPool2d(2),  # b, 128, 11, 11
            nn.LeakyReLU(0,True),
        )
        
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 256, 2, stride=1, padding=1),  # b, 256, 12, 12
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0,True),
        #     nn.MaxPool2d(2),  # b, 256, 6, 6
        #     nn.LeakyReLU(0,True),
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, stride=1, padding=1),  # b, 512, 6, 6
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0,True),
        #     nn.MaxPool2d(2),  # b, 512, 3, 3
        #     nn.LeakyReLU(0,True),
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, stride=1, padding=0),  # b, 512, 1, 1
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0,True),
        # )
            
            

        self.mean = nn.Sequential(
            nn.Linear(15488,latent_dim),# b, 9216 ==> b, latent_dim
            )
        
        self.var = nn.Sequential(
            nn.Linear(15488,latent_dim),# b, 9216 ==> b, latent_dim
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,15488),# b, latent_dim ==> b, 9216
            )
        
        # self.convTrans1 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 4, stride=2,padding = 1),  # b, 256, 6, 6
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0,True),
        #     )
        # self.convTrans2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, 4, stride=2,padding = 1),  # b, 128, 11, 11
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0,True),
        #     )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2,padding = 1),  # b, 64, 22, 22
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
            )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2,padding = 1),  # b, 32, 44, 44
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0,True),
            )
        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2,padding = 1),  # b, 16, 88, 88
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0,True),
            )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2,padding = 1),  # b, 8, 176, 176
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0,True),
            )
      
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, 4, stride=2,padding = 1),  # b, 3, 352, 352
            nn.BatchNorm2d(3),
            nn.Sigmoid()
            )
        
        
    #Encoderの出力に基づいてzをサンプリングする関数
    #誤差逆伝搬ができるようにreparameterization trickを用いる
    def _sample_z(self, mean, var):
      std = var.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mean)
       
    
    #Encoder
    def _encoder(self, x):
        x = self.conv00(x)
        # print(x.shape)
        x = self.conv0(x)
        # print(x.shape)
        x = self.conv1(x)     
        # print(x.shape) 
        x = self.conv2(x)    
        # print(x.shape)
        x = self.conv3(x)     
        # print(x.shape)
        # x = self.conv4(x)   
        # print(x.shape)
        # x = self.conv5(x)   
        # print(x.shape)
        x = x.view(-1,15488)
        # print(x.shape)
        mean = self.mean(x)
        var = self.var(x)
        return mean,var
    
    #Decoder
    def _decoder(self, z):
      z = self.decoder(z)
      z = z.view(-1,128, 11, 11)
    #   print(z.shape)
    #   x = self.convTrans1(z)
      # print(x.shape)
    #   x = self.convTrans2(z)
      # print(x.shape)
      x = self.convTrans3(z)
    #   print(x.shape)
      x = self.convTrans4(x)
    #   print(x.shape)
      x = self.convTrans5(x)
    #   print(x.shape)
      x = self.convTrans6(x)
    #   print(x.shape)
      x = self.convTrans7(x)
    #   print(x.shape)
      return x
    
    def Encoder(self, x):   ###LVAT###
        # print(x.shape)
        mean,var = self._encoder(x) #Decoderの出力はlog σ^2を想定
        z = self._sample_z(mean, var)#潜在変数の分布に基づいてzをサンプリング
        return z
    
    def Decoder(self, z):   ###LVAT###
        x = self._decoder(z) #サンプリングしたzに対して画像を再構成
        return x
  
    def forward(self, x):
        # xは元画像　
        mean,var = self._encoder(x) #Decoderの出力はlog σ^2を想定
        z = self._sample_z(mean, var)#潜在変数の分布に基づいてzをサンプリング
        # print(z.size())
        x = self._decoder(z) #サンプリングしたzに対して画像を再構成
        return x,mean,var,z
    
    def loss(self, x):
      mean, var = self._encoder(x) #Decoderの出力はlog σ^2を想定
      
      KL = -0.5 * torch.mean(torch.sum(1 + var- mean**2 - var.exp())) #KLダイバージェンス
      
      z = self._sample_z(mean, var) #潜在変数の分布に基づいてzをサンプリング
      y = self._decoder(z) #サンプリングしたzに対して画像を再構成
      delta = 1e-7 #logの中身がマイナスにならないように微小な値を与える
      reconstruction = torch.mean(torch.sum(x * torch.log(y+delta) + (1 - x) * torch.log(1 - y +delta))) #再構成誤差
      lower_bound = [-KL, reconstruction]                           
      return -sum(lower_bound),KL,-reconstruction