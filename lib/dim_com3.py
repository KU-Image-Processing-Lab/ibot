from torchinfo import summary
import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 64, 28, 28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 64, 14, 14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 128, 14, 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 128, 7, 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 256, 7, 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0,True),            
            nn.MaxPool2d(2)  # b, 256, 3, 3
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, padding=1),  # b, 512, 2, 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0,True),
        )
            

        self.mean = nn.Sequential(
            nn.Linear(2048,latent_dim),# b, 2048 ==> b, latent_dim
            )
        
        self.var = nn.Sequential(
            nn.Linear(2048,latent_dim),# b, 2048 ==> b, latent_dim
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,512),# b, latent_dim ==> b, 512
            )
        
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2,padding = 0),  # b, 256, 2, 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0,True),
            )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=4,padding = 0),  # b, 128, 7, 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0,True),
            )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2,padding = 1),  # b, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,True),
            )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2,padding = 1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
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
        x = self.conv1(x)      
        x = self.conv2(x)    
        x = self.conv3(x)     
        x = self.conv4(x)   
        x = x.view(-1,2048)
        mean = self.mean(x)
        var = self.var(x)
        return mean,var
    
    #Decoder
    def _decoder(self, z):
      z = self.decoder(z)
      z = z.view(-1,512, 1, 1)
      x = self.convTrans1(z)
      x = self.convTrans2(x)
      x = self.convTrans3(x)
      x = self.convTrans4(x)
      return x
  
    def forward(self, x):
        # xは元画像　
        mean,var = self._encoder(x) #Decoderの出力はlog σ^2を想定
        z = self._sample_z(mean, var) #潜在変数の分布に基づいてzをサンプリング
        x = self._decoder(z) #サンプリングしたzに対して画像を再構成
        return x,mean,var,z

model = VAE()
summary(model, 
        input_size=(512, 1, 28, 28),
        col_names=["output_size", "num_params"],)