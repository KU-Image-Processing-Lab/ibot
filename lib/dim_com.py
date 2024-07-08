from torchinfo import summary
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE_LVAT(nn.Module):
    def __init__(self, image_channels=3, h_dim=32768, z_dim=512):
        super(VAE_LVAT, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1), # b, 64, 128, 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0, True),
            nn.MaxPool2d(2),   # b, 64, 64, 64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # b, 128, 64, 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0, True),
            nn.MaxPool2d(2),  # b, 128, 32, 32
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # b, 256, 32, 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0, True),
            nn.MaxPool2d(2),  # b, 256, 16, 16
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # b, 512, 8, 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0, True),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)  # 32768 ==> 512
        self.fc2 = nn.Linear(h_dim, z_dim)  # 41472 ==> 512
        self.fc3 = nn.Linear(z_dim, 8192)
        
        self.decoder = nn.Sequential(
            # UnFlatten()
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2), # b, 512, 8, 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), # b, 256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # b, 128, 32 ,32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # b, 64, 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, image_channels, kernel_size = 2, stride=2), # b, 3, 128, 128
            nn.Sigmoid()
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp=esp.to('cuda')
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        h = self.encoder(x)
        # print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        # print(z.shape)
        z = self.fc3(z)
        # print(z.shape)
        z = self.lrelu(z)
        # print(z.shape)
        z = z.view(B, 512, 4, 4)
        # print(z.shape)
        z = self.decoder(z)
        return z, mu, logvar


model = VAE_LVAT()
summary(model, 
        input_size=(16, 3, 128, 128),
        col_names=["output_size", "num_params"],)