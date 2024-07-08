from torchinfo import summary
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     def forward(self, input, size=512):
#         return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=1, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=4),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, 256)
        
        self.decoder = nn.Sequential(
            # UnFlatten()
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size = 4, stride=4),
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
        z = self.lrelu(z)
        # print(z.shape)
        z = z.view(B, 256, 1, 1)
        # print(z.shape)
        z = self.decoder(z)
        # print(z.shape)
        return z, mu, logvar


model = VAE()
summary(model, 
        input_size=(16, 3, 352, 352),
        col_names=["output_size", "num_params"],)