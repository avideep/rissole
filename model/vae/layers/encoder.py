import torch.nn as nn
from residual import Residual_Block

# class Encoder(nn.Module):
#     def __init__(self, in_channels: int, latent_dim: int):
#         super(Encoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = nn.Sequential(
#             self._conv(in_channels, 32), # 31 x 31
#             self._conv(32, 32), # 14 x 14
#             self._conv(32, 64), # 6 x 6
#             self._conv(64, 64), # 2 x 2 
#         )
#         self.fc_mu = nn.Linear(16384, latent_dim)
#         self.fc_var = nn.Linear(16384, latent_dim)

#     def _conv(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(
#                 in_channels, out_channels,
#                 kernel_size=2, stride=2,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2)
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, 16384)
#         return self.fc_mu(x), self.fc_var(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.latent_dim = latent_dim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(in_channels, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 2*latent_dim)           
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)                
        return mu, logvar

if __name__ == "__main__":
    import torch

    ipt = torch.randn((16, 3, 64, 64))
    enc = Encoder(3, 10)
    mu, logvar = enc(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3]
    print("Mu shape:", mu.shape)   # [bs, 10]
    print("Logvar shape:", logvar.shape)