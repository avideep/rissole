import torch
import torch.nn as nn
from residual import Residual_Block

# class Decoder(nn.Module):
#     def __init__(self, latent_dim: int, out_channels: int):
#         super(Decoder, self).__init__()
#         self.latent_dim = latent_dim
#         # decoder
#         self.decoder = nn.Sequential(
#             self._deconv(64, 64),
#             self._deconv(64, 32),
#             self._deconv(32, 32),
#             self._deconv(32, 3),
#             nn.Sigmoid()
#         )
#         self.fc_z = nn.Linear(latent_dim, 16384)
#     def _deconv(self, in_channels, out_channels, out_padding=0):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels, out_channels,
#                 kernel_size=2, stride=2, output_padding=out_padding
#             ),
#             # nn.Conv2d(out_channels, out_channels,
#             #           kernel_size = 3, stride = 1, padding = 1
#             # ),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2)
#         )
#     def forward(self, z):
#         z = self.fc_z(z)
#         z = z.view(-1, 64, 16, 16)
#         return self.decoder(z)

class Decoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]
        self.fc = nn.Sequential(
                      nn.Linear(latent_dim, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, in_channels, 5, 1, 2))
                    
    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y


if __name__ == "__main__":
    latent = torch.randn((16, 10))
    dec = Decoder(10, 3)
    out = dec(latent)

    print("Input shape:", latent.shape)     # [bs, 10, 32, 32]
    print("Output shape:", out.shape)       # [bs, 3, 128, 128]
    print(f"\t {out.min()} < {torch.mean(out)} < {out.max()}")
