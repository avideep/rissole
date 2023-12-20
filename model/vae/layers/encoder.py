import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            self._conv(in_channels, 32), # 31 x 31
            self._conv(32, 32), # 14 x 14
            self._conv(32, 64), # 6 x 6
            self._conv(64, 64), # 2 x 2 
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=2, stride=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

if __name__ == "__main__":
    import torch

    ipt = torch.randn((16, 3, 64, 64))
    enc = Encoder(3, 10)
    mu, logvar = enc(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3]
    print("Mu shape:", mu.shape)   # [bs, 10]
    print("Logvar shape:", logvar.shape)