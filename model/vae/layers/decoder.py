import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super(Decoder, self).__init__()

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32),
            self._deconv(32, 3),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_dim, 1024)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 4, 4)
        return self.decoder(z)


if __name__ == "__main__":
    latent = torch.randn((16, 10))
    dec = Decoder(10, 3)
    out = dec(latent)

    print("Input shape:", latent.shape)     # [bs, 10, 32, 32]
    print("Output shape:", out.shape)       # [bs, 3, 128, 128]
    print(f"\t {out.min()} < {torch.mean(out)} < {out.max()}")
