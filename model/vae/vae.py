import torch
import torch.nn as nn

from model.vae.layers import Encoder
from model.vae.layers import Decoder


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z: torch.Tensor):
        x_hat = self.decoder(z)
        return x_hat


if __name__ == "__main__":
    ipt = torch.randn((16, 3, 32, 32))

    vqvae = VAE(3, 10)
    rec, mu, logvar = vqvae(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 64, 64]
    print("rec shape:", rec.shape)      # [bs, 3, 64, 64]
    print("mu:", mu.shape)              # [bs, 10]
    print("logvar:", logvar.shape)      # [bs, 10]
