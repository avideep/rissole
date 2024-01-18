import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing

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
    
class IntroVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroVAE, self).__init__()         
        
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(in_channels, latent_dim, channels, image_size)
        
        self.decoder = Decoder(in_channels, latent_dim, channels, image_size)
        
      
    def forward(self, x):        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)        
        return mu, logvar, z, y
        
    def sample(self, z):        
        y = self.decode(z)
        return y
    
    def encode(self, x):  
        mu, logvar = self.encoder(x)
        return mu, logvar
        
    def decode(self, z):        
        y = self.decoder(z)
        return y
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5) # (batch, 2)
        return v_kl
    
    def reconstruction_loss(self, prediction, target, size_average=False):        
        error = (prediction - target).view(prediction.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=-1)
        
        if size_average:
            error = error.mean()
        else:
            error = error.sum()
               
        return error
        

if __name__ == "__main__":
    ipt = torch.randn((16, 3, 64, 64))

    vqvae = VAE(3, 10)
    rec, mu, logvar = vqvae(ipt)

    print("Input shape:", ipt.shape)    # [bs, 3, 64, 64]
    print("rec shape:", rec.shape)      # [bs, 3, 64, 64]
    print("mu:", mu.shape)              # [bs, 10]
    print("logvar:", logvar.shape)      # [bs, 10]
