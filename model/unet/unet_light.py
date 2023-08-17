import torch
import torch.nn as nn

from model.layers import UpSample, DownSample
from model.layers import LinearAttention, Attention, CrossAttention
from model.layers import TimeEmbedding, ConditionalEmbedding
from model.layers import ResidualBlockUNet

from typing import List

class UNetLight(nn.Module):
    def __init__(self,
                 in_channels: int, time_emb_dim: int, pos_emb_dim: int, cond_emb_dim: int,
                 channels: List[int] = None, n_groups: int = 8, 
                 dim_keys: int = 64, n_heads: int = 4):
        """
        U-Net model, first proposed in (https://arxiv.org/abs/1505.04597) and equipped for
        our DDPM with (linear) attention and time conditioning.

        Args:
            in_channels: Channels of the input image
            time_emb_dim: Dimension of time embedding
            pos_emb_dim: Dimension of fixed sinusoidal positional embedding
            cond_emb_dim: Dimension of the conditional information
            channels: List of channels for the number of down/up steps
            n_groups: Number of groups for group normalization
            dim_keys: Dimension of keys, queries, values for attention layers
            n_heads: Number of heads for multi-head attention
        """
        super().__init__()
        #in_channels = in_channels*2 
        self.channels = channels if channels is not None else [16, 32, 64]
        self.n_blocks = len(self.channels)

        # time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)
        self.cond_embedding = ConditionalEmbedding(cond_emb_dim, self.channels[0])
        # initial convolutional layer
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)
        # self.cond_attn = CrossAttention(in_channels, in_channels, dim_keys, n_heads)

        # contracting path
        self.down_blocks = nn.ModuleList([])
        prev_channel = self.channels[0]
        for c in self.channels:
            cond_emb_dim = c
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlockUNet(prev_channel, c, time_emb_dim, cond_emb_dim, n_groups),
                    CrossAttention(c, cond_emb_dim, dim_keys, n_heads),
                    ResidualBlockUNet(c, c, time_emb_dim, cond_emb_dim, n_groups),
                    CrossAttention(c, cond_emb_dim, dim_keys, n_heads),
                    nn.GroupNorm(1, c),
                    DownSample(c)
                ])
            )
            prev_channel = c

        # bottleneck
        self.mid_block1 = ResidualBlockUNet(self.channels[-1], self.channels[-1], time_emb_dim, cond_emb_dim, n_groups)
        self.mid_attn = CrossAttention( self.channels[-1], cond_emb_dim, dim_keys, n_heads)
        self.mid_block2 = ResidualBlockUNet(self.channels[-1], self.channels[-1], time_emb_dim, cond_emb_dim, n_groups)

        # expanding path
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            cond_emb_dim = c
            self.up_blocks.append(
                nn.ModuleList([
                    UpSample(prev_channel),
                    ResidualBlockUNet(prev_channel + c, c, time_emb_dim, cond_emb_dim, n_groups),
                    CrossAttention(c, cond_emb_dim, dim_keys, n_heads),
                    ResidualBlockUNet(c, c, time_emb_dim, cond_emb_dim, n_groups),
                    CrossAttention(c, cond_emb_dim, dim_keys, n_heads),
                    nn.GroupNorm(1, c),
                ])
            )
            prev_channel = c

        # final output 1x1 convolution
        self.final_conv = nn.Conv2d(self.channels[0], in_channels, 1)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor):
        t = self.time_embedding(t)
        c = self.cond_embedding(x_cond)
        x = self.init_conv(x)

        skips = []

        # down sample
        for block1, attn1, block2, attn2, norm, downsample in self.down_blocks:
            x = block1(x, c, t)
            x = attn1(x, c)
            x = block2(x, c, t)
            x = attn2(x, c)
            x = norm(x)
            skips.append(x)
            x = downsample(x)
            c = downsample(c)

        # bottleneck
        x = self.mid_block1(x, c, t)
        x = self.mid_attn(x, c)
        x = self.mid_block2(x, c, t)

        # up sample
        for upsample, block1, attn1, block2, attn2, norm in self.up_blocks:
            x = upsample(x)
            c = upsample(c)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, c, t)
            x = attn1(x, c)
            x = block2(x, t)
            x = attn2(x, c)
            x = norm(x)

        # output convolution
        x = self.final_conv(x)  

        return x


if __name__ == "__main__":
    t_emb_dim = 16
    p_emb_dim = 8

    u = UNetLight(3, t_emb_dim, p_emb_dim, channels=[16, 32, 64, 128])

    ipt = torch.randn((8, 3, 128, 128))
    time = torch.randint(0, 10, (8,))

    out = u(ipt, time)
    print("Input:", ipt.shape)
    print("Output:", out.shape)
