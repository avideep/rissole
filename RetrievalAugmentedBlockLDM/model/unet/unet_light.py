import torch
import torch.nn as nn

from model.layers import UpSample, DownSample
from model.layers import LinearAttention, Attention, CrossAttention, SpatialTransformer
from model.layers import TimeEmbedding, ConditionalEmbedding
from model.layers import ResidualBlockUNet

from typing import List

class UNetLight(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, time_emb_dim: int, pos_emb_dim: int, cond_emb_dim: int, activate_cond_layer: bool = False, use_addition: bool = False,
                 channels: List[int] = None, n_groups: int = 8, 
                 dim_keys: int = 64, n_heads: int = 4, use_spatial_transformer: bool = False):
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
        self.use_spatial_transformer = use_spatial_transformer
        # time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)
        # self.cond_embedding = ConditionalEmbedding(cond_emb_dim, self.channels[0])
        self.cond_embedding = nn.Conv2d(cond_emb_dim, self.channels[0], kernel_size=7, padding=3)
        # self.low_cond_embedding = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)
        self.pos_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)
        # initial convolutional layer
        # in_channels = 3 * in_channels
        # self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3) # for Celeb. Padding = 4 for ImageNet100 
        self.activate_cond_layer = activate_cond_layer
        self.use_addition = use_addition
        if self.activate_cond_layer:
            self.pre_init_conv = nn.Conv2d(in_channels, self.channels[0] // 2, kernel_size = 7, padding = 4)
            self.cond_conv = nn.Conv2d(cond_emb_dim, self.channels[0] // 2, kernel_size = 7, padding = 4)
            if self.use_addition:
                # self.layer_norm = nn.LayerNorm(None)
                self.init_conv = nn.Conv2d(self.channels[0] // 2, self.channels[0], kernel_size=7, padding=4)
        else:
            if self.use_addition:
                pass
                # self.layer_norm = nn.LayerNorm(None)
            else:
                in_channels += cond_emb_dim
            self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=4)
        # self.cond_attn = CrossAttention(in_channels, in_channels, dim_keys, n_heads)

        # contracting path
        self.down_blocks = nn.ModuleList([])
    
        prev_channel = self.channels[0] 
        cond_emb_dim = self.channels[0]
        for c in self.channels:
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlockUNet(prev_channel, c, time_emb_dim, cond_emb_dim, n_groups),
                    SpatialTransformer(c, n_heads, dim_keys, depth=1, context_dim= cond_emb_dim) if use_spatial_transformer else None,
                    ResidualBlockUNet(c, c, time_emb_dim, cond_emb_dim, n_groups),
                    SpatialTransformer(c, n_heads, dim_keys, depth=1, context_dim= cond_emb_dim) if use_spatial_transformer else None,
                    nn.GroupNorm(1, c),
                    DownSample(c)
                ])
            )
            prev_channel = c

        # bottleneck
        self.mid_block1 = ResidualBlockUNet(self.channels[-1], self.channels[-1], time_emb_dim, cond_emb_dim, n_groups)
        self.mid_attn = SpatialTransformer(self.channels[-1], n_heads, dim_keys, depth=1, context_dim= cond_emb_dim) if use_spatial_transformer else Attention(self.channels[-1], n_heads, dim_keys)
        # self.mid_attn = Attention(self.channels[-1], n_heads, dim_keys)

 
        self.mid_block2 = ResidualBlockUNet(self.channels[-1], self.channels[-1], time_emb_dim, cond_emb_dim, n_groups)

        # expanding path
        self.up_blocks = nn.ModuleList([])
        prev_channel = self.channels[-1]
        for c in reversed(self.channels):
            self.up_blocks.append(
                nn.ModuleList([
                    UpSample(prev_channel),
                    ResidualBlockUNet(prev_channel + c, c, time_emb_dim, cond_emb_dim, n_groups),
                    SpatialTransformer(c, n_heads, dim_keys, depth=1, context_dim= cond_emb_dim) if use_spatial_transformer else None,
                    ResidualBlockUNet(c, c, time_emb_dim, cond_emb_dim, n_groups),
                    SpatialTransformer(c, n_heads, dim_keys, depth=1, context_dim= cond_emb_dim) if use_spatial_transformer else None,
                    nn.GroupNorm(1, c),
                ])
            )
            prev_channel = c

        # final output 1x1 convolution
        self.final_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size = 5, padding = 1) #ImageNet100
        # self.final_conv = nn.Conv2d(self.channels[0], out_channels, 1) #CelebA


    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor, p: torch.Tensor, l: torch.Tensor = None):
        t = self.time_embedding(t)
        if self.activate_cond_layer:
            x_cond = self.cond_conv(x_cond)
            x = self.pre_init_conv(x)
            if self.use_addition:
                x = x + x_cond
                ln = nn.LayerNorm([x.size(1), x.size(2), x.size(3)]).cuda()
                ln.weight.requires_grad = True
                ln.bias.requires_grad = True
                x = ln(x)
                x = self.init_conv(x)
            else:
                x = torch.cat([x, x_cond], dim = 1)
        else:
            if self.use_addition:
                x = x + x_cond.view(-1, x.size(1), x.size(2), x.size(3))
                ln = nn.LayerNorm([x.size(1), x.size(2), x.size(3)]).cuda()
                ln.weight.requires_grad = True
                ln.bias.requires_grad = True
                x = ln(x)
            else:
                x = torch.cat([x, x_cond], dim = 1)
            x = self.init_conv(x)
        p = self.pos_embedding(p)
        if self.use_spatial_transformer:
            if l is not None:
                c = torch.cat([x_cond, l], dim=1)
                c = self.cond_embedding(c)
            else:
                c = self.cond_embedding(x_cond)
        else:
            c = None
        # print(x.shape)
        skips = []
        # down sample
        for block1, attn1, block2, attn2, norm, downsample in self.down_blocks:
        # for block1, block2, norm, downsample in self.down_blocks:
            x = block1(x, t, p)
            # print(x.shape)
            if attn1 is not None:
                x = attn1(x, c)
            x = block2(x, t, p)
            # print(x.shape)
            if attn2 is not None:
                x = attn2(x, c)
            x = norm(x)
            # print(x.shape)
            skips.append(x)
            x = downsample(x)
            # print(x.shape)

        # bottleneck
        x = self.mid_block1(x, t, p)
        # print(x.shape)
        if self.use_spatial_transformer:
            x = self.mid_attn(x,c)
        else:
            x = self.mid_attn(x)
            # print(x.shape)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t, p)
        # print(x.shape)
        # up sample
        for upsample, block1, attn1, block2, attn2, norm in self.up_blocks:
        # for upsample, block1, block2, norm in self.up_blocks:
            x = upsample(x)
            # print(x.shape, skips[-1].shape)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t, p)
            if attn1 is not None:
                x = attn1(x, c)
            x = block2(x, t, p)
            if attn2 is not None:
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