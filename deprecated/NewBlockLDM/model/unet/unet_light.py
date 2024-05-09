import torch
import torch.nn as nn
import math

from model.layers import UpSample, DownSample
from model.layers import LinearAttention, Attention, CrossAttention, SpatialTransformer
from model.layers import TimeEmbedding, ConditionalEmbedding, SinusoidalPosEmb
from model.layers import ResidualBlockUNet, ConvNextBlock

from typing import List

class UNetLight(nn.Module):
    def __init__(self,
                 in_channels: int, time_emb_dim: int, pos_emb_dim: int, cond_emb_dim: int,
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
        self.cond_embedding = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)
        self.low_cond_embedding = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)
        self.pos_embedding = TimeEmbedding(time_emb_dim, pos_emb_dim)
        # initial convolutional layer
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)
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
        self.final_conv = nn.Conv2d(self.channels[0], in_channels, 1)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor, p: torch.Tensor, l: torch.Tensor = None):
        t = self.time_embedding(t)
        c = self.cond_embedding(x_cond)
        x = self.init_conv(x)
        p = self.pos_embedding(p)
        if l is not None:
            l = self.low_cond_embedding(l)

        skips = []

        # down sample
        for block1, attn1, block2, attn2, norm, downsample in self.down_blocks:
        # for block1, block2, norm, downsample in self.down_blocks:
            x = block1(x, c, t, p, l)
            if attn1 is not None:
                x = attn1(x, c)
            x = block2(x, c, t, p, l)
            if attn2 is not None:
                x = attn2(x, c)
            x = norm(x)
            skips.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, c, t, p, l)
        if self.use_spatial_transformer:
            x = self.mid_attn(x,c)
        else:
            x = self.mid_attn(x)
        x = self.mid_block2(x, c, t, p, l)

        # up sample
        for upsample, block1, attn1, block2, attn2, norm in self.up_blocks:
        # for upsample, block1, block2, norm in self.up_blocks:
            x = upsample(x)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, c, t, p, l)
            if attn1 is not None:
                x = attn1(x, c)
            x = block2(x, c, t, p , l)
            if attn2 is not None:
                x = attn2(x, c)
            x = norm(x)
        # output convolution
        x = self.final_conv(x)  

        return x
class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """
    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, position_conditioned=True):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
            position_conditioned (bool):
                Whether to condition the network on the difference between the current and previous frames. Should
                be True when training a DDPM frame predictor.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 2 if position_conditioned else time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], emb_dim=emb_dim, norm=True))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        if position_conditioned:
            # Encoder for positional embedding of position
            self.position_encoder = nn.Sequential(
                 SinusoidalPosEmb(time_dim),
                 nn.Linear(time_dim, time_dim * 4),
                 nn.GELU(),
                 nn.Linear(time_dim * 4, time_dim)
            )

    def forward(self, x, t, position=None):
        time_embedding = self.time_encoder(t)

        if position is not None:
            position_embedding = self.position_encoder(position)
            embedding = torch.cat([time_embedding, position_embedding], dim=1)
        else:
            embedding = time_embedding

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding)

        return self.final_conv(x)

if __name__ == "__main__":
    t_emb_dim = 16
    p_emb_dim = 8

    u = UNetLight(3, t_emb_dim, p_emb_dim, channels=[16, 32, 64, 128])

    ipt = torch.randn((8, 3, 128, 128))
    time = torch.randint(0, 10, (8,))

    out = u(ipt, time)
    print("Input:", ipt.shape)
    print("Output:", out.shape)