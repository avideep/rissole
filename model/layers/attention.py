import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Attention(nn.Module):
    def __init__(self, n_channels: int, dim_keys: int = 32, n_heads: int = 2):
        """
        Applies self-attention like in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
        to an image by reshaping it into a sequence. Only for small field sizes.

        Args:
            n_channels (int): Number of channels of the input feature maps
            dim_keys (int): Dimension of queries, keys, and values
            n_heads (int): Number of heads for attention
        """
        super().__init__()
        self.scale = dim_keys ** -0.5
        self.heads = n_heads
        hidden_dim = dim_keys * n_heads
        self.to_qkv = nn.Conv2d(n_channels, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, n_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attention = sim.softmax(dim=-1)

        res = torch.einsum("b h i j, b h d j -> b h i d", attention, v)
        res = rearrange(res, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self.to_out(res)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# class CrossAttention(nn.Module):
#     def __init__(self, n_channels: int, n_channels_cond: int = None, dim_keys: int = 32, n_heads: int = 2, dropout: float = 0):
#         """
#         Applies self-attention like in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
#         to an image by reshaping it into a sequence. Only for small field sizes.

#         Args:
#             n_channels (int): Number of channels of the input feature maps
#             n_channels_cond (int): Number of channels of the condtional feature map
#             dim_keys (int): Dimension of queries, keys, and values
#             n_heads (int): Number of heads for attention
#         """
#         super().__init__()
#         self.scale = dim_keys ** -0.5
#         self.heads = n_heads
#         hidden_dim = dim_keys * n_heads
#         if n_channels_cond is None:
#             n_channels_cond = n_channels
#         self.to_q = nn.Conv2d(n_channels, hidden_dim, 1, bias=False)
#         self.to_k = nn.Conv2d(n_channels_cond, hidden_dim, 1, bias=False)
#         self.to_v = nn.Conv2d(n_channels_cond, hidden_dim, 1, bias=False)
#         # self.to_out = nn.Sequential(
#         #     nn.Conv2d(hidden_dim, n_channels, 1),
#         #     nn.Dropout(dropout)
#         # )
#         self.to_out = nn.Sequential(
#             nn.Linear(hidden_dim, n_channels),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, cond = None):
#         b, c, h, w = x.shape
#         # if cond is None:
#         #     cond = x
#         cond = default(cond, x)
#         q = self.to_q(x)
#         k = self.to_k(cond)
#         v = self.to_v(cond)
#         q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), (q, k , v))
#         q = q * self.scale

#         sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attention = sim.softmax(dim=-1)

#         res = torch.einsum("b h i j, b h d j -> b h i d", attention, v)
#         res = rearrange(res, "b h (x y) d -> b (h d) x y", x=h, y=w)

#         return self.to_out(res)
class CrossAttention(nn.Module):
    def __init__(self, n_channels, n_channels_cond=None, n_heads=8, dim_keys=64, dropout=0.):
        super().__init__()
        hidden_dim = dim_keys * n_heads
        n_channels_cond = default(n_channels_cond, n_channels)

        self.scale = dim_keys ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(n_channels, hidden_dim, bias=False)
        self.to_k = nn.Linear(n_channels_cond, hidden_dim, bias=False)
        self.to_v = nn.Linear(n_channels_cond, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, n_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond=None, mask=None):
        h = self.n_heads

        q = self.to_q(x)
        cond = default(cond, x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, n_channels: int, dim_keys: int = 32, n_heads: int = 2):
        """
        Efficient Attention (https://arxiv.org/abs/1812.01243), which instead of
        computing V (Q K.T) like in dot-product attention, computes Q (K.T V).
        This results in less complexity, O(d_k * d_v) instead of O(n²).

        Args:
            n_channels (int): Number of channels of the input feature maps
            dim_keys (int): Dimension of queries, keys, and values
            n_heads (int): Number of heads for attention
        """
        super().__init__()
        self.scale = dim_keys ** -0.5
        self.n_heads = n_heads
        hidden_dim = dim_keys * n_heads
        self.to_qkv = nn.Conv2d(n_channels, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, n_channels, 1),
                                    nn.GroupNorm(1, n_channels))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.n_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        res = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        res = rearrange(res, "b h c (x y) -> b (h c) x y", h=self.n_heads, x=h, y=w)

        return self.to_out(res)
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(n_channels=dim, n_heads=n_heads, dim_keys=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(n_channels=dim, n_channels_cond=context_dim,
                                    n_heads=n_heads, dim_keys=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        print('x.shape before LayerNorm 1', x.shape)

        x_norm = self.norm1(x)
        print('x.shape after Layer norm 1', x_norm.shape)

        x_attn = self.attn1(x_norm)
        print('x.shape after self-attention', x_attn.shape)
        x = x + x_attn
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        print('x.shape', x.shape)
        x = self.norm(x)
        print('x.shape after group norm', x.shape)
        x = self.proj_in(x)
        print('x.shape after proj_in', x.shape)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        print('x.shape after rearrange', x.shape)

        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


if __name__ == "__main__":
    ipt = torch.randn((4, 1024, 8, 8))

    attn = Attention(1024, dim_keys=32, n_heads=2)
    out = attn(ipt)
    print("Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)

    lin_attn = LinearAttention(1024, dim_keys=32, n_heads=2)
    out = lin_attn(ipt)
    print("Linear Attention")
    print("\tInput:", ipt.shape)
    print("\tOutput:", out.shape)
