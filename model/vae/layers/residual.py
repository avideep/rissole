import torch.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        """
        Residual layer as specified in https://arxiv.org/abs/1711.00937. The original
        layer order is ReLU, 3x3 conv, ReLU, 1x1 conv (paper p. 5) with all having
        256 hidden units.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            hidden_dim (int): number of hidden units
        """
        super(ResidualLayer, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3,
                                 padding=1, stride=1, bias=False)
        self.conv1x1 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                 padding=0, stride=1, bias=False)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        identity = x

        x = self.activation(x)
        x = self.conv3x3(x)

        x = self.activation(x)
        x = self.conv1x1(x)

        x += identity

        return x
class Residual_Block(nn.Module): 
    def __init__(self, in_channels: int = 64, out_channels: int = 64, groups: int = 1, scale = 1.0):
        super(Residual_Block, self).__init__()
        
        mid_channels=int(out_channels*scale)
        
        if in_channels is not out_channels:
          self.conv_expand = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 


if __name__ == "__main__":
    import torch

    ipt = torch.randn((32, 64, 128, 128))
    rl = ResidualLayer(64, 64, 128)

    print("Input shape:", ipt.shape)        # [bs, 32, 64, 128, 128]
    print("Output shape:", rl(ipt).shape)   # [bs, 32, 64, 128, 128]
