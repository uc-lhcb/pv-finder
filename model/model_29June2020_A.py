'''
UNet model with the following properties:
- 3 skip connections
**- 1/2 -> 1/4 -> 1/4 downsample rates for each layer** (i.e. increased MaxPooling to k=4, s=4 on x3, x4)
- Kernel size decay (i.e. kernel size in convolutional layers proportionally decreases with downsampling)
'''

from torch import nn
import torch

class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBNrelu, self).__init__(
        nn.Conv1d(in_channels, out_channels, k_size, stride=1, padding=(k_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU())
    

class Up(nn.Sequential):
    def __init__(self, inc, outc, k_size, s):
        super().__init__(
            nn.ConvTranspose1d(inc, outc, k_size, s),
            ConvBNrelu(outc, outc, k_size=5))


class UNet3SC(nn.Module):
    def __init__(self, n=24):
        super(UNet3SC, self).__init__()
        self.d2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.d4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.rcbn1 = ConvBNrelu(1, n, k_size=25)
        self.rcbn2 = ConvBNrelu(n, n, k_size=7)
        self.rcbn3 = ConvBNrelu(n, n, k_size=5)
        self.rcbn4 = ConvBNrelu(n, n, k_size=3)

        self.up1 = Up(n, n, k_size=4, s=4)
        self.up2 = Up(n*2, n, k_size=4, s=4)
        self.up3 = Up(n*2, n, k_size=2, s=2)
        self.outc = nn.Conv1d(n*2, 1, 5, padding=2)

    def forward(self, x):
        x1 = self.rcbn1(x)
        x2 = self.d2(self.rcbn2(x1))# 2000       
        x3 = self.d4(self.rcbn3(x2))# 500
        x = self.d4(self.rcbn4(x3)) # 125

        x = self.up1(x) # 500
        x = self.up2(torch.cat([x, x3], 1)) # 2000
        x = self.up3(torch.cat([x, x2], 1)) # 4000

        logits_x0 = self.outc(torch.cat([x, x1], 1))

        ret = torch.nn.Softplus()(logits_x0).squeeze()
        return  ret