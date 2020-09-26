from torch import nn
import torch

'''
Baseline UNet model
UNet model with the following properties:
- 2 skip connections
- 1/2 downsample rate
- Fixed kernel size (exc. block1)
'''
class ConvBNRelu_A(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBNRelu_A, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

class Up_A(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

class UNet_A(nn.Module):
    def __init__(self, n=48):
        super(UNet_A, self).__init__()
        
        self.block1 = ConvBNRelu_A(1, n, kernel_size=25)
        self.block2 = ConvBNRelu_A(n, n)
        self.block3 = ConvBNRelu_A(n, n)
        self.up1 = Up_A(n, n)
        self.up2 = Up_A(n*2, n)
        self.up3 = Up_A(n*2, n)
        self.up4 = Up_A(n, n)
        self.outc = nn.Conv1d(n, 1, 3, padding=1)
        self.down = nn.MaxPool1d(2)

    def forward(self, x):
        x1 = self.block1(x)
        x1 = self.down(x1)

        x2 = self.block2(x1)
        x2 = self.down(x2)

        x = self.block3(x2)
        x = self.down(x)

        x = self.up1(x)
        x = self.up2(torch.cat([x, x2], 1))
        x = self.up3(torch.cat([x, x1], 1))

        x = self.outc(x)
        
        x = torch.nn.Softplus()(x).reshape(128, 4000)
        return x
        
        
'''
UNet model with the following properties:
- 3 skip connections
**- 1/2 -> 1/4 -> 1/4 downsample rates for each layer** (i.e. increased MaxPooling to k=4, s=4 on x3, x4)
- Kernel size decay (i.e. kernel size in convolutional layers proportionally decreases with downsampling)
- Convolutional layer in upsample
'''
class ConvBNRelu_B(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBNRelu_B, self).__init__(
        nn.Conv1d(in_channels, out_channels, k_size, stride=1, padding=(k_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU())
    

class Up_B(nn.Sequential):
    def __init__(self, inc, outc, k_size, s):
        super().__init__(
            nn.ConvTranspose1d(inc, outc, k_size, s),
            ConvBNRelu_B(outc, outc, k_size=5))


class UNet_B(nn.Module):
    def __init__(self, n=24):
        super(UNet_B, self).__init__()
        self.d2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.d4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.rcbn1 = ConvBNRelu_B(1, n, k_size=25)
        self.rcbn2 = ConvBNRelu_B(n, n, k_size=7)
        self.rcbn3 = ConvBNRelu_B(n, n, k_size=5)
        self.rcbn4 = ConvBNRelu_B(n, n, k_size=3)

        self.up1 = Up_B(n, n, k_size=4, s=4)
        self.up2 = Up_B(n*2, n, k_size=4, s=4)
        self.up3 = Up_B(n*2, n, k_size=2, s=2)
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

    
'''
UNet model with the following properties:
- 4 skip connections
- 1/2 downsample rate
**- No MaxPool layers; all downsampling occurs in the convolutional layers** (i.e. stride=2)
- Kernel size decay (i.e. kernel size in convolutional layers proportionally decreases with downsampling)
- Convolutional layer in upsample
'''
class ConvBNRelu_C(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBNRelu_C, self).__init__(
            nn.Conv1d(in_channels, out_channels, k_size, stride=2, padding=(k_size-2)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
    
class ConvBNReluSame(nn.Sequential):
    """preserves dimension of input"""
    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBNreluSame, self).__init__(
            nn.Conv1d(in_channels, out_channels, k_size, stride=1, padding=(k_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
    
class Up_C(nn.Sequential):
    def __init__(self, inc, outc, k_size):
        super().__init__(
            nn.ConvTranspose1d(inc, outc, kernel_size=2, stride=2),
            nn.Conv1d(outc, outc, k_size, stride=1, padding=(k_size-1)//2),
            nn.BatchNorm1d(outc),
            nn.ReLU())


class UNet_C(nn.Module):
    def __init__(self, n=24):
        super(UNet_C, self).__init__()
        
        self.rcbn1 = ConvBNReluSame(1, n, k_size=25)
        self.rcbn2 = ConvBNRelu_C(n, n, k_size=12)
        self.rcbn3 = ConvBNRelu_C(n, n, k_size=6)
        self.rcbn4 = ConvBNRelu_C(n, n, k_size=4)
        self.rcbn5 = ConvBNRelu_C(n, n, k_size=4)

        self.up1 = Up_C(n, n, k_size=3)
        self.up2 = Up_C(n*2, n, k_size=3)
        self.up2 = Up_C(n*2, n, k_size=7)
        self.up3 = Up_C(n*2, n, k_size=13)
        self.up4 = Up_C(n*2, n, k_size=25)
        self.outc = nn.Conv1d(n*2, 1, 3, padding=1)

    def forward(self, x):
        
        x1 = self.rcbn1(x) # 4000
        x2 = self.rcbn2(x1) # 2000       
        x3 = self.rcbn3(x2) # 1000
        x4 = self.rcbn4(x3) # 500
        x = self.rcbn5(x4) # 250

        x = self.up1(x) # 500
        x = self.up2(torch.cat([x, x4], 1)) # 1000
        x = self.up3(torch.cat([x, x3], 1)) #2000
        x = self.up4(torch.cat([x, x2], 1)) #4000

        logits_x0 = self.outc(torch.cat([x, x1], 1))

        ret = torch.nn.Softplus()(logits_x0).squeeze()
        return  ret
    
    
'''
UNet model with the following properties:
- 3 skip connections
**- 1/2 -> 1/4 -> 1/4 downsample rates for each layer** (i.e. increased MaxPooling to k=4, s=4 on x3, x4)
- Kernel size decay (i.e. kernel size in convolutional layers proportionally decreases with downsampling)
- Convolutional layer in upsample
- LeakyReLU mid-activation function
'''
class ConvBNRelu_D(nn.Sequential):
    """convolution => [BN] => LeakyReLU"""
    def __init__(self, in_channels, out_channels, k_size):
        super(ConvBNRelu_D, self).__init__(
        nn.Conv1d(in_channels, out_channels, k_size, stride=1, padding=(k_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.01))
    

class Up_D(nn.Sequential):
    def __init__(self, inc, outc, k_size, s):
        super().__init__(
            nn.ConvTranspose1d(inc, outc, k_size, s),
            ConvBNRelu_D(outc, outc, k_size=5))


class UNet_D(nn.Module):
    def __init__(self, n=18):
        super(UNet_D, self).__init__()
        self.d2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.d4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.rcbn1 = ConvBNRelu_D(1, n, k_size=25)
        self.rcbn2 = ConvBNRelu_D(n, n, k_size=7)
        self.rcbn3 = ConvBNRelu_D(n, n, k_size=5)
        self.rcbn4 = ConvBNRelu_D(n, n, k_size=3)

        self.up1 = Up_D(n, n, k_size=4, s=4)
        self.up2 = Up_D(n*2, n, k_size=4, s=4)
        self.up3 = Up_D(n*2, n, k_size=2, s=2)
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