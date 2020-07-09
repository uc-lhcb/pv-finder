from torch import nn
import torch

class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => Pooling => ReLU"""
    def __init__(self, INC, OUTC, k_size=3):
        super(ConvBNrelu, self).__init__(
        nn.Conv1d(INC, OUTC, k_size, stride = 1, padding = (k_size-1)//2),
        nn.BatchNorm1d(OUTC),
        nn.ReLU())
    

class Up(nn.Sequential):
    def __init__(self, inc, outc):
        super().__init__(
            nn.ConvTranspose1d(inc, outc, 2, stride=2),
            ConvBNrelu(outc, outc))


class UNet4SC(nn.Module):
    def __init__(self, n=32):
        super(UNet4SC, self).__init__()
        self.d = nn.MaxPool1d(2)
        
        self.rcbn1 = ConvBNrelu(1, n, k_size = 25)
        self.rcbn2 = ConvBNrelu(n, n)
        self.rcbn3 = ConvBNrelu(n, n)
        self.rcbn4 = ConvBNrelu(n, n)
        self.rcbn5 = ConvBNrelu(n, n)

        self.up1 = Up(n, n)
        self.up2 = Up(n*2, n)
        self.up3 = Up(n*2, n)
        self.up4 = Up(n*2, n)
        self.outc = nn.Conv1d(n*2, 1, 3, padding=1)

    def forward(self, x):
        x1 = self.rcbn1(x)
        x2 = self.d(self.rcbn2(x1))# 2000       
        x3 = self.d(self.rcbn3(x2))# 1000
        x4 = self.d(self.rcbn4(x3)) # 500
        x = self.d(self.rcbn5(x4)) # 250

        x = self.up1(x) # 500
        x = self.up2(torch.cat([x, x4], 1)) # 1000
        x = self.up3(torch.cat([x, x3], 1)) #2000
        
        x = self.up4(torch.cat([x, x2], 1)) #4000

        logits_x0 = self.outc(torch.cat([x, x1], 1))

        ret = torch.nn.Softplus()(logits_x0).squeeze()
        return  ret
    
    
    