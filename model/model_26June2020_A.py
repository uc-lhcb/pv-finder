from torch import nn
import torch

class Block(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Block, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

class Up(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )

class UNet(nn.Module):
    def __init__(self, n=32):
        super(UNet, self).__init__()
        
        self.block1 = Block(1, 24, kernel_size=25)
        self.block2 = Block(24, n)
        self.block3 = Block(n, n)
        self.up1 = Up(n, n)
        self.up2 = Up(n*2, n)
        self.up3 = Up(n+24, n)
        self.up4 = Up(n, n)
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
        
        x = torch.nn.Softplus()(x).reshape(64, 4000)
        return x
        