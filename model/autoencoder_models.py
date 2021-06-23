import torch
import torch.nn.functional as F
from torch import nn

from functools import partial
# swish custom activation, basically for fun. am bored
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i*i.sigmoid()
        ctx.save_for_backward(result,i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result,i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result+sigmoid_x*(1-result))


class Swish_module(nn.Module):
    def forward(self,x):
        return swish(x)
    
class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super(ConvBNrelu, self).__init__(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(p)
#         Swish_module(),
)

class ResConvBNrelu(nn.Module):
    """convolution => [BN] => ReLU => inplace addition of input"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__()
        assert k_size % 1 == 0, "even number kernel sizes will cause shape mismatch"
        self.resblock = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
#             Swish_module(),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.resblock(x)+x


class ResUp(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU => inplace addition of input"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResConvBNrelu(out_channels, out_channels, kernel_size=kernel_size, p=p))
        
class Convrelu(nn.Sequential):
    """convolution => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super(Convrelu, self).__init__(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(),
        nn.Dropout(p))

class ConvBNreluDouble(nn.Sequential):
    """convolution => [BN] => ReLU => convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super(ConvBNreluDouble, self).__init__(
            ConvBNrelu(in_channels, out_channels, kernel_size, p),
            ConvBNrelu(out_channels, out_channels, kernel_size, p))

class UpnoBN(nn.Sequential):
    """transpose convolution => convolution => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            Convrelu(out_channels, out_channels, kernel_size=kernel_size, p=0))

class Up(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ConvBNrelu(out_channels, out_channels, kernel_size=kernel_size, p=p))

class LongUp(nn.Sequential):
    """transpose convolution => ReLU convolution => [BN] => ReLU => convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            ConvBNreluDouble(out_channels, out_channels, kernel_size=kernel_size, p=p))

        
downsample_options = {
    'ConvBNrelu':ConvBNrelu,
    'ResConvBNrelu':ResConvBNrelu,
    'ConvBNreluDouble':ConvBNreluDouble,
    'Convrelu':Convrelu,
}

upsample_options = {
    'LongUp':LongUp,
    'Up':Up,
    'UpnoBN':UpnoBN,
}

def combine(x, y, mode='concat'):
    if mode == 'concat':
        return torch.cat([x, y], dim=1)
    elif mode == 'add':
        return x+y
    else:
        raise RuntimeError(f'''Invalid option {mode} from choices 'concat' or 'add' ''')
    
# ======================================================================
# U-Net Models
# ======================================================================
def combine(x, y, mode='concat'):
    if mode == 'concat':
        ret = torch.cat([x, y], dim=-2)
        return ret
    
    elif mode == 'add':
        return x+y
    else:
        raise RuntimeError(f'''Invalid option {mode} from choices 'concat' or 'add' ''')
    
class UNet(nn.Module):
    def __init__(self,
                 n=32,
                 sc_mode='concat',
                 dropout_p=0,
                 d_selection='ConvBNrelu',
                 u_selection='Up'
                ):
        super().__init__()
        if sc_mode == 'concat': 
            factor = 2
        else: 
            factor = 1
        self.mode = sc_mode
        self.p = dropout_p
        
        assert d_selection in downsample_options.keys(), f'Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}'
        assert u_selection in upsample_options.keys(), f'Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}'
        
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]
                
        self.rcbn1 = d_block(1, n, kernel_size = 25, p=dropout_p)
        self.rcbn2 = d_block(n, n, kernel_size = 7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size = 5, p=dropout_p)
        self.rcbn4 = d_block(n, n, kernel_size = 5, p=dropout_p)
        self.rcbn5 = d_block(n, n, kernel_size = 5, p=dropout_p)

        self.up1 = u_block(n, n, kernel_size = 5, p=dropout_p)
        self.up2 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        self.up3 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        self.up4 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        self.out_intermediate = nn.Conv1d(n*factor, n, 5, padding=2)
        self.outc = nn.Conv1d(n, 1, 5, padding=2)
        
        self.d = nn.MaxPool1d(2)

    def forward(self, x):
#         x = torch.cat([x, x[:, ::-1, :]], dim=0) experimental - flip samples in a batch to try and  learn symmetrical kernels 
        
        x1 = self.rcbn1(x) # 4000
        x2 = self.d(self.rcbn2(x1)) # 2000
        x3 = self.d(self.rcbn3(x2)) # 1000
        x4 = self.d(self.rcbn4(x3)) # 500
        x = self.d(self.rcbn5(x4)) # 250

        x = self.up1(x) # 500
        x = self.up2(combine(x, x4, mode=self.mode)) # 1000
        x = self.up3(combine(x, x3, mode=self.mode)) # 2000
        x = self.up4(combine(x, x2, mode=self.mode)) # 4000
        x = self.out_intermediate(combine(x, x1, mode=self.mode)) # 4000
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return  ret


    
class ಠ_ಠ_noSC(nn.Module):
    def __init__(self, n=32, dropout_p=0):
        super().__init__()
        self.d = nn.MaxPool1d(2)
        
        self.rcbn1 = ConvBNrelu(1, n, k_size = 25, p = dropout_p)
        self.rcbn2 = ConvBNrelu(n, n, k_size = 7, p = dropout_p)
        self.rcbn3 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn4 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn5 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)

        self.up1 = Up(n, n, k_size = 5, p = dropout_p)
        self.up2 = Up(n, n, k_size = 5, p = dropout_p)
        self.up3 = Up(n, n, k_size = 5, p = dropout_p)
        self.up4 = Up(n, n, k_size=5, p = dropout_p)
        self.outc = nn.Conv1d(n, 1, 5, padding=2, p = dropout_p)

    def forward(self, x):
        x1 = self.rcbn1(x)
        x2 = self.d(self.rcbn2(x1))# 2000       
        x3 = self.d(self.rcbn3(x2))# 1000
        x4 = self.d(self.rcbn4(x3)) # 500
        x = self.d(self.rcbn5(x4)) # 250

        x = self.up1(x) # 500
        x = self.up2(x) # 1000
        x = self.up3(x) #2000
        
        x = self.up4(x) #4000

        logits_x0 = self.outc(x)

        ret = torch.nn.Softplus()(logits_x0).squeeze()
        return  ret

class ಠ_ಠ_Residual(nn.Module):
    def __init__(self, n=32, sc_mode='concat', dropout_p=0):
        super().__init__()
        self.mode = sc_mode
        if sc_mode == 'concat': 
            factor = 2
        else: 
            factor = 1
        self.d = nn.MaxPool1d(2)
        
        self.rcbn1 = ConvBNrelu(1, n, k_size = 25, p = dropout_p)
        self.rcbn2 = ConvBNrelu(n, n, k_size = 7, p = dropout_p)
        self.rcbn3 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn4 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn5 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)

        self.up1 = ResUp(n, n, k_size = 5, p = dropout_p)
        self.up2 = ResUp(n*factor, n, k_size = 5, p = dropout_p)
        self.up3 = ResUp(n*factor, n, k_size = 5, p = dropout_p)
        self.up4 = ResUp(n*factor, n, k_size=5, p = dropout_p)
        self.outc = nn.Conv1d(n*factor, 1, 5, padding=2, p = dropout_p)

    def forward(self, x):
        x1 = self.rcbn1(x)
        x2 = self.d(self.rcbn2(x1))       
        x3 = self.d(self.rcbn3(x2))
        x4 = self.d(self.rcbn4(x3))
        x = self.d(self.rcbn5(x4)) 

        x = self.up1(x)
        x = self.up2(combine(x, x4, mode=self.mode))
        x = self.up3(combine(x, x3, mode=self.mode))
        x = self.up4(combine(x, x2, mode=self.mode))

        logits_x0 = self.outc(combine(x, x1, mode=self.mode))

        ret = F.softplus(logits_x0).squeeze(1)
        return  ret 
    

# ======================= Perturbative Models ====================================
class PerturbativeUNet(nn.Module):
    def __init__(self, args, n, sc_mode='concat', dropout_p=0):
        super().__init__()
        self.mode = sc_mode
        if sc_mode == 'concat': 
            factor = 2
        else: 
            factor = 1
        
        # network for perturbative features
        self.cbn1_x = ConvBNrelu(2, n, k_size = 11, p = dropout_p)
        self.cbn2_x = ConvBNrelu(n, n, p = dropout_p)
        self.cbn3_x = ConvBNrelu(n, n, p = dropout_p)
        self.cbn4_x = ConvBNrelu(n, n, p = dropout_p)
        self.up1_x = Up(n, n, p = dropout_p)
        self.up2_x = Up(n*factor, n, p = dropout_p)
        self.up3_x = Up(n*factor, n, p = dropout_p)
        self.up4_x = Up(n*factor, n, p = dropout_p)

        self.down = nn.MaxPool1d(2)
        self.d = nn.MaxPool1d(2)
        
        # network for X features
        self.rcbn1 = ConvBNrelu(1, n, k_size = 25, p = dropout_p)
        self.rcbn2 = ConvBNrelu(n, n, k_size = 7, p = dropout_p)
        self.rcbn3 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn4 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)
        self.rcbn5 = ConvBNrelu(n, n, k_size = 5, p = dropout_p)

        self.up1 = Up(n, n, k_size = 5, p = dropout_p)
        self.up2 = Up(n*factor, n, k_size = 5, p = dropout_p)
        self.up3 = Up(n*factor, n, k_size = 5, p = dropout_p)
        self.up4 = Up(n*factor, n, k_size=5, p = dropout_p)
        self.out_intermediate = nn.Conv1d(n*factor, n, 5, padding=2)

        self.outc_larger = nn.Conv1d(factor*n, 1, 3, padding=1)

    def forward(self, x):
        X = x[:, 0:1, :] # one-slice prevents need for .unsqueeze()
        x_y = x[:, -2:, :]
        
        # x / y  feature
        p_x = self.cbn1_x(x_y) 
        p_x = self.down(p_x)

        p_x2 = self.cbn2_x(p_x)
        p_x = self.down(p_x2)
        
        p_x3 = self.cbn3_x(p_x)
        p_x = self.down(p_x3)

        p_x4 = self.cbn4_x(p_x)
        p_x = self.down(p_x4)

        p_x = self.up1_x(p_x)
        p_x = self.up2_x(combine(p_x, p_x4, mode=self.mode))
        p_x = self.up3_x(combine(p_x, p_x3, mode=self.mode))
        logits_x1 = self.up4_x(combine(p_x, p_x2, mode=self.mode))

        # X feature based on U-Net
        x1 = self.rcbn1(X) # 4000
        x2 = self.d(self.rcbn2(x1)) # 2000
        x3 = self.d(self.rcbn3(x2)) # 1000
        x4 = self.d(self.rcbn4(x3)) # 500
        x = self.d(self.rcbn5(x4)) # 250

        x = self.up1(x) # 500
        x = self.up2(combine(x, x4, mode=self.mode)) # 1000
        x = self.up3(combine(x, x3, mode=self.mode)) # 2000
        x = self.up4(combine(x, x2, mode=self.mode)) # 4000
        logits_x0 = self.out_intermediate(combine(x, x1, mode=self.mode))

        logits_X_and_x = self.outc_larger(combine(logits_x0, logits_x1, mode=self.mode))

        ret = F.softplus(logits_X_and_x).squeeze(1)
        return  ret

class OGPerturbativeUNet(nn.Module):
    def __init__(self, args, n):
        super().__init__(args)
        self.cbn1_x = ConvBNrelu(2, n, k_size = 11)
        self.cbn2_x = ConvBNrelu(n, n)
        self.cbn3_x = ConvBNrelu(n, n)
        self.cbn4_x = ConvBNrelu(n, n)
        self.up1_x = Up(n, n)
        self.up2_x = Up(n*2, n)
        self.up3_x = Up(n*2, n)
        self.up4_x2 = Up(n*2, 1)

        self.down = nn.MaxPool1d(2)
        self.d = nn.MaxPool1d(2)
        
        self.rcbn1 = ConvBNrelu(1, n, k_size = 25)
        self.rcbn2 = ConvBNrelu(n, n, k_size = 7)
        self.rcbn3 = ConvBNrelu(n, n, k_size = 5)
        self.rcbn4 = ConvBNrelu(n, n, k_size = 5)
        self.rcbn5 = ConvBNrelu(n, n, k_size = 5)

        self.up1 = Up(n, n, k_size = 5)
        self.up2 = Up(n*2, n, k_size = 5)
        self.up3 = Up(n*2, n, k_size = 5)
        self.up4 = Up(n*2, n, k_size=5)
#         self.out_intermediate = nn.Conv1d(n*2, 1, 5, padding=2)
        self.out_intermediate = nn.Conv1d(n*2, n, 5, padding=2)
        self.outc = nn.Conv1d(n, 1, 5, padding=2)

    def forward(self, x):
        X = x[:, 0, :].unsqueeze(1)    
        x_y = 100*x[:, -2:, :]
        
        # x / y  feature
        p_x = self.cbn1_x(x_y) 
        p_x = self.down(p_x)

        p_x2 = self.cbn2_x(p_x)
        p_x = self.down(p_x2)
        
        p_x3 = self.cbn3_x(p_x)
        p_x = self.down(p_x3)

        p_x4 = self.cbn4_x(p_x)
        p_x = self.down(p_x4)

        p_x = self.up1_x(p_x)
        p_x = self.up2_x(torch.cat([p_x, p_x4], 1))
        p_x = self.up3_x(torch.cat([p_x, p_x3], 1))        
        logits_x1 = self.up4_x2(torch.cat([p_x, p_x2], 1))

        # X feature based on U-Net
        x1 = self.rcbn1(X)
        x2 = self.d(self.rcbn2(x1))# 2000       
        x3 = self.d(self.rcbn3(x2))# 1000
        x4 = self.d(self.rcbn4(x3)) # 500
        x = self.d(self.rcbn5(x4)) # 250

        x = self.up1(x) # 500
        x = self.up2(torch.cat([x, x4], 1)) # 1000
        x = self.up3(torch.cat([x, x3], 1)) #2000
        x = self.up4(torch.cat([x, x2], 1)) #4000
        x = self.out_intermediate(torch.cat([x, x1], 1))
        logits_x0 = self.outc(x)

        logits_X_and_x = logits_x0*logits_x1

        ret = torch.nn.Softplus()(logits_X_and_x).squeeze()
        return  ret

# Trying some stuff with inception blocks with a cardinality of 4 
# https://arxiv.org/pdf/1611.05431.pdf


# class ResidualConvBNrelu(nn.Module):
#     """convolution => [BN] => Pooling => ReLU"""
#     def __init__(self, INC, OUTC, k_size=3):
#         super(ResidualConvBNrelu, self).__init__()
#         self.residBlock = nn.Sequential()
#         nn.Conv1d(INC, OUTC, k_size, stride = 1, padding = (k_size-1)//2),
#         nn.BatchNorm1d(OUTC),
#         nn.ReLU())


class WeirdResidualAutoencoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.d = nn.MaxPool1d(2)

        self.rcbn1 = ResidualConvBNrelu(1, n, k_size = 25)
        self.rcbn2 = ResidualConvBNrelu(n, n)
        self.rcbn3 = ResidualConvBNrelu(n, n)
        self.rcbn4 = ResidualConvBNrelu(n, n)
        self.rcbn5 = ResidualConvBNrelu(n, n)

        self.up1 = Up(n, n)
        self.up2 = Up(n*2, n)
        self.up3 = Up(n*2, n)
        self.up4 = Up(n*2, n)
        self.outc = nn.Conv1d(n, 1, 3, padding=1)

    def forward(self, x):
        x = self.rcbn1(x)
        x1 = self.d(self.rcbn2(x))# 2000       
        x2 = self.d(self.rcbn3(x1))# 1000
        x3 = self.d(self.rcbn4(x2)) # 500
        x = self.d(self.rcbn5(x3)) # 250

        x = self.up1(x) # 500
        x = self.up2(torch.cat([x, x3], 1)) # 1000
        x = self.up3(torch.cat([x, x2], 1)) #2000
        
        x = self.up4(torch.cat([x, x1], 1)) #4000

        logits_x0 = self.outc(x)

        ret = torch.nn.Softplus()(logits_x0).squeeze()
        return  ret
    
    
    
class ConvBNleaky(nn.Sequential):
    """convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels, k_size=3, stride=1):
        super(ConvBNleaky, self).__init__(
        nn.Conv1d(in_channels, out_channels, k_size, stride=stride, padding=(k_size-1)//2),
#         nn.BatchNorm1d(out_channels),
        nn.LeakyReLU())


# ===================== NOT AUTOENCODER MODEL =============================================

# here for convenience
class Conv5fc1(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv1 = ConvBNleaky(
            in_channels=1,
            out_channels=n,
            k_size=25)

        self.conv2 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv3 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv4 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv5 = ConvBNleaky(
            in_channels=n,
            out_channels=1,
            k_size=7)

        self.fc = nn.Linear(4000, 4000)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x.squeeze())

        x = self.softplus(x)
        return x
    
class stupid(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv1 = ConvBNleaky(
            in_channels=1,
            out_channels=1,
            k_size=3)


    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class Conv5fc1_SC(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv1 = ConvBNleaky(
            in_channels=1,
            out_channels=n,
            k_size=25)

        self.conv2 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv3 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv4 = ConvBNleaky(
            in_channels=2*n,
            out_channels=n,
            k_size=7)


        self.conv5 = ConvBNleaky(
            in_channels=2*n,
            out_channels=1,
            k_size=7)

        self.fc = nn.Linear(4000, 4000)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.conv3(x2)
        x = self.conv4(torch.cat( [x, x2], 1))
        x = self.conv5(torch.cat( [x, x1], 1))
        x = self.fc(x.squeeze())

        x = self.softplus(x)
        return x

class Conv6_SC(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv1 = ConvBNleaky(
            in_channels=1,
            out_channels=n,
            k_size=25)

        self.conv2 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv3 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv4 = ConvBNleaky(
            in_channels=n,
            out_channels=n,
            k_size=7)

        self.conv5 = ConvBNleaky(
            in_channels=2*n,
            out_channels=n,
            k_size=7)

        self.conv6 = ConvBNleaky(
            in_channels=2*n,
            out_channels=1,
            k_size=7)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.conv3(x2)
        x = self.conv3(x)
        x = self.conv5(torch.cat( [x, x2], 1))
        x = self.conv6(torch.cat( [x, x1], 1))
        x = self.softplus(x)
        return x


