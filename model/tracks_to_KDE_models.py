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
# Transformer Models
# ======================================================================
class TorchTransformer(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim, n_layers, dropout_rate=0):
        super(TorchTransformer, self).__init__()
        
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.project_linear = nn.Linear(input_dim, hidden_dim)
        self.embedded_transform = nn.Linear(hidden_dim, 4000)

        self.up1 = Up(1, 32) # 125 -> 250
        self.cbr1 = ConvBNrelu(32, 32, 25)
        self.up2 = Up(32, 32) # 250 -> 500
        self.cbr2 = ConvBNrelu(32, 32, 25)
        self.up3 = Up(32, 32) # 500 -> 1000
        self.cbr3 = ConvBNrelu(32, 32, 25)
        self.up4 = Up(32, 32) # 1000 -> 2000
        self.cbr4 = ConvBNrelu(32, 32, 25)
        self.up5 = Up(32, 1) # 2000 -> 4000
        
    def forward(self, x):
        mask = ~(x[:, 0, :] > -98)
        mask[:, 0] = False
        
        # skipping for simplicity of training code / might not need with project_linear layer 
        x -= self.means.unsqueeze(0).unsqueeze(-1)
        x /= self.stds.unsqueeze(0).unsqueeze(-1) #[16, 600, 64] N,n_tracks,embedding_dim)
        
        x = self.project_linear(x.transpose(2, 1)).transpose(1, 0) # (L, N, E)
        x = self.transformer(x, src_key_padding_mask = mask)
        x = x.transpose(0, 1)
#         x_perturbative = self.embedded_transform(x)
        x = x.mean(-2).unsqueeze(1)
        x_perturbative = self.embedded_transform(x)
#         x = self.up1(x)
#         x = self.cbr1(x)
#         x = self.up2(x)
#         x = self.cbr2(x)
#         x = self.up3(x)
#         x = self.cbr3(x)
#         x = self.up4(x)
#         x = self.cbr4(x)
#         x = self.up5(x) 
        return x_perturbative.squeeze(1)
#         return x.squeeze(1)
#         return torch.clip(x_perturbative, min=0, max=0.8).squeeze(1)
#         return torch.clip(x, min=0, max=0.8).squeeze(1)
    
    
    