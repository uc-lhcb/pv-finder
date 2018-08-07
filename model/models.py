from torch import nn

class SimpleCNN2Layer(nn.Module):
    def __init__(self):
        super(SimpleCNN2Layer, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels = 1,
            out_channels = 5,
            kernel_size = 25,
            stride = 1,
            padding = 12
        )
        
        self.conv2 = nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 15,
            stride = 1,
            padding = 7
        )
            
        self.fc1 = nn.Linear(
            in_features = 4000*self.conv2.out_channels,
            out_features = 4000
        )
        
        self.conv1 = nn.DataParallel(self.conv1)
        self.conv2 = nn.DataParallel(self.conv2)
        self.fc1 = nn.DataParallel(self.fc1)
        
       
    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        
        x = leaky(self.conv1(x))
        x = leaky(self.conv2(x))
        
        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])
        
        x = nn.functional.sigmoid(self.fc1(x))
        
        return x
    
    
class SimpleCNN3Layer(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer, self).__init__()
        
        self.conv1=nn.Conv1d(
            in_channels = 1,
            out_channels = 10,
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )
        
        assert self.conv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        
        
        self.conv2=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 5,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )
        
        assert self.conv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        
        
        self.conv3=nn.Conv1d(
            in_channels = self.conv2.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )
        
        assert self.conv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        

        self.conv3dropout = nn.Dropout(0.35)
        
        self.fc1 = nn.Linear(
            in_features = 4000 * self.conv2.out_channels,
            out_features = 4000)
       
    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        
        x = leaky(self.conv1(x))
        x = leaky(self.conv2(x))
        x = leaky(self.conv3(x))
        
        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])
        
        x = self.conv3dropout(x)
        x = self.fc1(x)
        
        x = nn.functional.sigmoid(x)
        
        return x