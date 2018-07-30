import torch
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN2Layer(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN2Layer, self).__init__()
        # input channel size 1, output channel size 15
        self.nChan_out_layer1 = 5
        kernel_size_layer1 = 25
        stride_layer1 = 1
        padding_layer1 = 12
        self.conv1=torch.nn.Conv1d(1, self.nChan_out_layer1, kernel_size_layer1, stride_layer1, padding_layer1)
        
        self.nChan_out_layer2 = 1
        kernel_size_layer2 = 15
        stride_layer2 = 1
        padding_layer2 = 7
        self.conv2=torch.nn.Conv1d(self.nChan_out_layer1, self.nChan_out_layer2, kernel_size_layer2, stride_layer2, padding_layer2)        

        self.fc1 = torch.nn.Linear(4000*self.nChan_out_layer2, 4000)
       
    def forward(self, x):
        leaky = torch.nn.LeakyReLU(0.01)
        x= leaky(self.conv1(x))
        x= leaky(self.conv2(x))
        x= x.view(-1, 4000*self.nChan_out_layer2)
        x = F.sigmoid(self.fc1(x))
        return(x)
    
class SimpleCNN3Layer(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer, self).__init__()
        # input channel size 1, output channel size 15
        self.nChan_out_layer1 = 10
        kernel_size_layer1 = 25
        stride_layer1 = 1
        assert kernel_size_layer1 % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding_layer1 = (kernel_size_layer1 - 1) // 2
        self.conv1=torch.nn.Conv1d(1, self.nChan_out_layer1, kernel_size_layer1, stride_layer1, padding_layer1)
        
        self.nChan_out_layer2 = 5
        kernel_size_layer2 = 15
        stride_layer2 = 1
        assert kernel_size_layer2 % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding_layer2 = (kernel_size_layer2 - 1) // 2
        self.conv2=torch.nn.Conv1d(self.nChan_out_layer1, self.nChan_out_layer2, kernel_size_layer2, stride_layer2, padding_layer2)        

        self.nChan_out_layer3 = 1
        kernel_size_layer3 = 5
        stride_layer3 = 1
        assert kernel_size_layer3 % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding_layer3 = (kernel_size_layer3 - 1) // 2
        self.conv3=torch.nn.Conv1d(self.nChan_out_layer2, self.nChan_out_layer3, kernel_size_layer3, stride_layer3, padding_layer3) 
        self.conv3OutputDropout = torch.nn.Dropout(0.35)
        
        self.fc1 = torch.nn.Linear(4000*self.nChan_out_layer3, 4000)
       
    def forward(self, x):
        leaky = torch.nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = leaky(self.conv2(x))
        x = leaky(self.conv3(x))
        x= x.view(-1, 4000*self.nChan_out_layer3)
        x = self.conv3OutputDropout(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return(x)
    
    
def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return output