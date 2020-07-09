import torch
import torch.nn as nn
import numpy as np

"""
class Conv creates a convolutional block that can be modularly changed based on the layer to have different input channel counts, output
channel counts, kernel sizes, and droupout rates.
"""
class Conv(nn.Module):
    """
    in_channels - the number of input channels that the block will operate on
    out_channels - the number of output channels that the block will output
    kernel_size - the size of the one-dimensional kernel/filter
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0):
        super(Conv, self).__init__()
        """
        self.conv is the convolutional block. Sequential is used to create this block, with parameters given by the Conv parameters in the
        definition statement. The arguments in the Sequential block serve the following function:

        -Conv1d - performs the bulk of the block's computation and "machine learning"
        -LeakyReLU - Used over softplus (used in SimpleCNN5Layer_Ca) because it "saturates less completely" and is generally considered
        better than a softplus (NOT APPLICABLE)
        -Dropout - reduces overfitting and has some regularization effect (NOT APPLICABLE)
        -MaxPool1d is commented out because it may not be necessary to downsample the model. Though downsampling the model helps reduce
        overfitting by providing an abstract form of representation and can reduce computational cost, it does not do the latter in this 
        case. This is because the model must preserve its dimension of 1 x 4000 x C (a.k.a dimension purity) to the output. Thus, performing
        a MaxPool downsample means that upsampling ought to be done before the next convolutional block. This means that the model will not
        benefit from the reduced number of parameters that MaxPool should afford.
        """
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            # nn.BatchNorm1d(out_channels),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.conv(x)

"""
class Net implements Conv, applies a fully-connected layer, and creates a neural network with all necessary methodology for pv-finder. This
Net is based off of SimpleCNN5Layer_Ca with some important changes:
1. The final activation function is a relu instead of a softplus (NOT APPLICABLE)
2. Different dropout rates are used for the first and last convolutional layer
3. Kernel sizes are smaller (15, 7, 7, 5, 3) instead of (25, 15, 15, 15, 5)
4. BatchNorm is used (NOT APPLICABLE)
!!!! 5. MaxPool is used (DISREGARD) !!!! 
"""
class CNN5Blocks(nn.Module):
    def __init__(self):
        super(CNN5Blocks, self).__init__()
        # 5 layers are used to hold the size of the model fixed with the model of SimpleCNN5Layer_Ca. This model's performance will then be
        # compared with the aforementioned one.
        self.conv1 = Conv(1, 20, 25, 0.15)
        self.conv2 = Conv(20, 10, 15, 0.15)
        self.conv3 = Conv(10, 10, 15, 0.15)
        self.conv4 = Conv(10, 10, 15, 0.15)
        self.conv5 = Conv(10, 1, 5, 0.35)
        # Apply a fully connected layer at the end
        self.fc = nn.Linear(
            in_features=4000 * 1, out_features=4000
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        x = self.relu(x).reshape(64, 4000)
        return x