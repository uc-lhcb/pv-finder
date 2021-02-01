import torch
from torch import nn
import numpy as np

class SimpleCNN5Layer_Ca(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x

'''
Same as SimpleCNN5Layer_Ca but with 16 channels in the first layer and 9 in the hidden layers
'''
class SimpleCNN5Layer_Ca_B(nn.Module):
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_B, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
    
class SimpleCNN5Layer_Ca_F(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_F, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=10,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x

class SimpleCNN5Layer_Ca_G(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_G, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=12,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=8,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
       
class SimpleCNN5Layer_Ca_H(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_H, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=12,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=3,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
    
class SimpleCNN5Layer_Ca_I(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_I, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=6,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x    
    
class SimpleCNN5Layer_Ca_J(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_J, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
    
class SimpleCNN5Layer_Ca_K(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_K, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
    
class SimpleCNN5Layer_Ca_L(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_L, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
      
class SimpleCNN5Layer_Ca_M(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_M, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=6,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
     
class SimpleCNN5Layer_Ca_N(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_N, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x
    
class SimpleCNN5Layer_Ca_O(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_O, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=35,
            stride=1,
            padding=(35 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv5.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv5dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x


'''
Benchmark network architecture from Dr. Sokoloff
'''
class All_CNN6Layer_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_A, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x

'''
Same as All_CNN6Layer_A but with less channels (16-9-9-9-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_B(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_B, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x

'''
Same as All_CNN6Layer_A but with one skip connection
This is a test to see whether skip connections increases performance, which would allow for a reduced network size
or better efficiency/false positive ratio.
'''
class All_CNN6Layer_C(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_C, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3a = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3a.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3a.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x1 = leaky(self.conv1(x))
        x1 = self.conv1dropout(x1)
        x2 = leaky(self.conv2(x1))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.conv3a(torch.cat([x1, x2], 1)))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.conv4(x3))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.conv5(x4))
        x = self.conv5dropout(x5)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x

'''
Same as All_CNN6Layer_A but with Batch Normalization.
This is a test to see whether BatchNorm increases performance, which would allow for a reduced network size.
'''
class All_CNN6Layer_D(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.bn1(self.conv1(x)))
        x = self.conv1dropout(x)
        x = leaky(self.bn2(self.conv2(x)))
        x = self.conv2dropout(x)
        x = leaky(self.bn3(self.conv3(x)))
        x = self.conv3dropout(x)
        x = leaky(self.bn4(self.conv4(x)))
        x = self.conv4dropout(x)
        x = leaky(self.bn5(self.conv5(x)))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x

'''
Synthesis of all the previous ALL_CNN6Layer_A-D models.
'''
class All_CNN6Layer_E(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_E, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3a = nn.Conv1d(
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3a.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3a.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

        self.bn1 = nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3a.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x1 = leaky(self.bn1(self.conv1(x)))
        x1 = self.conv1dropout(x1)
        x2 = leaky(self.bn2(self.conv2(x1)))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.bn3(self.conv3a(torch.cat([x1, x2], 1))))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.bn4(self.conv4(x3)))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.bn5(self.conv5(x4)))
        x = self.conv5dropout(x5)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_A but with half channels (10-5-5-5-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_F(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_F, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=10,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_A but with decreasing channel size (20-12-8-6-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_G(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_G, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=12,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=8,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_G but with more decreased channel size (12-9-6-3-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_H(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_H, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=12,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=3,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_A but with lowered, flat channel size (6-6-6-6-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_I(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_I, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=6,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_I but with more channels (9-9-9-9-1)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_J(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_J, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_J but with smaller kernels (15-11-11-11-5)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_K(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_K, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=9,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
    
'''
Same as All_CNN6Layer_A but with smaller kernels (15-9-9-9-91)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_L(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_L, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=20,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=10,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
       
'''
Same as All_CNN6Layer_A but with low channel count (6-6-6-6-1) and a smaller kernel size (11-9-9-9-5)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_M(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_M, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=6,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=6,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
       
'''
Same as All_CNN6Layer_A but with low channel count (16-9-9-9-1) and a smaller kernel size (15-11-11-11-5)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_N(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_N, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x
        
'''
Same as All_CNN6Layer_A but with low channel count (16-9-9-9-1) and a smaller kernel size (35-25-25-25-10)
This is an unprincipled approach to reduce the network size.
'''
class All_CNN6Layer_O(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(All_CNN6Layer_O, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=35,
            stride=1,
            padding=(35 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=9,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=11,
            stride=1,
            padding=(11 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4(x))
        x = self.conv4dropout(x)
        x = leaky(self.conv5(x))
        x = self.conv5dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x