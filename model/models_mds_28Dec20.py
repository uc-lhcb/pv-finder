import torch
from torch import nn
import numpy as np


class SimpleCNN2Layer(nn.Module):
    def __init__(self):
        super(SimpleCNN2Layer, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=5, kernel_size=25, stride=1, padding=12
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=1,
            kernel_size=15,
            stride=1,
            padding=7,
        )

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv2.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)

        x = leaky(self.conv1(x))
        x = leaky(self.conv2(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = torch.sigmoid(self.fc1(x))

        return x


class SimpleCNN3Layer(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)

        x = leaky(self.conv1(x))
        x = leaky(self.conv2(x))
        x = leaky(self.conv3(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv3dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN3Layer_A(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer_A, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = leaky(self.conv3(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv3dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN4Layer_A(nn.Module):
    def __init__(self):
        super(SimpleCNN4Layer_A, self).__init__()

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
            out_channels=25,
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
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
        )

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.35)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        ## print('x.shape after conv1 = ',x.shape)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        ## print('x.shape after conv2 = ', x.shape)
        x = leaky(self.conv3(x))
        ## print('x.shape after conv3 = ', x.shape)

        x = self.conv4(x)
        ## print('x.shape after conv4 = ',x.shape)

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN3Layer_B(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer_B, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv3dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN3Layer_C(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer_C, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv3dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class All_CNN3Layer_C(nn.Module):
    def __init__(self):
        super(All_CNN3Layer_C, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        ## x = leaky(self.conv3(x))
        ## 180825 try removing the fully connected layer and simply
        ## use the output of the third CVN layer as  input to
        ## the sigmoid function producing the predicted values
        x = self.conv3(x)

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        ## mds x = self.conv3dropout(x)
        ## mds x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN4Layer_C(nn.Module):
    def __init__(self):
        super(SimpleCNN4Layer_C, self).__init__()

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
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN4Layer_D(nn.Module):
    def __init__(self):
        super(SimpleCNN4Layer_D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.55)
        self.conv2dropout = nn.Dropout(0.55)
        self.conv3dropout = nn.Dropout(0.55)
        self.conv4dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN4Layer_D35(nn.Module):
    def __init__(self):
        super(SimpleCNN4Layer_D35, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.35)
        self.conv2dropout = nn.Dropout(0.35)
        self.conv3dropout = nn.Dropout(0.35)
        self.conv4dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN4Layer_D35_sp(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(SimpleCNN4Layer_D35_sp, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.35)
        self.conv2dropout = nn.Dropout(0.35)
        self.conv3dropout = nn.Dropout(0.35)
        self.conv4dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = self.softplus(x)

        return x


class SimpleCNN4Layer_D25(nn.Module):
    def __init__(self):
        super(SimpleCNN4Layer_D25, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.25)
        self.conv2dropout = nn.Dropout(0.25)
        self.conv3dropout = nn.Dropout(0.25)
        self.conv4dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


class SimpleCNN5Layer_C(nn.Module):
    def __init__(self):
        super(SimpleCNN5Layer_C, self).__init__()

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

        x = torch.sigmoid(x)

        return x


class SimpleCNN5Layer_Ca(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca, self).__init__()

        self.counter = 0
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

        print("x.shape = ",x.shape)
        if (self.counter == 0):
          print("x.shape = ",x.shape)
          self.counter = self.counter+1
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


class TwoFeatures_CNN4Layer_D35(nn.Module):
    def __init__(self):
        super(TwoFeatures_CNN4Layer_D35, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.35)
        self.conv2dropout = nn.Dropout(0.35)
        self.conv3dropout = nn.Dropout(0.35)
        self.conv4dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


## TwoFeatures_CNN4Layer_D15 has same architecture as TwoFeatures_CNN4Layer_D35
## but a lower dropout rate.
class TwoFeatures_CNN4Layer_D15(nn.Module):
    def __init__(self):
        super(TwoFeatures_CNN4Layer_D15, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv4.out_channels, out_features=4000
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

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.conv4dropout(x)
        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x


## TwoFeature_CNN5Layer is derived from SimpleCNN5Layer
class TwoFeature_CNN5Layer_Ca(nn.Module):
    ## with Softplus activation
    def __init__(self):
        super(TwoFeature_CNN5Layer_Ca, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
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

        x = torch.nn.softplus(x)

        return x


class TwoFeature_CNN5Layer_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(TwoFeature_CNN5Layer_A, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv4.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv4.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  Dec 23 try dropout 0.05 rather than 0.35 (default) as algorithm is stalling
        ##  in first 20 epochs;  this may provide more flexibility to adapt quickly
        self.conv1dropout = nn.Dropout(0.05)
        self.conv2dropout = nn.Dropout(0.05)
        self.conv3dropout = nn.Dropout(0.05)
        self.conv4dropout = nn.Dropout(0.05)

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

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x


class TwoFeature_CNN6Layer_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(TwoFeature_CNN6Layer_A, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4of6 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5of6 = nn.Conv1d(
            in_channels=self.conv4of6.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5of6.out_channels,
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
        self.conv4of6dropout = nn.Dropout(0.15)
        self.conv5of6dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1dropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4of6(x))
        x = self.conv4of6dropout(x)
        x = leaky(self.conv5of6(x))
        x = self.conv5of6dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x


##  the following model is meant to be used "perturbatively"
##  create the following model from TwoFeature_CNN6Layer_A
##  it is intended to be used after the TwoFeature_CNN6Lyear_A architecture
##  has been used to train a network.  The expectation is that the weights
##  from a model using that for training will be re-used, except for those
##  connecting the first hidden layer to the second hidden layer. For this
##  reason, the first convolutional layer should have a different name than
##  in the TwoFeature model and all additional layers should have the
##  same names.
class FourFeature_CNN6Layer_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(FourFeature_CNN6Layer_A, self).__init__()

        ## as noted in the comments above, the first convolutional layer
        ## should have a different name than use in TwoFeature_CNN6Layer_A
        self.conv1_4Features = nn.Conv1d(
            in_channels=4,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1_4Features.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1_4Features.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4of6 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5of6 = nn.Conv1d(
            in_channels=self.conv4of6.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5of6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_A) to mitigate overtraining
        self.conv1_4Featuresdropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4of6dropout = nn.Dropout(0.15)
        self.conv5of6dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1_4Features(x))
        x = self.conv1_4Featuresdropout(x)
        x = leaky(self.conv2(x))
        x = self.conv2dropout(x)
        x = leaky(self.conv3(x))
        x = self.conv3dropout(x)
        x = leaky(self.conv4of6(x))
        x = self.conv4of6dropout(x)
        x = leaky(self.conv5of6(x))
        x = self.conv5of6dropout(x)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x)
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

        return x


##  190725
## This model is designed as a "perturbative" model.
## The feature set passed in (initial neuron values) is of the form
## (X,Xsq,x,y) where X is the original KDE, Xsq is the element-wise square of X
## and x and y are the values of x and y at each z where the KDE is maximum.
## These feature set will be divided into two parts (X,Xsq) and (x,y) and
## each of these will be run through some convolutional layers to produce
## 4000 bin tensors.
## Then, the product will be taken.
## The hope is that the learning from the (X,Xsq) features can start from
## a previously trained model with the same structure that works well.
## Then, the model will learn a filter that will pass most learned features
## with essentially no change, but will sometimes "mask out" regions where
## we see that changes in (x,y) appear to flag the presence of false positives
## in the original approach.
##
## With luck, this will allow the algorithm to reduce the number of false positives
## for a fixed efficiency, so improve overall performance relative to the same
## architecture processing only (X,Xsq)

##  It is intended to be used after the TwoFeature_CNN6Lyear_A architecture
##  has been used to train a network.  The expectation is that the weights
##  from that model can be used "as is" while the perturbative filter based
##  on (x,y) ia initially trained. Once this training produces a good enough
##  result, all weights can be floated and the training iterated.
class FourFeature_CNN6Layer_B(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(FourFeature_CNN6Layer_B, self).__init__()
        ##
        ##  we will re-use the names of the convolutional layers from TwoFeature_CNN6Layer_A
        ##  for the (X,Xsq) feature set; then use similar (but different) names for
        ##  the layers that process the "pertubative" features (x,y)
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4of6 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5of6 = nn.Conv1d(
            in_channels=self.conv4of6.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5of6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## mds 170925##  now use almost the same names for processing the "perturbative" features
        ## mds 170925        self.pConv1=nn.Conv1d(
        ## mds 170925            in_channels = 2,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 25,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (25 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925
        ## mds 170925        self.pConv2=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv1.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925        self.pConv3=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv2.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925        self.pConv4=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv3.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv4.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925
        ## mds 170925        self.pConv5=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv4.out_channels,
        ## mds 170925            out_channels = 1,
        ## mds 170925            kernel_size = 5,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (5 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv5.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925## the "finalFilter" is meant to replace the fully connected layer with a
        ## mds 170925## convolutional layer that extends over the full range where we saw
        ## mds 170925## significant structure in the 4K x 4K matrix
        ## mds 170925        self.pFinalFilter=nn.Conv1d(
        ## mds 170925            in_channels = self.conv5.out_channels,
        ## mds 170925            out_channels = 1,
        ## mds 170925            kernel_size = 91,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (91 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        ## mds 170925
        ## mds 170925
        #######  190725  create ppConv_n filters to mimic SimpleCNN3Layer

        ##  now use almost the same names for processing the "perturbative" features
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=10,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppFc1 = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,Xsq,x,y)
        ## here, we will use the name x0 to denote the (X,Xsq) feature set and
        ## the name X0 to denote the (x,y) feature set

        ## mds        print('neuronValues.size = ',neuronValues.size())
        ## --> [64,4,4000] for batch size 64
        x0 = neuronValues[:, 0:2, :]  ## picks out the 0 & 1 feature sets, X & Xsq
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4of6(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5of6(x0))
        x0 = self.conv5dropout(x0)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x0)
        x0 = x0.view(x0.shape[0], x0.shape[-1])

        ## mds ##  now repeat the "architecture" for the perturbative element
        ## mds ##  there is no reason it should be the same -- it is probably
        ## mds ##  overkill, but if it works ....
        ## mds         x1 = leaky(self.pConv1(x1))
        ## mds         x1 = self.conv1dropout(x1)
        ## mds         x1 = leaky(self.pConv2(x1))
        ## mds         x1 = self.conv2dropout(x1)
        ## mds         x1 = leaky(self.pConv3(x1))
        ## mds         x1 = self.conv3dropout(x1)
        ## mds         x1 = leaky(self.pConv4(x1))
        ## mds         x1 = self.conv4dropout(x1)
        ## mds         x1 = leaky(self.pConv5(x1))
        ## mds         x1 = self.conv5dropout(x1)
        ## mds         x1 = self.pFinalFilter(x1)
        ## mds         x1 = x1.view(x1.shape[0], x1.shape[-1])
        ## mds
        ## mds         neuronValues = self.softplus(x0*x1)

        ##  now create an "architecture" for the perturbative element
        ##  similar to the original SimpleCNN3Layer  model with
        ##  3 convolutional layers followed by a fully connected layer
        ##  as this began to learn very quickly
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.conv3dropout(x1)
        x1 = self.ppFc1(x1)

        neuronValues = self.softplus(x0 * x1)
        ##        neuronValues = self.softplus(x0)

        return neuronValues


##  190725
## This is designed as a second  "perturbative" model.
## The feature set passed in (initial neuron values) is of the form
## (X,Xsq,x,y) where X is the original KDE, Xsq is the element-wise square of X
## and x and y are the values of x and y at each z where the KDE is maximum.
## These feature set will be divided into two parts (X,Xsq) and (x,y) and
## each of these will be run through some convolutional layers to produce
## 4000 bin tensors.
## Then, the product will be taken.
## The hope is that the learning from the (X,Xsq) features can start from
## a previously trained model with the same structure that works well.
## Then, the model will learn a filter that will pass most learned features
## with essentially no change, but will sometimes "mask out" regions where
## we see that changes in (x,y) appear to flag the presence of false positives
## in the original approach.
##
## With luck, this will allow the algorithm to reduce the number of false positives
## for a fixed efficiency, so improve overall performance relative to the same
## architecture processing only (X,Xsq)

##  It is intended to be used after a first "perturbative" architecture
##  has been used to train a network.  The first architecture will include
##  the original CNN6Layer for the (X,Xsq) features and a CNN + fully
##  connected layer for the (x,y) features.  This model is observed to have
##  too much flexibility (presumably due to the FC layer), and it overtrains
##  very quickly.  This model will replace the FC layer with another CNN
##  layer

##   The expectation is that the "other" weights
##  from that model can be used "as is" while the
##  last layer of the purely CNN filter for (x,y) is
##  (x,y) is initially trained. Once this training produces a good enough
##  result, all weights can be floated and the training iterated.
class FourFeature_CNN6Layer_D(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(FourFeature_CNN6Layer_D, self).__init__()
        ##
        ##  we will re-use the names of the convolutional layers from TwoFeature_CNN6Layer_A
        ##  for the (X,Xsq) feature set; then use similar (but different) names for
        ##  the layers that process the "pertubative" features (x,y)
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4of6 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5of6 = nn.Conv1d(
            in_channels=self.conv4of6.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv5of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv5of6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## mds 170925##  now use almost the same names for processing the "perturbative" features
        ## mds 170925        self.pConv1=nn.Conv1d(
        ## mds 170925            in_channels = 2,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 25,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (25 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925
        ## mds 170925        self.pConv2=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv1.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925        self.pConv3=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv2.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925        self.pConv4=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv3.out_channels,
        ## mds 170925            out_channels = 25,
        ## mds 170925            kernel_size = 15,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (15 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv4.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925
        ## mds 170925        self.pConv5=nn.Conv1d(
        ## mds 170925            in_channels = self.pConv4.out_channels,
        ## mds 170925            out_channels = 1,
        ## mds 170925            kernel_size = 5,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (5 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.pConv5.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' pConv."
        ## mds 170925
        ## mds 170925## the "finalFilter" is meant to replace the fully connected layer with a
        ## mds 170925## convolutional layer that extends over the full range where we saw
        ## mds 170925## significant structure in the 4K x 4K matrix
        ## mds 170925        self.pFinalFilter=nn.Conv1d(
        ## mds 170925            in_channels = self.conv5.out_channels,
        ## mds 170925            out_channels = 1,
        ## mds 170925            kernel_size = 91,
        ## mds 170925            stride = 1,
        ## mds 170925            padding = (91 - 1) // 2
        ## mds 170925        )
        ## mds 170925
        ## mds 170925        assert self.finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        ## mds 170925
        ## mds 170925
        #######  190725  create ppConv_n filters to mimic SimpleCNN3Layer

        ##  now use almost the same names for processing the "perturbative" features
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=10,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppFc1 = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )
        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.conv5of6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,Xsq,x,y)
        ## here, we will use the name x0 to denote the (X,Xsq) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## mds        print('neuronValues.size = ',neuronValues.size())
        ## --> [64,4,4000] for batch size 64
        x0 = neuronValues[:, 0:2, :]  ## picks out the 0 & 1 feature sets, X & Xsq
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4of6(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5of6(x0))
        x0 = self.conv5dropout(x0)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x0)
        x0 = x0.view(x0.shape[0], x0.shape[-1])

        ##  now create an "architecture" for the perturbative element
        ##  similar to the original SimpleCNN3Layer  model with
        ##  3 convolutional layers followed by a fully connected layer
        ##  as this began to learn very quickly
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = self.softplus(x0 * x1)
        ##        neuronValues = self.softplus(x0)

        return neuronValues


##  190817
##  this adds one convolutional layer compared to FourFeature_CNN6Layer_D
##  with great luck, we will be able to start it using the weights for the
##  first four layers from CNN6Layer_D and it will manage the rest on its
##  own, after which all weights can be learned.


class FourFeature_CNN7Layer_D(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(FourFeature_CNN7Layer_D, self).__init__()
        ##
        ##  we will re-use the names of the convolutional  first four layers from
        ##  FourFeature__CNN6Layer_D
        ##  for the (X,Xsq) feature set and all those of the "perturbative" features (x,y)
        ##  the layers that process the "pertubative" features (x,y)
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=25,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.conv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv3 = nn.Conv1d(
            in_channels=self.conv2.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv4of6 = nn.Conv1d(
            in_channels=self.conv3.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv4of6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv5 = nn.Conv1d(
            in_channels=self.conv4of6.out_channels,
            out_channels=25,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  now use almost the same names for processing the "perturbative" features
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=10,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=5,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv2.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv3 = nn.Conv1d(
            in_channels=self.ppConv2.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppFc1 = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )
        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.conv6.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,Xsq,x,y)
        ## here, we will use the name x0 to denote the (X,Xsq) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## mds        print('neuronValues.size = ',neuronValues.size())
        ## --> [64,4,4000] for batch size 64
        x0 = neuronValues[:, 0:2, :]  ## picks out the 0 & 1 feature sets, X & Xsq
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4of6(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5(x0))
        x0 = self.conv5dropout(x0)
        x0 = leaky(self.conv6(x0))
        x0 = self.conv6dropout(x0)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x0)
        x0 = x0.view(x0.shape[0], x0.shape[-1])

        ##  now create an "architecture" for the perturbative element
        ##  similar to the original SimpleCNN3Layer  model with
        ##  3 convolutional layers followed by a fully connected layer
        ##  as this began to learn very quickly
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = self.softplus(x0 * x1)
        ##        neuronValues = self.softplus(x0)

        return neuronValues
