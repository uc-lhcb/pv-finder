from torch import nn


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


## 180808  mds  add a 3-layer model with drop-out following layers 1 and 3
class SimpleCNN3Layer_A(nn.Module):
    def __init__(self):
        super(SimpleCNN3Layer_A, self).__init__()
        # input channel size 1, output channel size 15
        self.nChan_out_layer1 = 20
        kernel_size_layer1 = 25
        stride_layer1 = 1
        assert kernel_size_layer1 % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding_layer1 = (kernel_size_layer1 - 1) // 2
        ## self.outputSize_layer1 = outputSize(4000,kernel_size_layer1,stride_layer1,padding_layer1)
        self.conv1 = nn.Conv1d(
            1, self.nChan_out_layer1, kernel_size_layer1, stride_layer1, padding_layer1
        )

        self.nChan_out_layer2 = 5
        kernel_size_layer2 = 15
        stride_layer2 = 1
        assert kernel_size_layer2 % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding_layer2 = (kernel_size_layer2 - 1) // 2
        ## self.outputSize_layer2 = outputSize(self.outputSize_layer1,kernel_size_layer2,stride_layer2,padding_layer2)
        self.conv2 = nn.Conv1d(
            self.nChan_out_layer1,
            self.nChan_out_layer2,
            kernel_size_layer2,
            stride_layer2,
            padding_layer2,
        )

        self.nChan_out_layer3 = 1
        kernel_size_layer3 = 5
        stride_layer3 = 1
        assert kernel_size_layer3 % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding_layer3 = (kernel_size_layer3 - 1) // 2
        ##  self.outputSize_layer3 = outputSize(self.outputSize_layer2,kernel_size_layer3,stride_layer3,padding_layer3)
        self.conv3 = nn.Conv1d(
            self.nChan_out_layer2,
            self.nChan_out_layer3,
            kernel_size_layer3,
            stride_layer3,
            padding_layer3,
        )

        self.conv1OutputDropout = nn.Dropout(0.35)
        self.conv3OutputDropout = nn.Dropout(0.35)

        ##        self.fc1 = torch.nn.Linear(self.outputSize_layer3*self.nChan_out_layer3, 4000)
        self.fc1 = nn.Linear(
            in_features=4000 * self.conv3.out_channels, out_features=4000
        )

        ##print('point AA')

    def forward(self, x):

        leaky = nn.LeakyReLU(0.01)
        x = leaky(self.conv1(x))
        x = self.conv1OutputDropout(x)
        x = leaky(self.conv2(x))
        x = leaky(self.conv3(x))
        ## x= x.view(-1, self.outputSize_layer3*self.nChan_out_layer3)
        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])
        x = self.conv3OutputDropout(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
