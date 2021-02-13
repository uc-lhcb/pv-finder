import torch
from torch import nn
import numpy as np

## NOTE: this comment thread comes from Dr. Sokoloff (with necessary modifications)
## 190725
## This model is designed as a "perturbative" model.
## The feature set passed in (initial neuron values) is of the form
## (X,Xsq,x,y) where X is the original KDE, Xsq is the element-wise square of X
## and x and y are the values of x and y at each z where the KDE is maximum.
## These feature set will be divided into two parts (X,Xsq) and (x,y) and
## each of these will be run through some convolutional layers to produce
## 4000 bin tensors.
## Then, the concantenation of the feature sets will be passed through a final
## convolutional layer.
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
'''
Benchmark network architectures with the following attributes:
1. Three feature set using X, x, y.
2. 6 layer convolutional architecture for X and Xsq feature set.
3. 4 layer conovlutional architecture for x and y feature set.
4. Concatenates two feature sets and passes through a convolutional layer.
5. LeakyRELU activation used for convolutional layer.
6. Softplus activaation used for output.
7. Channel count follows the format:    20-10-10-10-1-1 (X), 20-10-10-1 (x, y), 20-1 (X, x, y)
8. Kernel size follows the format:      25-15-15-15-15-91 (X), 25-15-15-91 (x, y),  91  (X, x, y)
'''
class SimpleCNN5Layer_Ca_A(nn.Module):
    ## same as SimpleCNN5Layer_C, except that sigmoid activation is replaced
    ## with Softplus activation
    def __init__(self):
        super(SimpleCNN5Layer_Ca_A, self).__init__()

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

        x = torch.nn.Softplus()(x)

        return x

class All_CNN6Layer_A(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
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

        x = torch.nn.Softplus()(x)

        return x

class ThreeFeature_6Layer_XYPretrain_A(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ThreeFeature_6Layer_XYPretrain_A, self).__init__()

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
            out_channels=1, # NOTE: might need to be 10, but to make comptible w/ AllCNN it is 1
            kernel_size=5, # NOTE: should be 15, but compatibility issues w/ ALLCNN
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

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5(x0))
        x0 = self.conv5dropout(x0)
        x0 = self.finalFilter(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ThreeFeature_All_CNN6Layer_A(nn.Module):
    def __init__(self):
        super(ThreeFeature_All_CNN6Layer_A, self).__init__()
        ##
        ##  we will re-use the names of the convolutional layers from All_CNN6Layer_A
        ##  for the (X) feature set; then use similar (but different) names for
        ##  the layers that process the "pertubative" features (x,y)
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
            out_channels=1, # NOTE: might need to be 10, but to make comptible w/ AllCNN it is 1
            kernel_size=5, # NOTE: should be 15, but compatibility issues w/ ALLCNN
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

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # layer for concatenated two feature sets
        self.largeConv = nn.Conv1d(
            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
            out_channels=1,
            kernel_size = 91, # this is a totally random guess and has no logical backing
            stride=1, # might be worth trying 2
            padding=(91 - 1) // 2,
        )

        assert (
            self.largeConv.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5(x0))
        x0 = self.conv5dropout(x0)
        x0 = self.finalFilter(x0)
        
        

        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)

        # Run concatenated "perturbative" and "non-perturbative" features through a convolutional layer,
        # then softplus for output layer. This should, maybe, work better than just taking the product of 
        # the two feature sets as an output layer (which is what used to be done). 
        x0_and_x1 = self.largeConv(torch.cat([x0, x1], 1))
        x0_and_x1 = x0_and_x1.view(x0_and_x1.shape[0], x0_and_x1.shape[-1])
        neuronValues = torch.nn.Softplus()(x0_and_x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues
    
'''
Same as ThreeFeature_All_CNN6Layer_A but element wise multiplication used instead of
concatenation and a "largeConv" layer.
'''
class ThreeFeature_All_CNN6Layer_A1(nn.Module):
    def __init__(self):
        super(ThreeFeature_All_CNN6Layer_A1, self).__init__()
        ##
        ##  we will re-use the names of the convolutional layers from All_CNN6Layer_A
        ##  for the (X) feature set; then use similar (but different) names for
        ##  the layers that process the "pertubative" features (x,y)
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
            out_channels=1, # NOTE: might need to be 10, but to make comptible w/ AllCNN it is 1
            kernel_size=5, # NOTE: should be 15, but compatibility issues w/ ALLCNN
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

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
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

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X (excludes Xsq)
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x0 = leaky(self.conv1(x0))
        x0 = self.conv1dropout(x0)
        x0 = leaky(self.conv2(x0))
        x0 = self.conv2dropout(x0)
        x0 = leaky(self.conv3(x0))
        x0 = self.conv3dropout(x0)
        x0 = leaky(self.conv4(x0))
        x0 = self.conv4dropout(x0)
        x0 = leaky(self.conv5(x0))
        x0 = self.conv5dropout(x0)
        x0 = self.finalFilter(x0)
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        

        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues

'''
Testing method(s) of reducing parameters through procedure/methodology
Same as Model A with following changes:
1. Element-wise addition used instead of concatenation or multiplication before final layer
(NO MODEL AVAILABLE)
'''

'''
NOTE: The reason there is no SimpleCNN5Layer_Ca_C-E is that they have compatible starting architectures
with the "E model". Thus, we can use this "E model" for AllCNN_B-E models
'''
'''
Modified network architecture of Model_A with the following attributes:
NOTE: All attributes shared with AllCNN_A are omitted
1. Batch Normalization in each layer
2. One skip connection added
3. Channel count follows the format:    16-9-9-9-1-1 (X), 16-9-9-01 (x, y), 20-1 (X, x, y)
4. Kernel size follows the format:      25-15-15-15-15-91 (X), 25-15-15-91 (x, y),  91  (X, x, y)
5. SimpleCNN5Layer, TwoFeature_CNN6Layer_A, and All_CNN6Layer_A are intermediate models to obtain well-trained models to use for the perturbative model.
'''
class SimpleCNN5Layer_Ca_E(nn.Module):
    softplus = torch.nn.Softplus()
    def __init__(self):
        super(SimpleCNN5Layer_Ca_E, self).__init__()

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
    
class All_CNN6Layer_E(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
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
        
        self.bn1 =  nn.BatchNorm1d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm1d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm1d(self.conv3a.out_channels)
        self.bn4 = nn.BatchNorm1d(self.conv4.out_channels)
        self.bn5 = nn.BatchNorm1d(self.conv5.out_channels)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.bn1(self.conv1(x)))
        x01 = self.conv1dropout(x01)
        x02 = leaky(self.bn2(self.conv2(x01)))
        x02 = self.conv2dropout(x02)
        x03 = leaky(self.bn3(self.conv3a(torch.cat([x01, x02], 1))))
        x03 = self.conv3dropout(x03)
        x04 = leaky(self.bn4(self.conv4(x03)))
        x04 = self.conv4dropout(x04)
        x05 = leaky(self.bn5(self.conv5(x04)))
        x05 = self.conv5dropout(x05)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x0 = self.finalFilter(x05)
        x0 = x0.view(x0.shape[0], x0.shape[-1])

        x0 = torch.nn.Softplus()(x0)

        return x0

class ThreeFeature_6Layer_XYPretrain_E(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ThreeFeature_6Layer_XYPretrain_E, self).__init__()

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

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
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

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.bn1(self.conv1(x)))
        x01 = self.conv1dropout(x01)
        x02 = leaky(self.bn2(self.conv2(x01)))
        x02 = self.conv2dropout(x02)
        x03 = leaky(self.bn3(self.conv3a(torch.cat([x01, x02], 1))))
        x03 = self.conv3dropout(x03)
        x04 = leaky(self.bn4(self.conv4(x03)))
        x04 = self.conv4dropout(x04)
        x05 = leaky(self.bn5(self.conv5(x04)))
        x05 = self.conv5dropout(x05)
        x0 = self.finalFilter(x05)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ThreeFeature_All_CNN6Layer_E(nn.Module):
    def __init__(self):
        super(ThreeFeature_All_CNN6Layer_E, self).__init__()
        ##
        ##  we will re-use the names of the convolutional layers from All_CNN6Layer_A
        ##  for the (X) feature set; then use similar (but different) names for
        ##  the layers that process the "pertubative" features (x,y)
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

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."
        
        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # layer for concatenated two feature sets
        self.largeConv = nn.Conv1d(
            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
            out_channels=1,
            kernel_size = 91, # this is a totally random guess and has no logical backing
            stride=1, # might be worth trying 2
            padding=(91 - 1) // 2,
        )

        assert (
            self.largeConv.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
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

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.bn1(self.conv1(x)))
        x01 = self.conv1dropout(x01)
        x02 = leaky(self.bn2(self.conv2(x01)))
        x02 = self.conv2dropout(x02)
        x03 = leaky(self.bn3(self.conv3a(torch.cat([x01, x02], 1))))
        x03 = self.conv3dropout(x03)
        x04 = leaky(self.bn4(self.conv4(x03)))
        x04 = self.conv4dropout(x04)
        x05 = leaky(self.bn5(self.conv5(x04)))
        x05 = self.conv5dropout(x05)
        x0 = self.finalFilter(x05)
        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)

        # Run concatenated "perturbative" and "non-perturbative" features through a convolutional layer,
        # then softplus for output layer. This should, maybe, work better than just taking the product of 
        # the two feature sets as an output layer (which is what used to be done). 
        x0_and_x1 = self.largeConv(torch.cat([x0, x1], 1))
        x0_and_x1 = x0_and_x1.view(x0_and_x1.shape[0], x0_and_x1.shape[-1])
        neuronValues = torch.nn.Softplus()(x0_and_x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues


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
    
 
'''
Modified network architecture of Model_A with the following attributes:
NOTE: All attributes shared with AllCNN_A are omitted
1. Three feature set using X, x, y.
2. 8 layer convolutional architecture for X and Xsq feature set.
3. 4 layer conovlutional architecture for x and y feature set.
4. Takes element-wise product of the two feature sets for softplus.
7. Channel count follows the format:    01-20-10-10-10-10-10-01-01 (X), 02-10-01-01 (x, y)
8. Kernel size follows the format:      25-15-15-15-15-15-05-91 (X),    25-15-15-91 (x, y)
9. 3 skip connections, located at layers 3, 5, and 7
'''
## In order for this model to work, try an 8 layer, skip connection SimpleCNN 
## model as foundation
class SimpleCNN7Layer_Ca_W(nn.Module):
    def __init__(self):
        super(SimpleCNN7Layer_Ca_W, self).__init__()

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
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
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
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(
            in_features=4000 * self.conv7.out_channels, out_features=4000
        )

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.conv1(x))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.conv2(x01))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.conv3(torch.cat([x01, x2], 1)))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.conv4(x3))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.conv5(torch.cat([x3, x4], 1)))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.conv6(x5))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.conv7(torch.cat([x5, x6], 1)))
        x7 = self.conv7dropout(x7)
        
        # Remove empty middle shape diminsion
        x = x7.view(x7.shape[0], x7.shape[-1])
        
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                   
                   
class All_CNN8Layer_W(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(All_CNN8Layer_W, self).__init__()

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
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
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
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels,
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
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.15)

    def forward(self, x):
        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.conv1(x))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.conv2(x01))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.conv3(torch.cat([x01, x2], 1)))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.conv4(x3))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.conv5(torch.cat([x3, x4], 1)))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.conv6(x5))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.conv7(torch.cat([x5, x6], 1)))
        x7 = self.conv7dropout(x7)

        ##  with a little luck, the following two lines instantiate the
        ##  finalFilter and reshape its output to work as output to the
        ##  softplus activation
        x = self.finalFilter(x7)
        x = x.view(x.shape[0], x.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ThreeFeature_8Layer_XYPretrain_W(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ThreeFeature_8Layer_XYPretrain_W, self).__init__()

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
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
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
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFC = nn.Linear(
            in_features=4000 * self.ppConv3.out_channels, out_features=4000
        )

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:
        #x0 = neuronValues[:, 0:1, :]
        #x1 = neuronValues[:, 2:4, :]
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 1 & 2 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.conv1(x0))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.conv2(x01))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.conv3(torch.cat([x01, x2], 1)))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.conv4(x3))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.conv5(torch.cat([x3, x4], 1)))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.conv6(x5))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.conv7(torch.cat([x5, x6], 1)))
        x7 = self.conv7dropout(x7)
        
        x0 = self.finalFilter(x7)
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.ppFC(x1)

        # Take the product of the two feature sets as an output layer. This can
        # be modified to a convolutional layer later, but this must be done in the
        # "transition" model.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ThreeFeature_All_CNN8Layer_W(nn.Module):
    def __init__(self):
        super(ThreeFeature_All_CNN8Layer_W, self).__init__()
        
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
            in_channels=self.conv2.out_channels+self.conv1.out_channels,
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
            in_channels=self.conv4.out_channels+self.conv3.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv5.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        self.conv6 = nn.Conv1d(
            in_channels=self.conv5.out_channels,
            out_channels=10,
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.conv6.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."
        
        self.conv7 = nn.Conv1d(
            in_channels=self.conv6.out_channels+self.conv5.out_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )

        assert (
            self.conv7.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ## the "finalFilter" is meant to replace the fully connected layer with a
        ## convolutional layer that extends over the full range where we saw
        ## significant structure in the 4K x 4K matrix
        self.finalFilter = nn.Conv1d(
            in_channels=self.conv7.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.finalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # Perturbative layers
        self.ppConv1 = nn.Conv1d(
            in_channels=2,
            out_channels=20,
            kernel_size=25,
            stride=1,
            padding=(25 - 1) // 2,
        )

        assert (
            self.ppConv1.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        self.ppConv2 = nn.Conv1d(
            in_channels=self.ppConv1.out_channels,
            out_channels=10,
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
            kernel_size=15,
            stride=1,
            padding=(15 - 1) // 2,
        )

        assert (
            self.ppConv3.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' pConv."

        # notice how there are two less layers in "perturbative" compared to "non-perturbative"
        
        self.ppFinalFilter = nn.Conv1d(
            in_channels=self.ppConv3.out_channels,
            out_channels=1,
            kernel_size=91,
            stride=1,
            padding=(91 - 1) // 2,
        )

        assert (
            self.ppFinalFilter.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        # layer for concatenated two feature sets
        self.largeConv = nn.Conv1d(
            in_channels=self.finalFilter.out_channels+self.ppFinalFilter.out_channels,
            out_channels=1,
            kernel_size = 91, # this is a totally random guess and has no logical backing
            stride=1, # might be worth trying 2
            padding=(91 - 1) // 2,
        )

        assert (
            self.largeConv.kernel_size[0] % 2 == 1
        ), "Kernel size should be odd for 'same' conv."

        ##  18 July 2019 try dropout 0.15 rather than 0.05 (used in CNN5Layer_B) to mitigate overtraining
        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        self.conv3dropout = nn.Dropout(0.15)
        self.conv4dropout = nn.Dropout(0.15)
        self.conv5dropout = nn.Dropout(0.15)
        self.conv6dropout = nn.Dropout(0.15)
        self.conv7dropout = nn.Dropout(0.15)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        leaky = nn.LeakyReLU(0.01)
        x01 = leaky(self.conv1(x0))
        x01 = self.conv1dropout(x01)
        x2 = leaky(self.conv2(x01))
        x2 = self.conv2dropout(x2)
        x3 = leaky(self.conv3(torch.cat([x01, x2], 1)))
        x3 = self.conv3dropout(x3)
        x4 = leaky(self.conv4(x3))
        x4 = self.conv4dropout(x4)
        x5 = leaky(self.conv5(torch.cat([x3, x4], 1)))
        x5 = self.conv5dropout(x5)
        x6 = leaky(self.conv6(x5))
        x6 = self.conv6dropout(x6)
        x7 = leaky(self.conv7(torch.cat([x5, x6], 1)))
        x7 = self.conv7dropout(x7)
        
        x0 = self.finalFilter(x7)
        x0 = x0.view(x0.shape[0], x0.shape[-1])

        
        x1 = leaky(self.ppConv1(x1))
        x1 = self.conv1dropout(x1)
        x1 = leaky(self.ppConv2(x1))
        x1 = self.conv2dropout(x1)
        x1 = leaky(self.ppConv3(x1))
        x1 = self.conv3dropout(x1)
        x1 = self.ppFinalFilter(x1)
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues                 