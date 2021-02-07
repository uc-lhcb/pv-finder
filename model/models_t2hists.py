import torch.nn as nn
import torch.nn.functional as F
import torch

################
class TracksToHists_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToHists_A,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
        self.nOut6 = nOut6
        self.nOut7 = nOut7
        self.nOut8 = nOut8
        self.nOut9 = nOut9
        self.nOut10 = nOut10
        self.nOut11 = nOut11
      
        self.latentChannels = latentChannels 

        self.layer1 = nn.Linear(
                    in_features = 9,
                    out_features = self.nOut1,
                    bias = True)
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = self.nOut3,
                    bias = True)
        self.layer4 = nn.Linear(
                    in_features = self.layer3.out_features,
                    out_features = self.nOut4,
                    bias = True)
        self.layer5 = nn.Linear(
                    in_features = self.layer4.out_features,
                    out_features = self.nOut5,
                    bias = True)
        self.layer6 = nn.Linear(
                    in_features = self.layer5.out_features,
                    out_features = self.nOut6,
                    bias = True)
        self.layer7 = nn.Linear(
                    in_features = self.layer6.out_features,
                    out_features = self.nOut7,
                    bias = True)
        self.layer8 = nn.Linear(
                    in_features = self.layer7.out_features,
                    out_features = self.nOut8,
                    bias = True)
        self.layer9 = nn.Linear(
                    in_features = self.layer8.out_features,
                    out_features = self.nOut9,
                    bias = True)
        self.layer10 = nn.Linear(
                    in_features = self.layer9.out_features,
                    out_features = self.nOut10,
                    bias = True)
        self.layer11 = nn.Linear(
                    in_features = self.layer10.out_features,
                    out_features = self.nOut11,
                    bias = True)
        self.layer12new = nn.Linear(
                    in_features = self.layer11.out_features,
                    out_features = self.latentChannels*4000,
                    bias = True)
        
        self.conv1=nn.Conv1d(
            in_channels = self.latentChannels,
            out_channels = 25, 
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        self.conv2=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )
  
        self.fc1 = nn.Linear(
            in_features = 4000 * self.conv2.out_channels,
            out_features = 4000)


        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)

## end of code copied from TracksToKDE_Ellipsoids_DDplusCNN
## begin code derived from SimpleCNN5Layer_Ca.py
## change the layer names to be distinct from the 
## TracksToKDE layer names.

        self.hist_conv1=nn.Conv1d(
            in_channels = 1,
            out_channels = 20,
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        assert self.hist_conv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv2=nn.Conv1d(
            in_channels = self.hist_conv1.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv3=nn.Conv1d(
            in_channels = self.hist_conv2.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv4=nn.Conv1d(
            in_channels = self.hist_conv3.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv4.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv5=nn.Conv1d(
            in_channels = self.hist_conv4.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )

        assert self.hist_conv5.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv1dropout = nn.Dropout(0.15)
        self.hist_conv2dropout = nn.Dropout(0.15)
        self.hist_conv3dropout = nn.Dropout(0.15)
        self.hist_conv4dropout = nn.Dropout(0.15)
        self.hist_conv5dropout = nn.Dropout(0.35)

        self.hist_fc1 = nn.Linear(
            in_features = 4000 * self.hist_conv5.out_channels,
            out_features = 4000)

        
    def forward(self, x):
        
## mds        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
## mds        print("nEvts = ", nEvts,"   nFeatures = ", nFeatures, "  nTrks = ", nTrks)
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,4000)
##         print("filt.shape = ",filt.shape)
##        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
##         print("after transpose, x.shape = ", x.shape)
        ones = torch.ones(nEvts,nFeatures,nTrks)
      
## make a copy of the initial features so they can be passed along using a skip connection 
        x0 = x 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = leaky(self.layer6(x))
        x = leaky(self.layer7(x))
        x = leaky(self.layer8(x))
        x = leaky(self.layer9(x))
        x = leaky(self.layer10(x))
        x = leaky(self.layer11(x))
        x = leaky((self.layer12new(x)))  ## produces self.latentChannels*4000 bin feature

## at this point x should have the contents expected in the following line
        x = x.view(nEvts,nTrks,self.latentChannels,4000)
##        print(' at point AA, x.shape = ',x.shape)

## here we are summing over all the tracks, creating "y"
## which has a sum of all tracks' contributions in each of
## latentChannels for each event and each bin of the (eventual)
## KDE histogram
        f2 = torch.unsqueeze(f2,2)
        x = torch.mul(f2,x)
        y0 = torch.sum(x,dim=1) 
##         print(' at point AB, y0.shape = ',y0.shape)

## begin to process the latentChannels contributions to
## the final KDE using two convolutional layers
        y = leaky(self.conv1(y0))
        y = self.conv1dropout(y)
        y = leaky(self.conv2(y))
        y = self.conv2dropout(y)

## mds dec28        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion
        y = y.view(y.shape[0], y.shape[-1])
## mds dec28        print('at point Ba, y.shape = ',y.shape)
        y = self.fc1(y)   ####  a fully connected layer
        y = y.view(nEvts,-1,4000)
## mds dec28        print('at point C, y.shape = ',y.shape)
##  -->  at point C, y.shape =  torch.Size([16, 1, 4000])
        y = self.softplus(y)

        y_prime = y.view(-1,4000)
## mds dec28        print("y_prime.shape = ",y_prime.shape)
##  -->  y_prime.shape =  torch.Size([16, 4000])
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds dec28        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)

##  


## end of code copied from TracksToKDE_Ellipsoids_DDplus
## at this point, "y_pred" is a "predicted" poca ellipsoid probability KDE

## it seems that the shape wanted by the convolutional layers is
##  torch.Size([nEvts,1,4000]), as in y, not y_prime
        x = torch.mul(y,0.001)

##  code from SimpleCNN5Layer_Ca.py with "hist" prepended to 
##  layer names so they are distinct

        x = leaky(self.hist_conv1(x))
        x = self.hist_conv1dropout(x)
        x = leaky(self.hist_conv2(x))
        x = self.hist_conv2dropout(x)
        x = leaky(self.hist_conv3(x))
        x = self.hist_conv3dropout(x)
        x = leaky(self.hist_conv4(x))
        x = self.hist_conv4dropout(x)
        x = leaky(self.hist_conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.hist_conv5dropout(x)
        x = self.hist_fc1(x)

        x = self.softplus(x)

        return x

#--------------  from SimpleCNN5Layer_Ca.py


################
## TracksToHists_B is derived from TracksToHists_A
## Where _A used a fully connected layer at the very end,
## TracksToHists_B has a convolutional layer
class TracksToHists_B(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToHists_B,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
        self.nOut6 = nOut6
        self.nOut7 = nOut7
        self.nOut8 = nOut8
        self.nOut9 = nOut9
        self.nOut10 = nOut10
        self.nOut11 = nOut11
      
        self.latentChannels = latentChannels 

        self.layer1 = nn.Linear(
                    in_features = 9,
                    out_features = self.nOut1,
                    bias = True)
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = self.nOut3,
                    bias = True)
        self.layer4 = nn.Linear(
                    in_features = self.layer3.out_features,
                    out_features = self.nOut4,
                    bias = True)
        self.layer5 = nn.Linear(
                    in_features = self.layer4.out_features,
                    out_features = self.nOut5,
                    bias = True)
        self.layer6 = nn.Linear(
                    in_features = self.layer5.out_features,
                    out_features = self.nOut6,
                    bias = True)
        self.layer7 = nn.Linear(
                    in_features = self.layer6.out_features,
                    out_features = self.nOut7,
                    bias = True)
        self.layer8 = nn.Linear(
                    in_features = self.layer7.out_features,
                    out_features = self.nOut8,
                    bias = True)
        self.layer9 = nn.Linear(
                    in_features = self.layer8.out_features,
                    out_features = self.nOut9,
                    bias = True)
        self.layer10 = nn.Linear(
                    in_features = self.layer9.out_features,
                    out_features = self.nOut10,
                    bias = True)
        self.layer11 = nn.Linear(
                    in_features = self.layer10.out_features,
                    out_features = self.nOut11,
                    bias = True)
        self.layer12new = nn.Linear(
                    in_features = self.layer11.out_features,
                    out_features = self.latentChannels*4000,
                    bias = True)
        
        self.conv1=nn.Conv1d(
            in_channels = self.latentChannels,
            out_channels = 25, 
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        self.conv2=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )
  
        self.fc1 = nn.Linear(
            in_features = 4000 * self.conv2.out_channels,
            out_features = 4000)


        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)

## end of code copied from TracksToKDE_Ellipsoids_DDplusCNN
## begin code derived from SimpleCNN5Layer_Ca.py
## change the layer names to be distinct from the 
## TracksToKDE layer names.

        self.hist_conv1=nn.Conv1d(
            in_channels = 1,
            out_channels = 20,
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        assert self.hist_conv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv2=nn.Conv1d(
            in_channels = self.hist_conv1.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv3=nn.Conv1d(
            in_channels = self.hist_conv2.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv4=nn.Conv1d(
            in_channels = self.hist_conv3.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv4.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv5=nn.Conv1d(
            in_channels = self.hist_conv4.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )

        assert self.hist_conv5.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv1dropout = nn.Dropout(0.15)
        self.hist_conv2dropout = nn.Dropout(0.15)
        self.hist_conv3dropout = nn.Dropout(0.15)
        self.hist_conv4dropout = nn.Dropout(0.15)
        self.hist_conv5dropout = nn.Dropout(0.35)

        self.hist_fc1 = nn.Linear(
            in_features = 4000 * self.hist_conv5.out_channels,
            out_features = 4000)

## the "finalFilter" is meant to replace the fully connected layer with a
## convolutional layer that extends over the full range where we saw
## significant structure in the 4K x 4K matrix
        self.finalFilter=nn.Conv1d(
            in_channels = self.hist_conv5.out_channels,
            out_channels = 1,
            kernel_size = 91,
            stride = 1,
            padding = (91 - 1) // 2
        )

        assert self.finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        
    def forward(self, x):
        
## mds        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
## mds        print("nEvts = ", nEvts,"   nFeatures = ", nFeatures, "  nTrks = ", nTrks)
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,4000)
##         print("filt.shape = ",filt.shape)
##        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
##         print("after transpose, x.shape = ", x.shape)
        ones = torch.ones(nEvts,nFeatures,nTrks)
      
## make a copy of the initial features so they can be passed along using a skip connection 
        x0 = x 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = leaky(self.layer6(x))
        x = leaky(self.layer7(x))
        x = leaky(self.layer8(x))
        x = leaky(self.layer9(x))
        x = leaky(self.layer10(x))
        x = leaky(self.layer11(x))
        x = leaky((self.layer12new(x)))  ## produces self.latentChannels*4000 bin feature

## at this point x should have the contents expected in the following line
        x = x.view(nEvts,nTrks,self.latentChannels,4000)
##        print(' at point AA, x.shape = ',x.shape)

## here we are summing over all the tracks, creating "y"
## which has a sum of all tracks' contributions in each of
## latentChannels for each event and each bin of the (eventual)
## KDE histogram
        f2 = torch.unsqueeze(f2,2)
        x = torch.mul(f2,x)
        y0 = torch.sum(x,dim=1) 
##         print(' at point AB, y0.shape = ',y0.shape)

## begin to process the latentChannels contributions to
## the final KDE using two convolutional layers
        y = leaky(self.conv1(y0))
        y = self.conv1dropout(y)
        y = leaky(self.conv2(y))
        y = self.conv2dropout(y)
##        print('y.shape = ',y.shape)

## mds dec28        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion when used with  self.fc1
#  BUT not when using self.finalFilter
##        y = y.view(y.shape[0], y.shape[-1])
## mds dec28        print('at point Ba, y.shape = ',y.shape)
##        y = self.fc1(y)   ####  a fully connected layer
##  replace the final, fully connected layer with a 
##  convolutional layer
        y = self.finalFilter(y)
        y = y.view(nEvts,-1,4000)
## mds dec28        print('at point C, y.shape = ',y.shape)
##  -->  at point C, y.shape =  torch.Size([16, 1, 4000])
        y = self.softplus(y)

        y_prime = y.view(-1,4000)
## mds dec28        print("y_prime.shape = ",y_prime.shape)
##  -->  y_prime.shape =  torch.Size([16, 4000])
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds dec28        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)

##  


## end of code copied from TracksToKDE_Ellipsoids_DDplus
## at this point, "y_pred" is a "predicted" poca ellipsoid probability KDE

## it seems that the shape wanted by the convolutional layers is
##  torch.Size([nEvts,1,4000]), as in y, not y_prime
        x = torch.mul(y,0.001)

##  code from SimpleCNN5Layer_Ca.py with "hist" prepended to 
##  layer names so they are distinct

        x = leaky(self.hist_conv1(x))
        x = self.hist_conv1dropout(x)
        x = leaky(self.hist_conv2(x))
        x = self.hist_conv2dropout(x)
        x = leaky(self.hist_conv3(x))
        x = self.hist_conv3dropout(x)
        x = leaky(self.hist_conv4(x))
        x = self.hist_conv4dropout(x)
        x = leaky(self.hist_conv5(x))

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.hist_conv5dropout(x)
        x = self.hist_fc1(x)

        x = self.softplus(x)

        return x
#########################  end of TracksToHists_A


################
## TracksToHists_C is derived from TracksToHists_B
## Where _B used a fully connected layer at the very end,
## TracksToHists_C has a convolutional layer
class TracksToHists_C(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToHists_C,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
        self.nOut6 = nOut6
        self.nOut7 = nOut7
        self.nOut8 = nOut8
        self.nOut9 = nOut9
        self.nOut10 = nOut10
        self.nOut11 = nOut11
      
        self.latentChannels = latentChannels 

        self.layer1 = nn.Linear(
                    in_features = 9,
                    out_features = self.nOut1,
                    bias = True)
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = self.nOut3,
                    bias = True)
        self.layer4 = nn.Linear(
                    in_features = self.layer3.out_features,
                    out_features = self.nOut4,
                    bias = True)
        self.layer5 = nn.Linear(
                    in_features = self.layer4.out_features,
                    out_features = self.nOut5,
                    bias = True)
        self.layer6 = nn.Linear(
                    in_features = self.layer5.out_features,
                    out_features = self.nOut6,
                    bias = True)
        self.layer7 = nn.Linear(
                    in_features = self.layer6.out_features,
                    out_features = self.nOut7,
                    bias = True)
        self.layer8 = nn.Linear(
                    in_features = self.layer7.out_features,
                    out_features = self.nOut8,
                    bias = True)
        self.layer9 = nn.Linear(
                    in_features = self.layer8.out_features,
                    out_features = self.nOut9,
                    bias = True)
        self.layer10 = nn.Linear(
                    in_features = self.layer9.out_features,
                    out_features = self.nOut10,
                    bias = True)
        self.layer11 = nn.Linear(
                    in_features = self.layer10.out_features,
                    out_features = self.nOut11,
                    bias = True)
        self.layer12new = nn.Linear(
                    in_features = self.layer11.out_features,
                    out_features = self.latentChannels*4000,
                    bias = True)
        
        self.conv1=nn.Conv1d(
            in_channels = self.latentChannels,
            out_channels = 25, 
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        self.conv2=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )
  
        self.fc1 = nn.Linear(
            in_features = 4000 * self.conv2.out_channels,
            out_features = 4000)


        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)

## end of code copied from TracksToKDE_Ellipsoids_DDplusCNN
## begin code derived from SimpleCNN5Layer_Ca.py
## change the layer names to be distinct from the 
## TracksToKDE layer names.

        self.hist_conv1=nn.Conv1d(
            in_channels = 1,
            out_channels = 20,
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        assert self.hist_conv1.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv2=nn.Conv1d(
            in_channels = self.hist_conv1.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv2.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv3=nn.Conv1d(
            in_channels = self.hist_conv2.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv3.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."

        self.hist_conv4=nn.Conv1d(
            in_channels = self.hist_conv3.out_channels,
            out_channels = 10,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.hist_conv4.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv5=nn.Conv1d(
            in_channels = self.hist_conv4.out_channels,
            out_channels = 1,
            kernel_size = 5,
            stride = 1,
            padding = (5 - 1) // 2
        )

        assert self.hist_conv5.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' hist_conv."


        self.hist_conv1dropout = nn.Dropout(0.15)
        self.hist_conv2dropout = nn.Dropout(0.15)
        self.hist_conv3dropout = nn.Dropout(0.15)
        self.hist_conv4dropout = nn.Dropout(0.15)
        self.hist_conv5dropout = nn.Dropout(0.35)

## mds         self.hist_fc1 = nn.Linear(
## mds             in_features = 4000 * self.hist_conv5.out_channels,
## mds             out_features = 4000)

## the "finalFilter" is meant to replace the fully connected layer with a
## convolutional layer that extends over the full range where we saw
## significant structure in the 4K x 4K matrix
        self.hist_finalFilter=nn.Conv1d(
            in_channels = self.hist_conv5.out_channels,
            out_channels = 1,
            kernel_size = 91,
            stride = 1,
            padding = (91 - 1) // 2
        )

        assert self.hist_finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."
        
    def forward(self, x):
        
## mds        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
## mds        print("nEvts = ", nEvts,"   nFeatures = ", nFeatures, "  nTrks = ", nTrks)
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,4000)
##         print("filt.shape = ",filt.shape)
##        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
##         print("after transpose, x.shape = ", x.shape)
        ones = torch.ones(nEvts,nFeatures,nTrks)
      
## make a copy of the initial features so they can be passed along using a skip connection 
        x0 = x 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = leaky(self.layer6(x))
        x = leaky(self.layer7(x))
        x = leaky(self.layer8(x))
        x = leaky(self.layer9(x))
        x = leaky(self.layer10(x))
        x = leaky(self.layer11(x))
        x = leaky((self.layer12new(x)))  ## produces self.latentChannels*4000 bin feature

## at this point x should have the contents expected in the following line
        x = x.view(nEvts,nTrks,self.latentChannels,4000)
##        print(' at point AA, x.shape = ',x.shape)

## here we are summing over all the tracks, creating "y"
## which has a sum of all tracks' contributions in each of
## latentChannels for each event and each bin of the (eventual)
## KDE histogram
        f2 = torch.unsqueeze(f2,2)
        x = torch.mul(f2,x)
        y0 = torch.sum(x,dim=1) 
##         print(' at point AB, y0.shape = ',y0.shape)

## begin to process the latentChannels contributions to
## the final KDE using two convolutional layers
        y = leaky(self.conv1(y0))
        y = self.conv1dropout(y)
        y = leaky(self.conv2(y))
        y = self.conv2dropout(y)
##        print('y.shape = ',y.shape)

# Remove empty middle shape diminsion
        y = y.view(y.shape[0], y.shape[-1])
        y = self.fc1(y)   ####  a fully connected layer
        y = y.view(nEvts,-1,4000)

## mds dec28        print('at point C, y.shape = ',y.shape)
##  -->  at point C, y.shape =  torch.Size([16, 1, 4000])
        y = self.softplus(y)

        y_prime = y.view(-1,4000)
## mds dec28        print("y_prime.shape = ",y_prime.shape)
##  -->  y_prime.shape =  torch.Size([16, 4000])
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds dec28        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)

##  


## end of code copied from TracksToKDE_Ellipsoids_DDplus
## at this point, "y_pred" is a "predicted" poca ellipsoid probability KDE

## it seems that the shape wanted by the convolutional layers is
##  torch.Size([nEvts,1,4000]), as in y, not y_prime
        x = torch.mul(y,0.001)

##  code from SimpleCNN5Layer_Ca.py with "hist" prepended to 
##  layer names so they are distinct

        x = leaky(self.hist_conv1(x))
        x = self.hist_conv1dropout(x)
        x = leaky(self.hist_conv2(x))
        x = self.hist_conv2dropout(x)
        x = leaky(self.hist_conv3(x))
        x = self.hist_conv3dropout(x)
        x = leaky(self.hist_conv4(x))
        x = self.hist_conv4dropout(x)
        x = leaky(self.hist_conv5(x))
        x = self.hist_conv5dropout(x)

        x = self.hist_finalFilter(x)

        # Remove empty middle shape diminsion
        x = x.view(x.shape[0], x.shape[-1])

        x = self.softplus(x)

########## remove the fully connected final layer code
##        # Remove empty middle shape diminsion
##        x = x.view(x.shape[0], x.shape[-1])
##
##        x = self.hist_conv5dropout(x)
##        x = self.hist_fc1(x)
##
##        x = self.softplus(x)
###########  end of removed code

        return x
#########################  end of TracksToHists_C

