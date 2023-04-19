##  This is a first model to use track parameters data to predict the KDE.
##  For the moment, the goal is to predict the central KDE values only,
##  not the corresponding Xmax and Ymax values.

import torch.nn as nn
import torch.nn.functional as F
import torch


class TracksToKDE_Ellipsoids_DirtyDozenSlicer(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25):
        super(TracksToKDE_Ellipsoids_DirtyDozenSlicer,self).__init__()

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
        self.layer12 = nn.Linear(
                    in_features = self.layer11.out_features,
                    out_features = 4000,
                    bias = True)
        

        
    def forward(self, x):
        
        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
        print("nEvts = ", nEvts,"   nFeatures = ", nFeatures, "  nTrks = ", nTrks)
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,4000)
        print("filt.shape = ",filt.shape)
        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
        print("after transpose, x.shape = ", x.shape)
        print("x[0,0:9,:] = ",x[0,0:9,:])
        print("x[0,0:99,0] = ",x[0,0:99,0])
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
        x = (self.layer12(x))  ## produces 4000 bin feature
        x = self.softplus(x)
       
        print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
        print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


        print("y_prime.shape = ",y_prime.shape)
       
        print("y_pred[0,0:40] =  ",y_pred[0,0:40])
        print("y_prime[0,0:10] =  ",y_prime[0,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred
############### ----------  end of "original"  model
#
## this model reads in a set of parameters that includes zBin and zOffset
## in addition to pocaz

class TracksToKDE_Ellipsoids_DirtyDozenSlicerA(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25):
        super(TracksToKDE_Ellipsoids_DirtyDozenSlicerA,self).__init__()

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
        self.layer12 = nn.Linear(
                    in_features = self.layer11.out_features,
                    out_features = 4000,
                    bias = True)
        

        
    def forward(self, x):
        
        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
        print("nEvts = ", nEvts,"   nFeatures = ", nFeatures, "  nTrks = ", nTrks)
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,4000)
        print("filt.shape = ",filt.shape)
        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
        print("after transpose, x.shape = ", x.shape)

#######
## 210829  added zBin and zOffset to x[];
##         want to remove them (temporarily) so the existing 
##         learning algorithm will continue to function "as is"
        x = x[:,:,2:11]
#######

        print("x[0,0:9,:] = ",x[0,0:9,:])
        print("x[0,0:99,0] = ",x[0,0:99,0])
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
        x = (self.layer12(x))  ## produces 4000 bin feature
        x = self.softplus(x)
       
        print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
        print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


        print("y_prime.shape = ",y_prime.shape)
       
        print("y_pred[0,0:40] =  ",y_pred[0,0:40])
        print("y_prime[0,0:10] =  ",y_prime[0,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred
