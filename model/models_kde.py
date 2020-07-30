##  200723  mds, based on a conversation with Henry Schreiner

##  This is a first model to use track parameters data to predict the KDE.
##  For the moment, the goal is to predict the central KDE values only,
##  not the corresponding Xmax and Ymax values.

import torch.nn as nn
import torch.nn.functional as F
import torch

class TracksToKDE_A(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self):
        super(TracksToKDE_A,self).__init__()
       

        self.layer1 = nn.Linear(
                    in_features = 6,
                    out_features = 12,
                    bias = True
                    )
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = 15,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = 4000,
                    bias = True)
        

        
    def forward(self, x):
        
##        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nFeatures = 6
        nEvts = x.shape[0]
##        print("nEvts = ", nEvts)
        x = x.view(nEvts,nFeatures,-1)
        print("after view, x.shape = ", x.shape)
        x = x.transpose(1,2)
        print("after transpose, x.shape = ", x.shape)
        ones = torch.ones([x.shape[0],x.shape[1],4000])
        print("ones.shape = ",ones.shape)
        mask = x[:,:,0] > -98
        maskT = torch.transpose(mask,0,1)
        maskTF = maskT.float()
        print("mask.shape = ",mask.shape)
        print("maskTF.shape = ",maskTF.shape)
        print("x.shape[0] = ",x.shape[0])
        print("x.shape[1] = ",x.shape[1])
        print("x.shape[2] = ",x.shape[2])
        
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))  ## produces 4000 bin feature
        
        x.view(nEvts,-1,4000)
        print("just before sum, x.shape = ",x.shape)
        y_pred = torch.sum(x,dim=1)
        print("y_pred.shape = ",y_pred.shape)

        xPrime = x.view(nEvts,-1,4000)
        xPP = torch.matmul(maskTF,xPrime)
        print("xPP.shape = ",xPP.shape)
        print("xPrime.shape = ",xPrime.shape)
        y_prime = torch.sum(xPrime,dim=1)
        print("y_prime.shape = ",y_prime.shape)
        
        return y_pred


class TracksToKDE_B(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=12, nOut2=15):
        super(TracksToKDE_B,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
       

        self.layer1 = nn.Linear(
                    in_features = 6,
                    out_features = self.nOut1,
                    bias = True
                    )
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = 4000,
                    bias = True)
        

        
    def forward(self, x):
        
        print("in forward, x.shape = ",x.shape)
        leaky = nn.LeakyReLU(0.01)
        
        nFeatures = 6
        nEvts = x.shape[0]
        print("nEvts = ", nEvts)
        x = x.view(nEvts,nFeatures,-1)
        print("after view, x.shape = ", x.shape)
        x = x.transpose(1,2)
        print("after transpose, x.shape = ", x.shape)
        
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = (self.layer3(x))  ## produces 4000 bin feature
        x = self.softplus(x)
        
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
        
        return y_pred


class TracksToKDE_C(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=12, nOut2=15):
        super(TracksToKDE_C,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
       

        self.layer1 = nn.Linear(
                    in_features = 6,
                    out_features = self.nOut1,
                    bias = True
                    )
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = 4000,
                    bias = True)
        

        
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
## mds        print("filt.shape = ",filt.shape)
## mds        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
## mds        print("after transpose, x.shape = ", x.shape)
        ones = torch.ones(nEvts,nFeatures,nTrks)
        
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = (self.layer3(x))  ## produces 4000 bin feature
        x = self.softplus(x)
       
## mds        print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
## mds        print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
## mds        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


## mds        print("y_prime.shape = ",y_prime.shape)
       
## mds        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

class TracksToKDE_D(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=12, nOut2=15, nOut3=25):
        super(TracksToKDE_D,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
       

        self.layer1 = nn.Linear(
                    in_features = 6,
                    out_features = self.nOut1,
                    bias = True
                    )
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
                    out_features = 4000,
                    bias = True)
        

        
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
## mds        print("filt.shape = ",filt.shape)
## mds        print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
## mds        print("after transpose, x.shape = ", x.shape)
        ones = torch.ones(nEvts,nFeatures,nTrks)
        
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = (self.layer4(x))  ## produces 4000 bin feature
        x = self.softplus(x)
       
## mds        print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
## mds        print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
## mds        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


## mds        print("y_prime.shape = ",y_prime.shape)
       
## mds        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

