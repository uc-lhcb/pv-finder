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

class TracksToKDE_Ellipsoids(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=12, nOut2=15, nOut3=25):
        super(TracksToKDE_Ellipsoids,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
       

        self.layer1 = nn.Linear(
                    in_features = 9,
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

class TracksToKDE_Ellipsoids_SevenLayerCake(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50):
        super(TracksToKDE_Ellipsoids_SevenLayerCake,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
        self.nOut6 = nOut6
       

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
        self.layer4a = nn.Linear(
                    in_features = self.layer3.out_features,
                    out_features = self.nOut4,
                    bias = True)
        self.layer5a = nn.Linear(
                    in_features = self.layer4a.out_features,
                    out_features = self.nOut5,
                    bias = True)
        self.layer6a = nn.Linear(
                    in_features = self.layer5a.out_features,
                    out_features = self.nOut6,
                    bias = True)
        self.layer7a = nn.Linear(
                    in_features = self.layer6a.out_features,
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
        x = leaky(self.layer4a(x))
        x = leaky(self.layer5a(x))
        x = leaky(self.layer6a(x))
        x = (self.layer7a(x))  ## produces 4000 bin feature
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


class TracksToKDE_Ellipsoids_DirtyDozen(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25):
        super(TracksToKDE_Ellipsoids_DirtyDozen,self).__init__()

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
## mds         print("filt.shape = ",filt.shape)
## mds         print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1,2)
## mds        print("after transpose, x.shape = ", x.shape)
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
       
## mds         print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
## mds         print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
## mds        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


## mds        print("y_prime.shape = ",y_prime.shape)
       
## mds        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

class TracksToKDE_Ellipsoids_Skipper(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25):
        super(TracksToKDE_Ellipsoids_Skipper,self).__init__()

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
## the number of input features here is the number of
## output features from the previous layer + 9 for the
## original input features that should be concatenated
## in the forward method just before this layer is
## used.
        self.layer4 = nn.Linear(
                    in_features = self.layer3.out_features+9,
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
## again, we are using a skip connection from the original
## input features, so add 9 to layer7.out_features
        self.layer8 = nn.Linear(
                    in_features = self.layer7.out_features+9,
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
## again, we are using a skip connection from the original
## input features, so add 9 to layer11.out_features
        self.layer12 = nn.Linear(
                    in_features = self.layer11.out_features+9,
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
      
## make a copy of the initial features so they can be passed along using a skip connection 
        x0 = x 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
## mds        print("after layer 3, x.shape = ",x.shape)
## let's add a skip connection for the original features here
        x1 = torch.cat((x,x0),dim=2)
        x = x1
## mds        print("after concatenation, x1.shape = ",x1.shape)
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = leaky(self.layer6(x))
        x = leaky(self.layer7(x))
## let's add a skip connection for the original features here
        x1 = torch.cat((x,x0),dim=2)
        x = x1
        x = leaky(self.layer8(x))
        x = leaky(self.layer9(x))
        x = leaky(self.layer10(x))
        x = leaky(self.layer11(x))
## let's add a skip connection for the original features here
        x1 = torch.cat((x,x0),dim=2)
        x = x1
        x = (self.layer12(x))  ## produces 4000 bin feature
        x = self.softplus(x)
       
## mds         print("after softplus, x.shape = ",x.shape)
 
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
## mds         print("y_pred.shape = ",y_pred.shape)

        x1 = torch.mul(f2,x)
## mds        print("x1.shape = ",x1.shape)
        x1.view(nEvts,-1,4000)
        y_prime = torch.sum(x1,dim=1)


## mds        print("y_prime.shape = ",y_prime.shape)
       
## mds        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

class TracksToKDE_Ellipsoids_DDplus(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToKDE_Ellipsoids_DDplus,self).__init__()

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

## the "finalFilter" is meant to replace the fully connected layer with a
## convolutional layer that extends over the full range where we saw
## significant structure in the 4K x 4K matrix
        self.finalFilter=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."


        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        
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
##        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion
        y = y.view(y.shape[0], y.shape[-1])
####        print('at point Ba, y.shape = ',y.shape)
        y = self.fc1(y)   ####  a fully connected layer
##        y = self.finalFilter(y)  #### a convolutional layer
        y = y.view(nEvts,-1,4000)
## ## ##        print('at point C, y.shape = ',y.shape)
        y = self.softplus(y)


        y_prime = y.view(-1,4000)
## mds## ##         print("y_prime.shape = ",y_prime.shape)
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

################
class TracksToKDE_Ellipsoids_DDplusCNN(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToKDE_Ellipsoids_DDplusCNN,self).__init__()

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

## the "finalFilterCNN" is meant to replace the fully connected layer with a
## convolutional layer that extends over the full range where we saw
## significant structure in the 4K x 4K matrix
        self.finalFilterCNN=nn.Conv1d(
            in_channels = self.conv2.out_channels,
            out_channels = 1,
            kernel_size = 35,
            stride = 1,
            padding = (35 - 1) // 2
        )

        assert self.finalFilterCNN.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."


        self.conv1dropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        
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
##        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion
##        y = y.view(y.shape[0], y.shape[-1])
####        print('at point Ba, y.shape = ',y.shape)
        y = self.finalFilterCNN(y)  #### a convolutional layer
        y = y.view(nEvts,-1,4000)
## ## ##        print('at point C, y.shape = ',y.shape)
        y = self.softplus(y)


        y_prime = y.view(-1,4000)
## mds## ##         print("y_prime.shape = ",y_prime.shape)
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

####################  210116
## derived from TracksToKDE_ Ellipsoids_DDplus

class TracksToKDE_DDplusplus(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToKDE_DDplusplus,self).__init__()

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
            kernel_size = 51,
            stride = 1,
            padding = (51 - 1) // 2
        )

        self.conv1A=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 15, 
            kernel_size = 25,
            stride = 1,
            padding = (25 - 1) // 2
        )

        self.conv2=nn.Conv1d(
            in_channels = self.conv1A.out_channels,
            out_channels = 1,
            kernel_size = 11,
            stride = 1,
            padding = (11 - 1) // 2
        )
  
        self.fc1 = nn.Linear(
            in_features = 4000 * self.conv2.out_channels,
            out_features = 4000)


        self.conv1dropout = nn.Dropout(0.15)
        self.conv1Adropout = nn.Dropout(0.15)
        self.conv2dropout = nn.Dropout(0.15)
        
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
        y = leaky(self.conv1A(y))
        y = self.conv1Adropout(y)
        y = leaky(self.conv2(y))
        y = self.conv2dropout(y)
##        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion
        y = y.view(y.shape[0], y.shape[-1])
####        print('at point Ba, y.shape = ',y.shape)
        y = self.fc1(y)   ####  a fully connected layer
        y = y.view(nEvts,-1,4000)
## ## ##        print('at point C, y.shape = ',y.shape)
        y = self.softplus(y)


        y_prime = y.view(-1,4000)
## mds## ##         print("y_prime.shape = ",y_prime.shape)
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred

################




class TracksToKDE_Ellipsoids_DDplus_noDropOut(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25, nOut6=50,
                       nOut7=25, nOut8=25, nOut9=50,
                       nOut10=25, nOut11=25, latentChannels=8):
        super(TracksToKDE_Ellipsoids_DDplus_noDropOut,self).__init__()

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

## the "finalFilter" is meant to replace the fully connected layer with a
## convolutional layer that extends over the full range where we saw
## significant structure in the 4K x 4K matrix
        self.finalFilter=nn.Conv1d(
            in_channels = self.conv1.out_channels,
            out_channels = 1,
            kernel_size = 15,
            stride = 1,
            padding = (15 - 1) // 2
        )

        assert self.finalFilter.kernel_size[0] % 2 == 1, "Kernel size should be odd for 'same' conv."


        self.conv1dropout = nn.Dropout(0)
        self.conv2dropout = nn.Dropout(0)
        
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
##        print('at point B, y.shape = ',y.shape)
# Remove empty middle shape diminsion
        y = y.view(y.shape[0], y.shape[-1])
####        print('at point Ba, y.shape = ',y.shape)
        y = self.fc1(y)   ####  a fully connected layer
##        y = self.finalFilter(y)  #### a convolutional layer
        y = y.view(nEvts,-1,4000)
## ## ##        print('at point C, y.shape = ',y.shape)
        y = self.softplus(y)


        y_prime = y.view(-1,4000)
## mds## ##         print("y_prime.shape = ",y_prime.shape)
       
## mds##         print("y_pred[:,0:10] =  ",y_pred[:,0:10])
## mds        print("y_prime[:,0:10] =  ",y_prime[:,0:10])
        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred
