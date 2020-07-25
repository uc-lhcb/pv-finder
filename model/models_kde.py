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
                    out_features = 8,
                    bias = True
                    )
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = 12,
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
##        print("after view, x.shape = ", x.shape)
        x = x.transpose(1,2)
##        print("after transpose, x.shape = ", x.shape)
        
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))  ## produces 4000 bin feature
        
        x.view(nEvts,-1,4000)
        y_pred = torch.sum(x,dim=1)
        
        return y_pred
