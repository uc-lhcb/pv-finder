##  according to Will, 25 January 2021
from .efficiency import efficiency, ValueSet
from model.alt_loss_A import Loss
import time
from model.training import PARAM_EFF, Results
from collections import OrderedDict
import pytorch_lightning as pl

class BaseTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(self.args)
        self.loss = Loss(epsilon=1e-5, coefficient=self.args.asymmetry_parameter)
        self.time_start = time.time()
        
        self.train_step, self.global_step, self.train_loss, self.val_loss, self.val_step = 0, 0, 0, 0, 0
        self.eff = ValueSet(0, 0, 0, 0)
        self.batch_stats = {}
        
    def training_step(self, batch, batch_idx):
        self.train_step += 1
        inputs, labels = batch
        
        outputs = self.forward(inputs)
        loss_output = self.loss(outputs, labels)
        
        self.train_loss += loss_output.data.item()
        
        return loss_output 
    
    def validation_step(self, batch, batch_idx):
        self.val_step += 1
        inputs, labels = batch
        
        val_outputs = self.forward(inputs)
        loss_output = self.loss(val_outputs, labels)
        
        self.val_loss += loss_output.data.item()
        
        for label, output in zip(labels.cpu().numpy(), val_outputs.cpu().numpy()):
            self.eff += efficiency(label, output, **PARAM_EFF)
        
        return loss_output 

    def on_epoch_start(self):
        self.train_step, self.val_step, self.train_loss, self.val_loss = 0, 0, 0, 0
        self.eff = ValueSet(0, 0, 0, 0)
        self.time_start = time.time()
        
    def on_validation_epoch_start(self):
        self.epoch_time = self.time_start - time.time()

    def on_validation_epoch_end(self):
        self.global_step += self.train_step
        if self.global_step > 0:
            results = Results(1, self.train_loss/self.global_step, self.val_loss/self.val_step, self.epoch_time, self.eff)
            result = results._asdict()
            self.batch_stats = {
                'Efficiency':result['eff_val'].eff_rate,
                'False Positive Rate':result['eff_val'].fp_rate,
                'Validation Loss':result['val']*2,
                'Training Loss':result['cost']*2,
            }

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

 
class TracksToKDE_Ellipsoids_DDplus(BaseTrainer):
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
