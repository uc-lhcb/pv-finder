import torch
import matplotlib.pyplot as plt

# how to define our cost function
class Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-5, coefficient=1.0):
        '''
        Epsilon is a parameter that can be adjusted.
        coefficient adjust asymmetry; 1.0 <==> symmetric
        '''

        # You must call the original constructor (torch.nn.Module.__init__(self))!
        super().__init__()
        
        # Now you can add things
        self.epsilon = epsilon
        self.coefficient = coefficient

    def forward(self, x, y):
        #  left over from pv-finder, but should not be a problem

        # Make a boolean mask of non-nan values of the target histogram.
        # This will be used to select items from y:
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html#boolean-or-mask-index-arrays
        #
        # Note that if masking was not requested when loading the data, there
        # will be no NaNs and this will be all Trues and will do nothing special.

##  4000 --> 100 for 100-bin intervals; replace all "4000" with "nFeatures" below
##        nFeatures = 4000
        nFeatures = 100
        nEvts = y.shape[0]
##        print("nEvts = ", nEvts)
        y = y.view(nEvts,nFeatures,-1)
        y = y.view(nEvts,-1,nFeatures)
        y = y.transpose(1,2) 

##        print("after view, y.shape = ", y.shape)
        y_kde = y[:,:,0]
##        print("y_kde.shape = ",y_kde.shape)
##        y_kde_np = y[0,:,0].cpu().numpy()
##        plt.figure()
##        plt.plot(y_kde_np, color="r")
##        plt.show()

        # Compute r, only including non-nan values. r will probably be shorter than x and y.
        valid = ~torch.isnan(y_kde)
        r = torch.abs((x[valid] + self.epsilon) / (y_kde[valid] + self.epsilon))

        # Compute -log(2r/(r² + 1))
        alpha = -torch.log(2*r / (r**2 + 1))
        alpha = alpha * (1.0 + self.coefficient * torch.exp(-r))

        # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
        # a .mean(), since alpha can be a bit shorter than x and y due to masking.
        beta = alpha.sum() / nFeatures

## let's add a chi-square type of term to give the values farther from zero
## more weight in the cost function

        sigma = 0.01
        diff = torch.sub(x[valid],y_kde[valid])
        diff = diff/sigma
        chisq = torch.pow(diff,2)
        ave_chisq = chisq.sum()/nFeatures
##
## 201025 starting with a fairly well trained model that has only
## beta and chisq terms in cost, normalize chi4 so it has (nominal)
## weights similar to chisq
        chi4 = 0.00001*torch.pow(diff,4)

        ave_chi4 = chi4.sum()/nFeatures

##  nEvts is the number of events in the batch; to get the average per events
##  rather than per batch, divide by this number
        ave_beta  = beta/nEvts
        ave_chisq = ave_chisq/nEvts
        ave_chi4  = ave_chi4/nEvts


        return ave_chisq
