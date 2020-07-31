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

        nFeatures = 4000
        nEvts = y.shape[0]
##        print("nEvts = ", nEvts)
##        y = y.view(nEvts,nFeatures,-1)
        y = y.view(nEvts,-1,nFeatures)
        y = y.transpose(1,2) 

##        print("after view, y.shape = ", y.shape)
        y_kde = y[:,:,0]
##        print("y_kde.shape = ",y_kde.shape)
##        y_kde = y[0,:,0].cpu().numpy()
##        plt.figure()
##        plt.plot(y_kde, color="r")
##        plt.show()

        return 100.*torch.sum((y_kde - x) ** 2)

        return beta
