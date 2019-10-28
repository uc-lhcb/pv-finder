import torch

# how to define our cost function
class Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-5, coefficient=5.0):
        """
        Epsilon is a parameter that can be adjusted.
        """

        # You must call the original constructor (torch.nn.Module.__init__(self))!
        super().__init__()

        # Now you can add things
        self.epsilon = epsilon
        self.coefficient = coefficient

    def forward(self, x, y):
        # Make a boolean mask of non-nan values of the target histogram.
        # This will be used to select items from y:
        # see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html#boolean-or-mask-index-arrays
        #
        # Note that if masking was not requested when loading the data, there
        # will be no NaNs and this will be all Trues and will do nothing special.
        valid = ~torch.isnan(y)

        # Compute r, only including non-nan values. r will probably be shorter than x and y.
        r = torch.abs((x[valid] + self.epsilon) / (y[valid] + self.epsilon))

        # Compute -log(2r/(r² + 1))
        alpha = -torch.log(2 * r / (r ** 2 + 1))
        alpha = alpha * (1.0 + self.coefficient * torch.exp(-r))

        # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
        # a .mean(), since alpha can be a bit shorter than x and y due to masking.
        beta = alpha.sum() / 4000

        return beta
