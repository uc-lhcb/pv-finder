import torch

# how to define our cost function
class Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
    
    def forward(self, x, y):
        valid = ~torch.isnan(y)
        r = torch.abs((x[valid] + self.value) / (y[valid] + self.epsilon))
        alpha = -torch.log(2*r / (r**2 + 1))
        beta = (1./4000.)*alpha.sum()

        return beta