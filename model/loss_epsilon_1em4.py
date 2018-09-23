import torch

# how to define our cost function
class Loss(torch.nn.Module):   
    def forward(self, x, y):
        r = torch.abs(((x+1e-4)/(y+1e-4))) 
        alpha = -torch.log(2*r/(r**2+1))
        beta = (1./4000.)*alpha.sum()
            
        return beta
