import torch

# how to define our cost function
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.counter = 0
        
    def forward(self , x, y):
        r = torch.abs(((x+1e-5)/(y+1e-5))) 
        alpha = -torch.log(2*r/(r**2+1))
        beta = (1/4000)*alpha.sum()
            
        self.counter += 1
        return beta