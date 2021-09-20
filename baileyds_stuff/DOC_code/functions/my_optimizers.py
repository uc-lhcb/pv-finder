from torch.optim import Optimizer
import torch

"""
Some basic optimization algorithms
"""
class GDMethod(Optimizer):

    def __init__(self, params, lr=1e-3):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr)
        #self.grad_store = []
        super(GDMethod, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                if p.grad is not None:
                    #print("-------------------------------")
                    #print("Pi:", p)
                    #print("grad:", p.grad)
                    p.add_(p.grad, alpha=-lr)
                    #print("Pf:", p)
        #self.grad_store.append([p.grad.clone() for p in group['params']])

class MomentumMethod(Optimizer):

    def __init__(self, params, lr=1e-3, beta=0):
        defaults = dict(lr=lr, beta=beta, momentum_list=None)#[torch.zeros_like(param).detach() for param in params])
        self.grad_store = []
        self.momentum_store = []
        super(MomentumMethod, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            momentum_list = group['momentum_list']

            if momentum_list == None:
                momentum_list = []
                for i,param in enumerate(group['params']):
                    momentum_list.append(torch.zeros_like(param).detach())

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    momentum = momentum_list[i]
                    p.add_(momentum.mul(lr), alpha=-1)
                    momentum.mul_(beta).add_(p.grad, alpha=1-beta)

                    momentum_list[i] = momentum

            group['momentum_list'] = momentum_list

        self.grad_store.append([p.grad.clone() for p in group['params']])
        self.momentum_store.append([t.clone() for t in group['momentum_list']])
            #print(p, 'step:', -lr*p.grad)

class Adagrad(Optimizer):

    def __init__(self, params, lr=1e-3, eps=1e-8):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, eps=eps, sq_grad_sum=[])
        self.grad_store = []
        super(Adagrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            sq_grad_sum = group['sq_grad_sum']
            for i, p in enumerate(group['params']):
                if p.grad is not None:

                    if i+1 > len(sq_grad_sum):
                        sq_grad_sum.append(torch.zeros_like(p.grad))

                    #print(sq_grad_sum)

                    #adding the element square of the gradient
                    sq_grad_sum[i] = torch.addcmul(sq_grad_sum[i], p.grad, p.grad)
                    #adding grad / element sqrt of sq sum, times -lr
                    p.add_(p.grad.div(sq_grad_sum[i].add_(eps).sqrt()), alpha=-lr)

            group['sq_grad_sum'] = sq_grad_sum

        self.grad_store.append([p.grad.clone() for p in group['params']])

class RMSProp(Optimizer):

    def __init__(self, params, lr=1e-3, ema_coeff=0.9, eps=1e-8):
        #required dict mapping parameter names to def
        #values

        #EMA = exponential moving average coeff

        defaults = dict(lr=lr, ema_coeff=ema_coeff, eps=eps, ema_sum=[])
        self.grad_store = []
        self.ema_store  = []
        super(RMSProp, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            ema_coeff = group['ema_coeff']
            ema_sum = group['ema_sum']
            for i, p in enumerate(group['params']):
                if p.grad is not None:

                    #initialize ema sum as tensor of zeros
                    if i+1 > len(ema_sum):
                        ema_sum.append(torch.zeros_like(p.grad))

                    #print(sq_grad_sum)

                    #adding the next element of ema sum
                    ema_sum[i] = torch.addcmul(ema_sum[i].mul(ema_coeff), p.grad, p.grad, value=(1-ema_coeff))
                    #adding grad / element sqrt of ema sum, times -lr
                    p.add_(p.grad.div(ema_sum[i].add_(eps).sqrt()), alpha=-lr)

            group['ema_sum'] = ema_sum

        self.grad_store.append([p.grad.clone() for p in group['params']])
        self.ema_store.append([t.clone() for t in group['ema_sum']])

class EpochMomentum(Optimizer):

    def __init__(self, params, lr=1, beta=0.9):
        defaults = dict(lr=lr, beta=beta, momentum_list=None)#[torch.zeros_like(param).detach() for param in params])
        super(EpochMomentum, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            momentum_list = group['momentum_list']

            if momentum_list == None:
                momentum_list = []
                for i,param in enumerate(group['params']):
                    momentum_list.append(torch.zeros_like(param).detach())

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    momentum = momentum_list[i]
                    p.add_(momentum.mul(lr), alpha=1)
                    this_ep_step = p.clone().detach() - self.model_state[i]
                    #print('Epoch step:', this_ep_step)
                    momentum.mul_(beta).add_(this_ep_step, alpha=1-beta)

                    momentum_list[i] = momentum

            group['momentum_list'] = momentum_list
            #print('Momenta:', momentum_list)
            
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
        
"""
Some basic learning rate schedulers, which will set the lr for an optimizer
"""
import numpy as np
from math import floor

def LinearLrUpdate(index, lr_0, optimizer, decay_rate):
    for group in optimizer.param_groups:
        lr = lr_0/(1+decay_rate*index)
        group['lr'] = lr
    return lr

def StepLrUpdate(index, lr_0, optimizer, decay_rate, drop_rate):
    for group in optimizer.param_groups:
        lr = lr_0*pow(decay_rate, int((1+index)/drop_rate))
        group['lr'] = lr
    return lr

class StepLrSchedule():
    def __init__(self, optimizer, decay_rate=0.7, drop_rate=10):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.drop_rate = drop_rate
        for group in self.optimizer.param_groups:
            self.lr_i = group['lr']
        self.epoch = 0
        
    def step(self, num_epochs = 1):
        for group in self.optimizer.param_groups:
            #lr = group['lr']
            print('lr inital:', self.lr_i)
            lr = self.lr_i*pow( self.decay_rate, floor( (1+self.epoch+num_epochs)/self.drop_rate) )
            print('Epoch', self.epoch, '| exponent is', floor( (1+self.epoch+num_epochs)/self.drop_rate) )
            print('lr factor is', self.decay_rate, '^', floor( (1+self.epoch+num_epochs)/self.drop_rate) )
            group['lr'] = lr
            print('final lr:', lr)
        self.epoch += num_epochs
        return 
    
def ExpLrUpdate(index, lr_0, optimizer, decay_rate):
    for group in optimizer.param_groups:
        lr = lr_0*np.exp(-decay_rate*index)
        group['lr'] = lr
    return lr

def CostLrUpdate(cost, lr_0, optimizer):
    for group in optimizer.param_groups:
        lr = lr_0*cost
        group['lr'] = lr
    return lr

