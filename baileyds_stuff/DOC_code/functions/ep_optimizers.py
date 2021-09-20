from torch.optim import Optimizer
import torch
import numpy as np
import time
import random

def test(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    return test_loss

def test_mb(data, model, loss_fn, device):
    model.eval()
    test_loss = 0
    X = data[0].to(device)
    y = data[1].to(device)
    with torch.no_grad():
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
    return test_loss

def normed_dot(list_a, list_b):
    dot = sum([torch.sum(torch.mul(list_a[i], list_b[i])) for i in range(len(list_a))])
    #print('num:', dot)
    #print('denom:', (step_norm(list_a)*step_norm(list_b)))
    return dot/(step_norm(list_a)*step_norm(list_b))

class EpochStep(Optimizer):

    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr, momentum_list=None)#[torch.zeros_like(param).detach() for param in params])
        super(EpochStep, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    @torch.no_grad()
    def step(self, dot, closure=None):
        #print('init model state:', self.model_state)
        for group in self.param_groups:
            new_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
        
        #print(len(new_state), len(self.model_state))
        
        this_epoch_step = [new_state[i]-self.model_state[i] for i in range(len(new_state))]

        for group in self.param_groups:
            lr = group['lr']
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    this_step = this_epoch_step[i]
                    p.add_(this_step, alpha=lr*dot)
               
        self.model_state = new_state
        #doing the below artificially makes the consecutive steps more collinear
#         for group in self.param_groups:
#             self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]


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

#implementing the EVE concept at the end of each epoch, in the direction of the overall epoch step
class EpochEVE(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, steps=[0, 0.5, 1, 2, 4, 8]):
        defaults = dict(steps=steps)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(EpochEVE, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)
        best_steps = []
        for group in self.param_groups:
            steps = group['steps']

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    this_step = p.clone().detach() - self.model_state[i]
                    losses = []
                    
                    #explores options in step direction
                    print('steps:', steps)
                    for s in steps:
                        time0 = time.time()
                        p.add_(this_step, alpha=s)
                        losses.append(test(self.data, self.model, self.loss_fn, self.device))
                        p.add_(this_step, alpha=-s)
                        print('param. change and forward prop. time:', time.time()-time0)
                    
                    #chooses best step size
                    best_step = steps[np.argmin(np.asarray(losses))]
                    best_steps.append(best_step)
                    #takes step
                    p.add_(this_step, alpha=best_step)
        
            #print('best steps:', best_steps)
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
        return best_steps
            
#implementing the EVE concept at the end of each epoch, in the direction of the overall epoch step
class EpochEVE_allparams(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, steps=[0, 0.5, 1, 2, 4, 8], num_mb=-1):
        defaults = dict(steps=steps, num_mb=num_mb)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        if num_mb != -1:
            self.data = torch.utils.data.Subset(data, [i for i in range(num_mb)])
        if num_mb == -1:
            self.data = data
        super(EpochEVE_allparams, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)
        best_steps = []
        for group in self.param_groups:
            steps = group['steps']
            #num_mb = group['num_mb']
            
            epoch_step = []
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    epoch_step.append(p.clone().detach() - self.model_state[i])
                    
            losses = []

            #explores options in step direction

            for s in steps:
                #time0 = time.time()
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(epoch_step[i], alpha=s)

                losses.append(test(self.data, self.model, self.loss_fn, self.device))

                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(epoch_step[i], alpha=-s)
                #print('param. change and forward prop. time:', time.time()-time0)

            #chooses best step size
            best_step = steps[np.argmin(np.asarray(losses))]
            best_steps.append(best_step)
            #print('best step scalar:', best_step)
            #takes step
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    p.add_(epoch_step[i], alpha=best_step)
        
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
            
        return best_steps
            
#implementing the EVE alorithm, but instead of checking a few set steps, steps forward by step_size until loss starts increasing
class EpochEVE2(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, step_size=0.1, step_lim=1000):
        defaults = dict(step_size=step_size, step_lim=step_lim)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(EpochEVE2, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]

    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)
        best_steps = []
        #losses = []
        for group in self.param_groups:
            step_size = group['step_size']
            step_lim = group['step_lim']

            for i,p in enumerate(group['params']):
                
                if p.grad is not None:
                    #getting step direction (and length)
                    this_step = p.clone().detach() - self.model_state[i]
                    
                    #getting baseline for loss
                    #p.add_(this_step, alpha=1)
                    curr_loss = test(self.data, self.model, self.loss_fn, self.device)
                    #losses.append(curr_loss)
                    #p.add_(this_step, alpha=-1)
                    
                    #steps in direction until the loss increases
                    step_idx = 0
                    ok = True
                    while ok == True:
                        p.add_(this_step, alpha=step_size)
                        this_loss = test(self.data, self.model, self.loss_fn, self.device)
                        #losses.append(curr_loss)
                        
                        if this_loss <= curr_loss:
                            step_idx += 1
                            curr_loss = this_loss
                            
                        elif this_loss > curr_loss:
                            p.add_(this_step, alpha=-step_size)
                            ok = False
                            
                        if step_lim != None:
                            if step_idx > step_lim:
                                ok = False
                        
                        #print(curr_loss, this_loss, ok)
                    #print('final step scalar:', step_size*step_idx)
                    best_step = step_idx*step_size
                    best_steps.append(best_step)
                    
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
        return best_steps
            
#same as EVE2, but instead of stepping for each ind. param., steps the whole model in the epoch dir., which hopefully will be more accurate and faster than EVE2 (and more true to Dr. Sokoloff's original intention as I understood it) 
class EpochEVE2_allparams(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, step_size=0.1, step_lim=1000):
        defaults = dict(step_size=step_size, step_lim=step_lim)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(EpochEVE2_allparams, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]

    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)
        best_steps = []
        losses = []
        for group in self.param_groups:
            step_size = group['step_size']
            step_lim = group['step_lim']
            new_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
            #print(len(self.model_state), len(new_state))
            epoch_step = [new_state[i]-self.model_state[i] for i in range(len(new_state))]
            #print(epoch_step)
            curr_loss = test(self.data, self.model, self.loss_fn, self.device)
            losses.append(curr_loss)
            
            
            #steps in direction until the loss increases
            step_idx = 0
            ok = True
            while ok == True:
                
                #iterate through the parameters and step forward
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        #print(step_size, p.add_(epoch_step[i], alpha=step_size))
                        p.add_(epoch_step[i], alpha=step_size)
                #print("stepped forward")
                
                this_loss = test(self.data, self.model, self.loss_fn, self.device)
                losses.append(this_loss)
                #print("Evaluated loss")
                
                if this_loss <= curr_loss:
                    step_idx += 1
                    curr_loss = this_loss
                    #print("cont. loop")
                
                elif this_loss > curr_loss:
                    for i,p in enumerate(group['params']):
                        if p.grad is not None:
                            p.add_(epoch_step[i], alpha=-step_size)
                    ok = False
                    #print("End loop")
                
                if step_lim != None:
                    if step_idx > step_lim:
                        ok = False
            #print('final step scalar:', step_size*step_idx)
            best_step = step_size*step_idx
            best_steps.append(best_step)
                    
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
        return best_steps