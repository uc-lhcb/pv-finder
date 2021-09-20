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

def test_mb_shuff(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    idx = random.randint(0, len(dataloader)-1)
    #print([_ for _ in dataloader][0])
    data = list(dataloader)[idx]
    X = data[0].to(device)
    y = data[1].to(device)
    with torch.no_grad():
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
    return test_loss

"""
Some basic optimization algorithms
"""

#implementing EVE for the SGD alogorithm, where steps are the scalars multiplied onto the opt step where the loss is calculated
#this calculates loss over entire training dataset
class GD_EVE(Optimizer):
    
    def __init__(self, params, model, loss_fn, data, device, lr=1e-3, steps=[0, 1, 3, 10, 100]):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, steps=steps)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(GD_EVE, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                steps = group['steps']
                if p.grad is not None:
                    #gets initial step
                    this_step = p.grad.clone().detach().mul_(-lr)
                    losses = []
                    
                    #explores options in step direction
                    
                    for s in steps:
                        time0 = time.time()
                        p.add_(this_step, alpha=s)
                        losses.append(test(self.data, self.model, self.loss_fn, self.device))
                        p.add_(this_step, alpha=-s)
                        #print('p change and forw. prop. time:', time.time()-time0)
                    
                    #chooses best step size
                    best_step = steps[np.argmin(np.asarray(losses))]
                    
                    #print(best_step)
                    
                    #steps
                    p.add_(this_step, alpha=best_step)
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        return grad

#same as above, but calculates training loss on same minibatch that the gradient is calculated on    
class GD_EVE_mb(Optimizer):
    
    def __init__(self, params, model, loss_fn, device, lr=1e-3, steps=[0, 1, 3, 10, 100]):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, steps=steps)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        #self.data = data
        super(GD_EVE_mb, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, data, closure=None):
        best_steps = []
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                steps = group['steps']
                if p.grad is not None:
                    #gets initial step
                    this_step = p.grad.clone().detach().mul_(-lr)
                    losses = []
                    
                    #explores options in step direction
                    
                    for s in steps:
                        time0 = time.time()
                        p.add_(this_step, alpha=s)
                        losses.append(test_mb(data, self.model, self.loss_fn, self.device))
                        p.add_(this_step, alpha=-s)
                        #print('p change and forw. prop. time:', time.time()-time0)
                    
                    #chooses best step size
                    best_step = steps[np.argmin(np.asarray(losses))]
                    best_steps.append(best_step)
                    
                    #print(best_step)
                    
                    #steps
                    p.add_(this_step, alpha=best_step)
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        return best_steps

class GD_EVE_allparams_mb(Optimizer):
    
    def __init__(self, params, model, loss_fn, device, lr=1e-3, steps=[0, 1, 3, 10, 100]):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, steps=steps)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        #self.data = data
        super(GD_EVE_allparams_mb, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, data, closure=None):
        best_steps = []
        for group in self.param_groups:
            lr = group['lr']
            steps = group['steps']
            
            this_step = [p.grad.clone().detach().mul_(-lr) for p in group['params'] if p.grad is not None]
            losses = []

            #explores options in step direction

            for s in steps:
                time0 = time.time()
                
                #iterate through the parameters and step forward
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(this_step[i], alpha=s)
                        
                losses.append(test_mb(data, self.model, self.loss_fn, self.device))
                
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(this_step[i], alpha=-s)
                print(losses)
                #print('p change and forw. prop. time:', time.time()-time0)

            #chooses best step size
            best_step = steps[np.argmin(np.asarray(losses))]
            best_steps.append(best_step)

            print(best_step)

            #takes the step
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    p.add_(this_step[i], alpha=s)
                    
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        return best_steps
    
#same as above, but calculates training loss on random minibatch 
class GD_EVE_mb_shuff(Optimizer):
    
    def __init__(self, params, model, loss_fn, data, device, lr=1e-3, steps=[0, 1, 3, 10, 100]):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, steps=steps)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(GD_EVE_mb_shuff, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        best_steps = []
        #pick random minibatch
        time1 = time.time()
        idx = random.randint(0, len(self.data)-1)
        print('randint select time:', time.time()-time1)
        time2 = time.time()
        this_data = list(self.data)[idx]
        print('data select time:', time.time()-time2)
        
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                steps = group['steps']
                if p.grad is not None:
                    #gets initial step
                    this_step = p.grad.clone().detach().mul_(-lr)
                    losses = []
                    
                    #explores options in step direction
                    
                    for s in steps:
                        time0 = time.time()
                        p.add_(this_step, alpha=s)
                        losses.append(test_mb(this_data, self.model, self.loss_fn, self.device))
                        p.add_(this_step, alpha=-s)
                        #print('p change and forw. prop. time:', time.time()-time0)
                    
                    #chooses best step size
                    best_step = steps[np.argmin(np.asarray(losses))]
                    best_steps.append(best_step)
                    
                    #print(best_step)
                    
                    #steps
                    p.add_(this_step, alpha=best_step)
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        return best_steps

#same as above, but calculates training loss on same minibatch that the gradient is calculated on    
class GD_EVE2_mb(Optimizer):
    
    def __init__(self, params, model, loss_fn, device, lr=1e-3, step_size=1, step_lim=100):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, step_size=step_size, step_lim=step_lim)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        #self.data = data
        super(GD_EVE2_mb, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, data, closure=None):
        best_steps = []
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                step_size = group['step_size']
                step_lim = group['step_lim']
            
                if p.grad is not None:
                    #gets initial step
                    this_step = p.grad.clone().detach().mul_(-lr)
                    curr_loss = test_mb(data, self.model, self.loss_fn, self.device)
                    losses = []
                    
                    #steps in direction until the loss increases
                    step_idx = 0
                    ok = True
                    while ok == True:
                        p.add_(this_step, alpha=step_size)
                        this_loss = test_mb(data, self.model, self.loss_fn, self.device)
                        
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
                    
                    best_step = step_size*step_idx
                    best_steps.append(best_step)
                    
       
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        #print(best_steps)
        return best_steps
    
class GD_EVE2_mb_shuff(Optimizer):
    
    def __init__(self, params, model, loss_fn, data, device, lr=1e-3, step_size=1):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, step_size=step_size)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(GD_EVE2_mb_shuff, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        best_steps = []
        
        #pick random minibatch
        time1 = time.time()
        idx = random.randint(0, len(self.data)-1)
        #print('randint select time:', time.time()-time1)
        time2 = time.time()
        this_data = list(self.data)[idx]
        #print('data select time:', time.time()-time2)
        
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                step_size = group['step_size']
            
                if p.grad is not None:
                    #gets initial step
                    this_step = p.grad.clone().detach().mul_(-lr)
                    curr_loss = test_mb(this_data, self.model, self.loss_fn, self.device)
                    losses = []
                    
                    #steps in direction until the loss increases
                    step_idx = 0
                    ok = True
                    while ok == True:
                        p.add_(this_step, alpha=step_size)
                        this_loss = test_mb(this_data, self.model, self.loss_fn, self.device)
                        
                        if this_loss <= curr_loss:
                            step_idx += 1
                            curr_loss = this_loss
                            
                        elif this_loss > curr_loss:
                            p.add_(this_step, alpha=-step_size)
                            ok = False
                        #print(curr_loss, this_loss, ok)
                    #print('final step scalar:', step_size*step_idx)
                    
                    best_step = step_size*step_idx
                    best_steps.append(best_step)
                    
       
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        #print(best_steps)
        return best_steps    

class GD_EVE2_allparams_mb(Optimizer):
    
    def __init__(self, params, model, loss_fn, device, lr=1e-3, step_size=1):
        #required dict mapping parameter names to def
        #values
        defaults = dict(lr=lr, step_size=step_size)
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        #self.data = data
        super(GD_EVE2_allparams_mb, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, data, closure=None):
        best_steps = []
        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                step_size = group['step_size']
                
                #the steps for all model params
                this_step = [p.grad.clone().detach().mul_(-lr) for p in group['params'] if p.grad is not None]
                
                #initial loss
                curr_loss = test_mb(data, self.model, self.loss_fn, self.device)
                    
                #steps in direction until the loss increases
                step_idx = 0
                ok = True
                while ok == True:

                    #iterate through the parameters and step forward
                    for i,p in enumerate(group['params']):
                        if p.grad is not None:
                            p.add_(this_step[i], alpha=step_size)
                            
                    #p.add_(this_step, alpha=step_size)
                    this_loss = test_mb(data, self.model, self.loss_fn, self.device)

                    if this_loss <= curr_loss:
                        step_idx += 1
                        curr_loss = this_loss

                    elif this_loss > curr_loss:
                        for i,p in enumerate(group['params']):
                            if p.grad is not None:
                                p.add_(this_step[i], alpha=-step_size)
                        ok = False
                    #print(curr_loss, this_loss, ok)
                #print('final step scalar:', step_size*step_idx)

                best_step = step_size*step_idx
                best_steps.append(best_step)
                
                    
       
            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        #print(best_steps)
        return best_steps

#implementing the EVE concept at the end of each epoch, in the direction of the overall epoch step
class EpochEVE(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, steps=[0]+[10**i for i in np.arange(0, 4, 0.5)]):
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

        for group in self.param_groups:
            steps = group['steps']

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    this_step = p.clone().detach() - self.model_state[i]
                    losses = []
                    
                    #explores options in step direction
                    
                    for s in steps:
                        #time0 = time.time()
                        p.add_(this_step, alpha=s)
                        losses.append(test(self.data, self.model, self.loss_fn, self.device))
                        p.add_(this_step, alpha=-s)
                        #print('param. change and forward prop. time:', time.time()-time0)
                    
                    #chooses best step size
                    best_step = steps[np.argmin(np.asarray(losses))]
                    print('best step scalar:', best_step)
                    #takes step
                    p.add_(this_step, alpha=best_step)
        
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
            
#implementing the EVE alorithm, but instead of checking a few set steps, steps forward by step_size until loss starts increasing
class EpochEVE2(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, step_size=10):
        defaults = dict(step_size=step_size)#[torch.zeros_like(param).detach() for param in params])
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

        for group in self.param_groups:
            step_size = group['step_size']

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    #getting step direction (and length)
                    this_step = p.clone().detach() - self.model_state[i]
                    
                    #getting baseline for loss
                    #p.add_(this_step, alpha=1)
                    curr_loss = test(self.data, self.model, self.loss_fn, self.device)
                    #p.add_(this_step, alpha=-1)
                    
                    #steps in direction until the loss increases
                    step_idx = 0
                    ok = True
                    while ok == True:
                        p.add_(this_step, alpha=step_size)
                        this_loss = test(self.data, self.model, self.loss_fn, self.device)
                        
                        if this_loss <= curr_loss:
                            step_idx += 1
                            curr_loss = this_loss
                            
                        elif this_loss > curr_loss:
                            p.add_(this_step, alpha=-step_size)
                            ok = False
                        #print(curr_loss, this_loss, ok)
                    print('final step scalar:', step_size*step_idx)
                    


        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
            
#same as EVE2, but instead of stepping for each ind. param., steps the whole model in the epoch dir., which hopefully will be more accurate and faster than EVE2 (and more true to Dr. Sokoloff's original intention as I understood it) 
class EpochEVE3(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, step_size=10):
        defaults = dict(step_size=step_size)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.data = data
        super(EpochEVE3, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]

    @torch.no_grad()
    def step(self, closure=None):
        #print('init model state:', self.model_state)

        for group in self.param_groups:
            step_size = group['step_size']
            new_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
            #print(len(self.model_state), len(new_state))
            epoch_step = [new_state[i]-self.model_state[i] for i in range(len(new_state))]
            #print(epoch_step)
            curr_loss = test(self.data, self.model, self.loss_fn, self.device)
            
            #steps in direction until the loss increases
            step_idx = 0
            ok = True
            while ok == True:
                
                #iterate through the parameters and step forward
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(epoch_step[i], alpha=step_size)
                
                this_loss = test(self.data, self.model, self.loss_fn, self.device)
                
                if this_loss <= curr_loss:
                    step_idx += 1
                    curr_loss = this_loss
                
                elif this_loss > curr_loss:
                    for i,p in enumerate(group['params']):
                        if p.grad is not None:
                            p.add_(epoch_step[i], alpha=-step_size)
                    ok = False
            print('final step scalar:', step_size*step_idx)
                    

        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]