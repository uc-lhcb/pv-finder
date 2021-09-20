from torch.optim import Optimizer
import torch
import math
import numpy as np
import time
import random

#first section is modified versions of Adam

# this is just Adam as implemented in pytorch, but modified so that it 
# returns 
class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    #adding location to store gradients, momentums
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            #these are normally referenced only in the functional call
            amsgrad=group['amsgrad']
            lr=group['lr']
            weight_decay=group['weight_decay']
            eps=group['eps']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            #replacing functional API call with the actual computation
            sq_step_size = 0
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                #print(grad)
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                if amsgrad:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                
                param.addcdiv_(exp_avg, denom, value=-step_size)
                
                ######################################################### 
                #below here are my modifications
                
                #add the squared norm of the change in this parameter, for all
                #all parameters to get the total squared norm of the step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy()
            
        #these are getting the gradients and momenta that Adam used for this step
        grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        m1 = [t.clone().cpu().numpy() for t in exp_avgs]
        m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]    
            
        #returns list of grad tensors, lists of momenta tensors, and the step size
        return grad, m1, m2, np.sqrt(sq_step_size)   
    
#this version of Adam stores the steps taken, until the steps are cleared
#designed to make comparing model parameter changes epoch to epoch
# - the most valuable part of this is using the prev_step attribute to compare
#   epoch steps during training
class reversible_Adam(Optimizer):
   
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(reversible_Adam, self).__init__(params, defaults)
        self.prev_step = []

    def __setstate__(self, state):
        super(reversible_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    #clears the prev step list
    @torch.no_grad()
    def clear_prev_step(self):
        for i,p in enumerate(self.prev_step):
            self.prev_step[i] = torch.zeros_like(p)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            #these are normally referenced only in the functional call
            amsgrad=group['amsgrad']
            lr=group['lr']
            weight_decay=group['weight_decay']
            eps=group['eps']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                    #print(p.name, state)
                    
            if len(self.prev_step) < 1:
                for i, param in enumerate(params_with_grad):
                    self.prev_step.append(torch.zeros_like(param))
    
            #replacing functional API call with the actual computation
            sq_step_size = 0
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                if amsgrad:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                
                param.addcdiv_(exp_avg, denom, value=-step_size)
                
                #add the squared elements of this part of the overall step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy()
                
                #adds this step to the stored previous step
                self.prev_step[i].addcdiv_(exp_avg, denom, value=-step_size)
                       
        grad = [p.grad.clone().cpu() for p in group['params'] if p.grad is not None]
        m1 = [t.clone().cpu() for t in exp_avgs]
        m2 = [t.clone().cpu() for t in exp_avg_sqs]
        
        #returns list of grad tensors, lists of momenta tensors, and the step size
        return grad, m1, m2, np.sqrt(sq_step_size)  
    
    #this steps the model backwards, by whatever is stored in the prev step
    #I tended to avoid using this, and instead loaded state dicts for both the 
    #model and optimizer if I wanted to backstep the training
    @torch.no_grad()
    def backstep(self):
        params_with_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
        
        for i,p in enumerate(params_with_grad):
            p.add_(self.prev_step[i], alpha=-1)
        return    
    
##########################################################################################
#this part is the epoch optimizers that acted at the end of each epoch

#this calculates the norm of a list of tensors, treating the list as a single vector 
def step_norm(tens_list):
    return torch.sqrt(sum([torch.sum(torch.square(t)) for t in tens_list]))

#this calculates the normed dot product between two lists of tensors
def normed_dot(list_a, list_b):
    dot = sum([torch.sum(torch.mul(list_a[i], list_b[i])) for i in range(len(list_a))])
    return dot/(step_norm(list_a)*step_norm(list_b))

#this used to evaluate the model's loss in the EVE and EVE2 optimizer
def test(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    return test_loss

#implementing the EVE concept at the end of each epoch, in the direction of the overall epoch step
class EpochEVE(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, steps=[0, 0.5, 1, 2, 4, 8]):
        defaults = dict(steps=steps, num_mb=num_mb)#[torch.zeros_like(param).detach() for param in params])
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        super(EpochEVE, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    @torch.no_grad()
    def step(self, closure=None):
        best_steps = []
        for group in self.param_groups:
            steps = group['steps']
            
            #determines epoch step direction
            epoch_step = []
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    epoch_step.append(p.clone().detach() - self.model_state[i])
                    
            losses = []

            #explores options in step direction
            for s in steps:
                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(epoch_step[i], alpha=s)

                losses.append(test(self.data, self.model, self.loss_fn, self.device))

                for i,p in enumerate(group['params']):
                    if p.grad is not None:
                        p.add_(epoch_step[i], alpha=-s)

            #chooses best step size
            best_step = steps[np.argmin(np.asarray(losses))]
            best_steps.append(best_step)

            #takes step
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    p.add_(epoch_step[i], alpha=best_step)
        
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
            
        #returns the step scalar(s) used
        return best_steps

# a variant on the EVE optimizer above, but instead of looking only at a set of predetermined
# step scalars it steps in the direction of the epoch by a step_size until either the
# loss starts increasing or it reaches the step_lim
class EpochEVE2(Optimizer):

    def __init__(self, params, model, loss_fn, data, device, step_size=0.1, step_lim=100):
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
                
                if step_lim != None:
                    if step_idx > step_lim:
                        ok = False
            best_step = step_size*step_idx
            best_steps.append(best_step)
                    
        #set model state to new state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
            
        #returns the step scalar(s) used
        return best_steps    
    
# this takes a step at the end of an epoch in the direction of the epoch, with a scalar multiple
# factor*lr. This was created so that the normed dot between the two most recent epoch steps could 
# be used as the factor to decide how big a step to take.
class EpochStep(Optimizer):

    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr, momentum_list=None)
        super(EpochStep, self).__init__(params, defaults)
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
         
    #this meant to be used at the end of each (or any) epoch, after the optimizer has iterated through the 
    #dataset
    @torch.no_grad()
    def step(self, factor, closure=None):
        
        #calculating current state
        for group in self.param_groups:
            new_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
        
        #calculating the epoch step
        this_epoch_step = [new_state[i]-self.model_state[i] for i in range(len(new_state))]
        
        #steps in the epoch step direction with scalar lr*factor
        for group in self.param_groups:
            lr = group['lr']
            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    this_step = this_epoch_step[i]
                    p.add_(this_step, alpha=lr*factor)
        
        
        #stores current model state
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.requires_grad is True]
            
        #returns the scalar multiplied onto the epoch
        return lr*factor

# an implementation of a version of momentum but over epochs, instead of minibatches.
# I've not experimented much with it, but I figured I would include it in case it was useful
class EpochMomentum(Optimizer):

    def __init__(self, params, lr=1, beta=0.9):
        defaults = dict(lr=lr, beta=beta, momentum_list=None)
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
            
        for group in self.param_groups:
            self.model_state = [param.clone().detach() for param in group['params'] if param.grad is not None]
        
        return None
        
        