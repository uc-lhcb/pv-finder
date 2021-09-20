import torch
#from torch.optim import functional as F
from torch.optim import Optimizer

#from functional import adam

import numpy as np
import torch
import math

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
                    #print(p.name, state)

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

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
                #print(step_size)
                param.addcdiv_(exp_avg, denom, value=-step_size)
                #add the squared elements of this part of the overall step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
        grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
        m1 = [t.clone().cpu().numpy() for t in exp_avgs]
        m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]    
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2, np.sqrt(sq_step_size)   

class Reversible_Adam(Optimizer):
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
        super(Reversible_Adam, self).__init__(params, defaults)
        self.prev_step = []

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        self.prev_step = []
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

            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
            m1 = [t.clone().cpu().numpy() for t in exp_avgs]
            m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

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
                #print(step_size)
                param.addcdiv_(exp_avg, denom, value=-step_size)
                #add the squared elements of this part of the overall step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
                self.prev_step.append(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size))
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2, np.sqrt(sq_step_size)  
    
    @torch.no_grad()
    def backstep(self):
        params_with_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
        
        for i,p in enumerate(params_with_grad):
            #print(p.size(), self.prev_step[i].size())
            p.add_(self.prev_step[i], alpha=-1)
        return  
    
class epoch_Reversible_Adam(Optimizer):
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
        super(epoch_Reversible_Adam, self).__init__(params, defaults)
        self.prev_step = []

    def __setstate__(self, state):
        super(epoch_Reversible_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    @torch.no_grad()
    def clear_prev_step(self):
        for i,p in enumerate(self.prev_step):
            self.prev_step[i] = torch.zeros_like(p)
    
    @torch.no_grad()
    def step(self, closure=None, lr_factor=0.8, loss_increase_limit=0.02):
        #self.prev_step = []
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

            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
            m1 = [t.clone().cpu().numpy() for t in exp_avgs]
            m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

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
                #print(step_size)
                param.addcdiv_(exp_avg, denom, value=-step_size)
                #add the squared elements of this part of the overall step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
                self.prev_step[i].addcdiv_(exp_avg, denom, value=-step_size)
                #self.prev_mbstep.append(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size))
                #if i == 0:
                #    print(self.prev_step[i])
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2, np.sqrt(sq_step_size)  
    
    @torch.no_grad()
    def backstep(self):
        params_with_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
        
        for i,p in enumerate(params_with_grad):
            #print(p.size(), self.prev_step[i].size())
            p.add_(self.prev_step[i], alpha=-1)
        return    
    
class Fixed_Adam(Optimizer):
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
        super(Fixed_Adam, self).__init__(params, defaults)

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
                    #print(p.name, state)

            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
            m1 = [t.clone().cpu().numpy() for t in exp_avgs]
            m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

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

                step_size = 1 / bias_correction1
                
                norm_step = torch.sqrt(torch.sum(torch.square(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-step_size)))).cpu().numpy()
                
                #print(step_size)
                param.addcdiv_(exp_avg, denom, value=-((lr*step_size)/norm_step))
                #add the squared elements of this part of the overall step
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-(lr/norm_step)) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2, np.sqrt(sq_step_size)  
  
class Test_Fixed_Adam(Optimizer):
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
        super(Test_Fixed_Adam, self).__init__(params, defaults)

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
                    #print(p.name, state)

            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
            m1 = [t.clone().cpu().numpy() for t in exp_avgs]
            m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

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

                step_size = 1# / bias_correction1
                
                #currently testing if the improvement is normalizing each parameters step size
                #next will test if removing the bias correction is the cause
                
                norm_step = torch.sqrt(torch.sum(torch.square(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-1)))).cpu().numpy()
                
                #print(step_size)
                param.addcdiv_(exp_avg, denom, value=-((lr*step_size)/norm_step))
                #add the squared elements of this part of the overall step
                #sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=-(lr/norm_step)) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2#, np.sqrt(sq_step_size)

class Fixed_Adam2(Optimizer):
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
        super(Fixed_Adam2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        print('stepping')
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

            grad = [p.grad.clone().cpu().numpy() for p in group['params'] if p.grad is not None]
            m1 = [t.clone().cpu().numpy() for t in exp_avgs]
            m2 = [t.clone().cpu().numpy() for t in exp_avg_sqs]

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])

            #replacing functional API call with the actual computation
            sq_step_size = 0.0
            raw_steps = []
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

                #step_size = lr# / bias_correction1
                
                #norm_step = torch.sqrt(torch.sum(torch.square(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=1)))).cpu().numpy()
                
                #print(step_size)
                #param.addcdiv_(exp_avg, denom, value=-(step_size/norm_step))
                #add the squared elements of this part of the overall step
                raw_steps.append(torch.zeros_like(param).addcdiv_(exp_avg, denom, value=1))
                sq_step_size += torch.sum(torch.square( torch.zeros_like(param).addcdiv_(exp_avg, denom, value=1) )).cpu().numpy()
                #print('part of sq sum:', torch.sum(torch.square( param.addcdiv_(exp_avg, denom, value=-step_size) )).cpu().numpy())
            
            full_norm = np.sqrt(sq_step_size)
            corrected_step_size = 0.0
            for i, param in enumerate(params_with_grad):
                param.add_(raw_steps[i], alpha=-(lr/full_norm))
                corrected_step_size += torch.sum(torch.square(torch.zeros_like(param).add_(raw_steps[i], alpha=-(lr/full_norm))))
            
            print(full_norm, corrected_step_size)
            
        #print('size of this step:', np.sqrt(sq_step_size))
        return grad, m1, m2, np.sqrt(sq_step_size) 