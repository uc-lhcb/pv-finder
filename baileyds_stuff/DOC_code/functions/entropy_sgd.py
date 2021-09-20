#from https://github.com/ucla-vision/entropy-sgd/blob/master/python/optim.py
from torch.optim import Optimizer
from copy import deepcopy
import numpy as np
import torch as th

class EntropySGD(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=0, eps=1e-4, g0=1e-2, g1=0)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropySGD, self).__init__(params, config)
        self.config = config
        
    def step(self, closure=None, model=None, criterion=None):
        assert (closure is not None) and (model is not None) and (criterion is not None), \
                'attach closure for Entropy-SGD, model and criterion'
        mf = closure()

        c = self.config
        lr = c['lr']
        mom = c['momentum']
        wd = c['weight_decay']
        damp = c['damp']
        nesterov = c['nesterov']
        L = int(c['L'])
        eps = c['eps']
        g0 = c['g0']
        g1 = c['g1']

        #params = self.param_groups[0]['params']
        
        params = []
        for p in self.param_groups[0]['params']:
            if p.grad is not None:
                params.append(p)

        with th.no_grad():
            state = self.state
            # initialize
            if not 't' in state:
                state['t'] = 0
                state['wc'], state['mdw'] = [], []
                for w in params:
                    state['wc'].append(deepcopy(w))
                    state['mdw'].append(deepcopy(w.grad))

                state['langevin'] = dict(mw=deepcopy(state['wc']),
                                        mdw=deepcopy(state['mdw']),
                                        eta=deepcopy(state['mdw']),
                                        lr = 0.1,
                                        beta1 = 0.75)

            lp = state['langevin']
            for i,w in enumerate(params):
                state['wc'][i].copy_(w)
                lp['mw'][i].copy_(w)
                lp['mdw'][i].zero_()
                lp['eta'][i].normal_()

            state['debug'] = dict(wwpd=0, df=0, dF=0, g=0, eta=0)
            llr, beta1 = lp['lr'], lp['beta1']
            g = g0*(1+g1)**state['t']

        for i in range(L):
            f = closure()
            for wc,w,mw,mdw,eta in zip(state['wc'], params, \
                                    lp['mw'], lp['mdw'], lp['eta']):
                with th.no_grad():
                    dw = w.grad
                    #print('grad:', dw)

                    if wd > 0:
                        dw.add_(w, alpha=wd)
                    if mom > 0:
                        mdw.mul_(mom).add_(dw, alpha=1-damp)
                        if nesterov:
                            dw.add_(mdw, alpha=mom)
                        else:
                            dw = mdw
                    #print('mom:', dw)

                    # add noise
                    eta.normal_()
                    dw.add_(wc-w, alpha=-g).add_(eta, alpha=eps/np.sqrt(0.5*llr))
                    #print('noise:', dw)
                    # update weights
                    w.add_(dw, alpha=-llr)
                    mw.mul_(beta1).add_(w, alpha=1-beta1)

        if L > 0:
            # copy model back
            with th.no_grad():
                for i,w in enumerate(params):
                    w.copy_(state['wc'][i])
                    w.grad.copy_(w-lp['mw'][i])

        with th.no_grad():
            for w,mdw,mw in zip(params, state['mdw'], lp['mw']):
                dw = w.grad

                if wd > 0:
                    dw.add_(w, alpha=wd)
                if mom > 0:
                    mdw.mul_(mom).add_(dw, alpha=1-damp)
                    if nesterov:
                        dw.add_(mdw, alpha=mom)
                    else:
                        dw = mdw
                #print(w, mdw, mw)
                #print(th.zeros_like(w).add_(dw, alpha=-lr))
                w.add_(dw, alpha=-lr)

        return mf