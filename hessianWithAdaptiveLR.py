import torch
import numpy as np
from torch.optim import Optimizer
from torch.autograd import Variable
class hessian(Optimizer):
    def __init__(self, params, lr= 0.1, momentum = 0.1, weight_decay = 0.0005):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))
        self.iteration = -1
        defaults = dict(lr=lr, momentum = momentum, weight_decay = weight_decay)
        super(hessian, self).__init__(params, defaults)
        #initialize variables 
        for group in self.param_groups:
             for p in group['params']:
                state = self.state[p]
                state['prev_param'] = torch.zeros_like(p)
                state['prev_g'] = torch.zeros_like(p)
                state['current_param'] = torch.zeros_like(p)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        self.iteration += 1
        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            vector = []
            grads = []
            param = []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                prev_param, prev_g, current_param = state['prev_param'], state['prev_g'], state['current_param']
                with torch.no_grad():
                    if(self.iteration == 0):
                        prev_g.add_(p.grad)
                        prev_param.add_(p)
                        g_norm = torch.div(prev_g, torch.norm(prev_g))
                        current_param.add_(prev_param.add(g_norm, alpha = -group['lr']))
                        p.add_(g_norm, alpha = -group['lr'])
                    else:
                        vector.append(current_param.add(prev_param, alpha = -1))
                        grads.append(p.grad)
                        param.append(p)
                    
            if(self.iteration > 0):
              
                hvp = torch.autograd.grad(outputs = grads, inputs = param, grad_outputs=vector) 
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        state = self.state[p]
                        prev_param, prev_g, current_param = state['prev_param'], state['prev_g'], state['current_param']
                        decay_rate = 1/(1+weight_decay*self.iteration)
                        prev_g.add_(hvp[i]).mul_(1-group['momentum']).add_(p.grad, alpha = group['momentum'])
                        prev_param.copy_(current_param).detach()
                        g_norm = torch.div(prev_g, torch.norm(prev_g))
                        current_param.add_(g_norm, alpha = -lr*decay_rate)
                        p.add_(g_norm, alpha = -lr*decay_rate)
                        i += 1
               
               
        return loss
                    
                    
                    
                    
                    
                
                