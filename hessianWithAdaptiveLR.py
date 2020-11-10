import torch
import numpy as np
from torch.optim import Optimizer
from torch.autograd import Variable


class hessian(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.1, weight_decay=.0005, max_iteration=1000, eta=0.001):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        self.iteration = -1
        self.hvp = []
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, max_iteration=max_iteration, eta=eta)
        super(hessian, self).__init__(params, defaults)
        # initialize variables
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
            max_iteration = group['max_iteration']
            eta = group['eta']
            vector = ()
            grads = ()
            param = ()
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                prev_param, prev_g, current_param = state['prev_param'], state['prev_g'], state['current_param']
                with torch.no_grad():
                    if (self.iteration == 0):
                        prev_g.add_(p.grad)
                        prev_param.add_(p)
                        g_norm = torch.div(prev_g, torch.norm(prev_g))
                        current_param.add_(prev_param.add(g_norm, alpha=-group['lr']))
                        p.add_(g_norm, alpha=-group['lr'])
                    else:
                        # vector.append(current_param.add(prev_param, alpha = -1))
                        vector = vector + (current_param.add(prev_param, alpha=-1),)
                        grads = grads + (p.grad,)
                        param = param + (p,)

            if (self.iteration > 0):
                # dot_product = sum([(g * v).sum() for g, v in zip(grads, vector)])
                # hvp = torch.autograd.grad(dot_product, param)
                # hvp = torch.autograd.grad(outputs = grads, inputs = param, grad_outputs=vector)
                # compute hvp every 3 iterations
                '''
                if (self.iteration % 3 == 1):
                    self.hvp = []
                    hvp = torch.autograd.grad(outputs=grads, inputs=param, grad_outputs=vector)
                    for i in range(len(hvp)):
                        self.hvp.append(hvp[i])
                '''
                decay = (1 - float(self.iteration) / max_iteration) ** 2
                global_lr = lr * decay
                #print("global", global_lr)
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        state = self.state[p]
                        prev_param, prev_g, current_param = state['prev_param'], state['prev_g'], state['current_param']
                        local_lr = torch.div(eta * torch.norm(p),
                                             torch.add(torch.norm(p.grad), weight_decay * torch.norm(p)))
                        # print("local", local_lr)
                        prev_g.add_(self.hvp[i]).mul_(1 - group['momentum']).add_(p.grad, alpha=group['momentum'])
                        prev_param.copy_(current_param).detach()
                        g_norm = torch.div(prev_g, torch.norm(prev_g))
                        #print(global_lr * local_lr)
                        current_param.add_(g_norm, alpha=-global_lr * local_lr)
                        p.add_(g_norm, alpha=-global_lr * local_lr)
                        i += 1
                # Step 1: test with other optimizer
                # Step 2: Linear regression -> hessian constant
                # Step 3: Sine
                # Measure total run-time

        return loss






