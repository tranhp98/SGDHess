import torch
import numpy as np
from torch.optim import Optimizer


class hessian(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))
        self.iteration = 0
        self.prev_param = torch.Tensor()
        self.vector = torch.Tensor()
        self.prev_g = torch.Tensor()
        self.current_param = torch.Tensor()
        defaults = dict(lr=lr, momentum=momentum)
        super(hessian, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                if self.iteration == 0:
                    # initialization step
                    self.prev_param = p.data
                    current_g = p.grad.data
                    g_norm = torch.norm(current_g)
                    self.current_param = p.data - lr * (current_g / g_norm)
                    self.prev_g = current_g
                    self.iteration += 1
                else:
                    self.vector = self.current_param - self.prev_param
                    # compute hessian
                    dot_product = p.grad.data @ self.vector
                    print(self.current_param.requires_grad)
                    # temp_dot = dot_product
                    # temp_dot.requires_grad = True
                    # temp = p.data
                    # temp.requires_grad = True
                    hvd = torch.autograd.grad(dot_product, self.current_param, create_graph=True)
                    # hvd = torch.autograd.grad(temp_dot, temp, create_graph = True)
                    # continue execution
                    current_g = (1 - momentum) * (self.prev_g + hvd) + momentum * self.grad.data

                    # update param
                    g_norm = torch.norm(current_g)
                    self.prev_param = self.current_param
                    self.current_param = self.prev_param - lr * (current_g / g_norm)
                    self.prev_g = current_g

        return loss






