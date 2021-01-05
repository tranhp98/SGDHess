import torch
from torch.optim import Optimizer
class SGDHess(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, clip=None):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.clip = clip
        self.iteration = -1
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDHess, self).__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            for p in group['params']:
                state = self.state[p]
                state['displacement'] = torch.zeros_like(p)
                state['max_grad'] = torch.zeros_like(p)
                if self.clip == 'coord':
                    state['max_grad'] = torch.zeros_like(p)
                if self.clip == 'norm':
                    state['max_grad'] = torch.zeros(1)

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
        self.iteration += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            vector = []
            grads = []
            param = []
            for p in group['params']:
                if p.grad is None:
                    continue
                vector.append(self.state[p]['displacement'])
                grads.append(p.grad)
                param.append(p)

            hvp = torch.autograd.grad(outputs = grads, inputs = param, grad_outputs=vector)

            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        displacement, max_grad = state['displacement'], state['max_grad'] 
                        with torch.no_grad():
                            d_p = p.grad
                            if weight_decay != 0:
                                d_p = d_p.add(p, alpha=weight_decay)
                                hvp[i].add_(weight_decay * displacement)
                            if momentum != 0:
                                if 'momentum_buffer' not in state:
                                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                                else:
                                    buf = state['momentum_buffer']
                                    buf.add_(hvp[i]).mul_(momentum).add_(d_p, alpha=1 - dampening)
                                    #g = buf.add(hvp[i]).mul(1-momentum).add(d_p, alpha=momentum)
                                    val = None
                                    if self.clip is not None:
                                        if self.clip == 'coord':
                                            torch.clamp_(buf, -max_grad, max_grad)
                                            max_grad.copy_(torch.maximum((1-dampening)/(1-momentum)*torch.abs(d_p), max_grad))
                                        if self.clip == 'norm':
                                            torch.nn.utils.clip_grad_norm_(buf, max_grad)
                                            max_grad.copy_(torch.maximum((1-dampening)/(1-momentum)*torch.norm(d_p), max_grad))
                                    #buf.add_(hvp[i]).mul_(momentum).add_(d_p, alpha=1 - dampening)
                                if nesterov:
                                    d_p = d_p.add(buf, alpha=momentum)
                                else:
                                    d_p = buf
                            displacement.copy_(-d_p * group['lr'])
                            p.add_(displacement)
                        i += 1
                            
        return loss
