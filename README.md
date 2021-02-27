# SGD with Hessian-based momentum

A Pytorch implementation of SGDHess, a SGD-based algorithm that incorporates Second-order information to expedite the training.

## Quick overview

Our implementation is based on the official Pytorch implementation of SGD. Our most important modification is the addition of Hessian-vectors product to "correct" the momentum of the gradient estimate. This "correction" would allow our algorithm to take advantage of Second-order information and make faster progress than the normal SGD. 

```python3
buf.add_(hvp[i]).add_(displacement, alpha = weight_decay).mul_(momentum).add_(d_p, alpha=1 - dampening)
```
This Hessian-vectors product term is efficiently computed through Pytorch's automatic differentiation package.
```python3
 hvp = torch.autograd.grad(outputs = grads, inputs = param, grad_outputs=vector)
```

