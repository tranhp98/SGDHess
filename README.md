# SGD with Hessian-based momentum

A Pytorch implementation of SGDHess, a SGD-based algorithm that incorporates Second-order information to expedite the training.

## Quick overview

Our implementation is based on the official Pytorch implementation of SGD. 

```python3
buf.add_(hvp[i]).add_(displacement, alpha = weight_decay).mul_(momentum).add_(d_p, alpha=1 - dampening)
```
