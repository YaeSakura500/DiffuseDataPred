import torch
import torch.nn as nn

def Not0Loss(pred,truth,ep=1e-5):
    return torch.mean((pred-truth)**2/(pred**2+ep))

# a=torch.randn((3,4,5,6,7,8),requires_grad=True)
# b=torch.rand((3,4,5,6,7,8),requires_grad=True)

# c=Not0Loss(a,b)

# print(c)