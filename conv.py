import torch

a=torch.rand((6,6,6,6,6),dtype=torch.float32)
conv = torch.nn.Conv3d(6,6,3,padding='same')
b=conv(a)
print(a)
print(b)
print("aaa")