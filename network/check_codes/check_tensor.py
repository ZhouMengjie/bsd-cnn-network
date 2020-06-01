import torch

x = torch.tensor([1.0],requires_grad=True)
is_train = True

with torch.set_grad_enabled(is_train):
    y = x * 2

print(y.requires_grad) # False or True