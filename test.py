import torch
import cppcuda_tutorial

feats = torch.ones(2)
point = torch.ones(2)

out = cppcuda_tutorial.trilinear_interpolation(feats, point)    

print(out)

a = 10
b = 10

c = cppcuda_tutorial.add(a, b)
print(c)