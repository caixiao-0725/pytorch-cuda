import torch
import cppcuda_tutorial



if __name__ == '__main__':
    feats = torch.ones(2,device='cuda')
    point = torch.ones(2,device='cuda')

    out = cppcuda_tutorial.trilinear_interpolation(feats, point)    

    print(out)