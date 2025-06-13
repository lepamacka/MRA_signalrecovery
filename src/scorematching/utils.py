import torch 
import numpy as np

@torch.no_grad()
def Langevin_sampler(
    input, 
    scoremodel=None,
    conditioner=None, 
    num_steps=1000, 
    eps=1e-3, 
    device='cpu',
): 
    sqrtfac = np.sqrt(2.*eps)
    output = input.clone()
    for _ in range(num_steps):
        tmp = torch.zeros_like(output, requires_grad=False)
        if scoremodel is not None:
            tmp += scoremodel(output) 
        if conditioner is not None:
            tmp += conditioner(output)
        noise = torch.randn_like(output, requires_grad=False)
        output += eps * tmp + sqrtfac * noise
    return output