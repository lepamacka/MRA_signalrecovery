import torch 
import numpy as np
from signalsamplers import circulant

def align(input, base):        
    input_circulant = circulant(input, dim=-1)
    best_shift_idx = (base.unsqueeze(0) - input_circulant).square().sum(dim=-1).min(dim=-1)[1]
    if input.ndim == 1:
        return input_circulant[best_shift_idx, :]
    elif input.ndim == 2:
        return input_circulant[torch.arange(input.shape[0]), best_shift_idx, :]
    else: 
        raise ValueError(f"{input.ndim = }, has to be 1 or 2.")

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