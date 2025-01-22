import numpy as np
import scipy.special as special
import torch
from torch.fft import fft, ifft

def pwrspec_score(x, z, M, device='cpu'):
    assert z.shape[-1] == x.shape[-1]

    x_new = x.to('cpu')
    x_tilde = fft(x_new)
    z = z.to('cpu')

    score = torch.zeros(size=x_new.shape)
    score += pwrspec_score_comps(x_new, x_tilde, z, M, 0)

    k = 1
    while k < x_new.shape[-1]/2:
        score += pwrspec_score_comps(x_new, x_tilde, z, M, k)
        k += 1

    if x_new.shape[-1]%2 == 0:
        score += pwrspec_score_comps(x_new, x_tilde, z, M, k)

    return score.to(device)

def pwrspec_score_comps(x_new, x_tilde, z, M, k, device='cpu'):
    if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
        k_fac = 0.5
    else:
        k_fac = 1
    x_abs_k = torch.abs(x_tilde[..., k])
    xi_k = torch.sqrt(2 * k_fac * M * z[k] * x_abs_k**2).numpy()
    ifft_mat_k = ifft(torch.eye(x_new.shape[-1])[k, :])
    sym_mat_k = torch.abs(torch.einsum("...i, ...j -> ...ij", torch.conj(ifft_mat_k), ifft_mat_k))
    res = (special.iv(k_fac*M, xi_k) + special.iv(k_fac*M-2, xi_k)) / special.iv(k_fac*M-1, xi_k) #LOG IN THIS BIATCH
    res = torch.tensor(res)
    res *= x_abs_k
    res += 1 - M * (x_abs_k**2 + k_fac)
    return res.unsqueeze(1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new) / 2.

# def pwrspec_score_zero(x_new, x_tilde, z, M, k, device='cpu'):
#     x_abs_k = torch.abs(x_tilde[..., k])
#     xi_k = torch.sqrt(M * z[k] * x_abs_k**2)
#     ifft_mat_k = ifft(torch.eye(x_new.shape[-1])[k, :])
#     sym_mat_k = torch.abs(torch.outer(torch.conj(ifft_mat_k), ifft_mat_k))
#     res = (special.iv(M/2, xi_k) + special.iv(M/2-2, xi_k))/special.iv(M/2-1, xi_k)
#     res[torch.isnan(res)] = 2
#     res *= x_abs_k
#     res += 1 - M * (x_abs_k**2 + 1/2)
#     res *= 1 / (2 * x_abs_k**2)
#     return torch.einsum('..., ...i -> ...i', res, torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new))

if __name__ == "__main__":
    x_new = torch.randn(11)
    z = torch.randn(11)**2
    M = 51
    
    score = pwrspec_score(x_new, z, M)
