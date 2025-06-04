import numpy as np
import torch
from torchtyping import TensorType
from torch.fft import fft, ifft, fft2, ifft2
from scipy.special import iv, ive

def pwrspec_score(
        x: TensorType[..., "L"], 
        rho: TensorType["L"],
        M: int, 
        sigma: float,
        CLT=False,
        device='cpu',
) -> TensorType[..., "L"]:
    assert rho.shape[-1] == x.shape[-1]

    x_new = x.clone().to(device)
    x_tilde = fft(x_new, norm='ortho')
    rho_true = rho.clone().to(device)

    score = torch.zeros_like(x_new)
    k = 0
    while k < x_new.shape[-1]/2 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
        if CLT:
            tmp = pwrspec_CLT_score_comps(x_new, x_tilde, rho_true, k, M, sigma, device=device)
        else:
            tmp = pwrspec_score_comps(x_new, x_tilde, rho_true, k, M, sigma, device=device)
        score += tmp
        k += 1

    return score

def pwrspec_score_comps(
        x_new: TensorType[..., "L"], 
        x_tilde: TensorType[..., "L"], 
        rho: TensorType["L"], 
        k: int, 
        M: int, 
        sigma: float,
        device='cpu',
) -> TensorType[..., "L"]:
    # Set index factor
    if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
        k_fac = 1
    else:
        k_fac = 2

    # Calculate matrix factor for gradient of power spectrum component
    fft_mat_k = fft(torch.eye(x_new.shape[-1], device=device)[:, k], norm='ortho')
    q_mat_k = torch.einsum("i, j -> ij", torch.conj(fft_mat_k), fft_mat_k) 
    sym_mat_k = (q_mat_k + q_mat_k.T).real

    # Define relevant variables
    x_Psqr_k = torch.abs(x_tilde[..., k]).square()
    xi_k = k_fac * M * torch.sqrt(rho[k] * x_Psqr_k) / (sigma ** 2)
    xi_k_cpu = xi_k.to('cpu').detach().numpy()

    # Calculate unapproximated values of Bessel terms
    div = ive(k_fac*M/2-1, xi_k_cpu)
    div_res = (ive(k_fac*M/2, xi_k_cpu[div!=0.]) + ive(k_fac*M/2-2, xi_k_cpu[div!=0.])) / (k_fac * 2. * div[div!=0.])
    res_cpu = np.inf * np.ones_like(div)
    res_cpu[div!=0.] = div_res * xi_k_cpu[div!=0.]
    res_tmp = torch.tensor(res_cpu, device=device)
    res_tmp -= M * x_Psqr_k / (sigma ** 2)
    res_tmp -= M / 2.
    res_tmp += 1. / k_fac
    res_tmp *= k_fac / (2. * x_Psqr_k)

    # Calculate approximated values of Bessel terms and replace these where appropriate
    res = torch.zeros_like(res_tmp)
    res += (M ** 2) * rho[k] / ((4 / k_fac) * (M + 2 / k_fac) * (sigma ** 4)) - M / ((2 / k_fac) * (sigma ** 2))
    idxs = torch.all(torch.stack((xi_k / M >= 1., torch.isfinite(res_tmp))), dim=0)
    res[idxs] = res_tmp[idxs]

    # Multiply result by the gradient factor
    out = res.unsqueeze(-1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new)
    return out

def pwrspec_CLT_score_comps(
        x_new: TensorType[..., "L"], 
        x_tilde: TensorType[..., "L"], 
        rho: TensorType["L"], 
        k: int, 
        M: int, 
        sigma: float,
        device='cpu',
) -> TensorType[..., "L"]:
    if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
        k_fac = 1
    else:
        k_fac = 2

    x_Psqr_k = torch.abs(x_tilde[..., k]).square()

    fft_mat_k = fft(torch.eye(x_new.shape[-1], device=device)[:, k], norm='ortho')
    q_mat_k = torch.einsum("i, j -> ij", torch.conj(fft_mat_k), fft_mat_k) 
    sym_mat_k = (q_mat_k + torch.conj(q_mat_k).T).real

    # res = M * (rho[k] / (sigma ** 2) - (1. + x_Psqr_k / (sigma ** 2))) * (rho[k] / (sigma ** 2) + x_Psqr_k / (sigma ** 2)) / (2. * (1. + 2. * x_Psqr_k / (sigma ** 2))**2)
    res = -1. * M * (x_Psqr_k * (sigma**2 + x_Psqr_k) + rho[k] * (sigma**2 - rho[k])) / (sigma**2 * (sigma**2 + 2. * x_Psqr_k)**2)
    res /= k_fac
    res += -1. / (2 * x_Psqr_k + sigma**2)
    out = res.unsqueeze(-1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new)

    return out
    
if __name__ == "__main__":
    x_new = torch.randn((11, 13))
    z = torch.randn(13) ** 2
    M = 51
    sigma = 1.
    
    score = pwrspec_score(x_new, z, M, sigma)
