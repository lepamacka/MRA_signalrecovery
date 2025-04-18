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

    if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
        k_fac = 1
    else:
        k_fac = 2

    x_Psqr_k = torch.abs(x_tilde[..., k]).square()
    xi_k = k_fac * M * torch.sqrt(rho[k] * x_Psqr_k) / (sigma ** 2)
    xi_k_cpu = xi_k.to('cpu').detach().numpy()

    fft_mat_k = fft(torch.eye(x_new.shape[-1], device=device)[:, k], norm='ortho')
    q_mat_k = torch.einsum("i, j -> ij", torch.conj(fft_mat_k), fft_mat_k) 
    sym_mat_k = (q_mat_k + q_mat_k.T).real

    div = ive(k_fac*M/2-1, xi_k_cpu)
    div_res = (ive(k_fac*M/2, xi_k_cpu[div!=0.]) + ive(k_fac*M/2-2, xi_k_cpu[div!=0.])) / (k_fac * 2. * div[div!=0.])
    res_cpu = np.inf * np.ones_like(div)
    res_cpu[div!=0.] = div_res * xi_k_cpu[div!=0.]
    res_cpu = torch.tensor(res_cpu, device=device)

    res = ((k_fac * M / 2.) - 1) * (xi_k**2) / (k_fac * 4.)
    idxs = torch.all(torch.stack((xi_k / M >= 1., torch.isfinite(res_cpu))), dim=0)
    res[idxs] = res_cpu[idxs]

    res -= M * x_Psqr_k / (sigma ** 2)
    res -= M / 2.
    res += 1. / k_fac
    res *= k_fac / (2. * x_Psqr_k)
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

    res = M * (rho[k] / (sigma ** 2) - (1. + x_Psqr_k / (sigma ** 2))) * (rho[k] / (sigma ** 2) + x_Psqr_k / (sigma ** 2)) / (2. * (1. + 2. * x_Psqr_k / (sigma ** 2))**2)
    res /= k_fac
    out = res.unsqueeze(-1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new)

    return out

# def pwrspec_score_comps_legacy(
#         x_new: TensorType[..., "L"], 
#         x_tilde: TensorType[..., "L"], 
#         z: TensorType["L"], 
#         M: int, 
#         k: int, 
#         device='cpu',
# ) -> TensorType[..., "L"]:

#     if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
#         k_fac = 1
#     else:
#         k_fac = 2

#     x_Psqr_k = torch.abs(x_tilde[..., k]).square()
#     xi_k = k_fac * torch.sqrt(M * z[k] * x_Psqr_k)
#     xi_k_cpu = xi_k.to('cpu').numpy()

#     fft_mat_k = fft(torch.eye(x_new.shape[-1], device=device)[:, k], norm='ortho')
#     q_mat_k = torch.einsum("i, j -> ij", torch.conj(fft_mat_k), fft_mat_k) 
#     sym_mat_k = (q_mat_k + q_mat_k.T).real
#     # fft_mat = fft(torch.eye(x_new.shape[-1], device=device), norm='ortho')
#     # ifft_mat = ifft(torch.eye(x_new.shape[-1], device=device), norm='ortho')
#     # delta_mat_k = torch.zeros_like(fft_mat)
#     # delta_mat_k[k, k] = 1.
#     # fft_mat_k = torch.einsum('ij, jl -> il', delta_mat_k, fft_mat)
#     # ifft_mat_k = torch.einsum('ij, jl -> il', ifft_mat, delta_mat_k)
#     # q_mat_k = torch.einsum('ij, jl -> il', ifft_mat_k, fft_mat_k)
#     # sym_mat_k = (q_mat_k + torch.conj(q_mat_k).T).real

#     div = ive(k_fac*M/2-1, xi_k_cpu)
#     div_res = (ive(k_fac*M/2, xi_k_cpu[div!=0.]) + ive(k_fac*M/2-2, xi_k_cpu[div!=0.])) / (2. * div[div!=0.])
#     res_cpu = np.inf * np.ones_like(div)
#     res_cpu[div!=0.] = div_res * xi_k_cpu[div!=0.]
#     res_cpu = torch.tensor(res_cpu, device=device)

#     res = ((k_fac * M / 2.) - 1) * xi_k**2 / 4.
#     idxs = torch.all(torch.stack((xi_k / M > 1., torch.isfinite(res_cpu))), dim=0)
#     res[idxs] = res_cpu[idxs]

#     res += 1. - k_fac * M * (x_Psqr_k + 1. / 2.)
#     res *= 1. / (2. * x_Psqr_k)
#     out = res.unsqueeze(-1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new)
#     return out

# def pwrspec_CLT_score_comps_legacy(
#         x_new: TensorType[..., "L"], 
#         x_tilde: TensorType[..., "L"], 
#         z: TensorType["L"], 
#         M: int, 
#         k: int, 
#         device='cpu',
# ) -> TensorType[..., "L"]:
#     if k == 0 or (k == x_new.shape[-1]//2 and x_new.shape[-1]%2 == 0):
#         k_fac = 1
#     else:
#         k_fac = 2

#     x_Psqr_k = torch.abs(x_tilde[..., k]).square()

#     fft_mat_k = fft(torch.eye(x_new.shape[-1], device=device)[:, k], norm='ortho')
#     q_mat_k = torch.einsum("i, j -> ij", torch.conj(fft_mat_k), fft_mat_k) 
#     sym_mat_k = (q_mat_k + torch.conj(q_mat_k).T).real

#     res = (z[k] - M * (1. + x_Psqr_k)) * (z[k]/M + x_Psqr_k) / (2. * (1. + 2. * x_Psqr_k)**2)
#     res /= k_fac
#     out = res.unsqueeze(-1) * torch.einsum('ij, ...j -> ...i', sym_mat_k, x_new)

#     return out
    
if __name__ == "__main__":
    x_new = torch.randn(11)
    z = torch.randn(11)**2
    M = 51
    
    score = pwrspec_score(x_new, z, M)
