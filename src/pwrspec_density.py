import numpy as np
import torch
from torchtyping import TensorType
from scipy.special import iv, ive, loggamma

def pwrspec_density(
        rho: TensorType["L"],
        rho_x: TensorType["L"],
        sigma: float,
        num_samples: int,
        rho_tol: float = 1e-9,
        xi_tol: float = 0.1,
        device: torch.device = 'cpu',
    ): 
    if num_samples < 1:
        raise ValueError("Number of samples needs to be greater than 0.")
    if rho.shape != rho_x.shape and rho.ndim != 1:
        raise ValueError("Input tensors rho and rho_x to pwrspec_density must be one-dimensional with the same length.")
    
    xi_np = ((num_samples / (sigma ** 2)) * torch.sqrt(rho * rho_x)).numpy(force=True)
    log_res_list = []
    for k in range(rho.shape[-1]+1//2):
        if rho_x[k] > rho_tol * rho[k]:
            log_res_list.append(pwrspec_logdens_comp_nonzero(
                k=k,
                rho=rho,
                rho_x=rho_x,
                xi_np=xi_np,
                sigma=sigma,
                num_samples=num_samples,
                xi_tol=xi_tol,
                device=device,
            ))
        else:
            log_res_list.append(pwrspec_logdens_comp_zero(
                k=k,
                rho=rho,
                sigma=sigma,
                num_samples=num_samples,
                device=device,
            ))
    return torch.cat(log_res_list, dim=-1).sum(dim=-1).exp()

def pwrspec_logdens_comp_nonzero(
        k: int,
        rho: TensorType[..., "L"],
        rho_x: TensorType["L"],
        xi_np: np.ndarray,
        sigma: float,
        num_samples: int,
        xi_tol: float,
        device: torch.device = 'cpu',
    ): 
    if k == 0 or (k == rho.shape[-1]//2 and rho.shape[-1]%2 == 0):
        k_fac = 1
    else:
        k_fac = 2
    M_div = num_samples * k_fac / 2
    tmp_iv = iv(M_div-1, k_fac * xi_np[k])
    tmp_ive = ive(M_div-1, k_fac * xi_np[k])
    if xi_np[k] / num_samples > xi_tol and (tmp_iv < np.inf and tmp_iv > 0.):
        maybe_approx = np.log(iv(M_div-1, k_fac * xi_np[k]))
    elif tmp_iv == np.inf and (tmp_ive < np.inf and tmp_ive > 0.):
        maybe_approx = np.log(ive(M_div-1, k_fac * xi_np[k])) + k_fac * xi_np[k]
    # elif tmp_iv == np.inf and tmp_ive == 0.:
    #     maybe_approx = M_div * (1 + np.log(k_fac * xi_np[k] / (2 * M_div))) - np.log(np.sqrt(2. * np.pi * k_fac * M_div))
    else:
        maybe_approx = (M_div - 1) * np.log(k_fac * xi_np[k] / 2) - loggamma(M_div)
    res = (
        (M_div - 1) * torch.log(rho[k] / rho_x[k]) / 2
        - M_div * (rho[k] + rho_x[k]) / (sigma ** 2)
        + np.log(M_div) + maybe_approx
    )
    return res.unsqueeze(-1)

def pwrspec_logdens_comp_zero(
        k: int,
        rho: TensorType[..., "L"],
        sigma: float,
        num_samples: int,
        device: torch.device = 'cpu',
    ): 
    if k == 0 or (k == rho.shape[-1] // 2 and rho.shape[-1] % 2 == 0):
        k_fac = 1
    else:
        k_fac = 2

    M_div = num_samples * k_fac / 2
    res = (
        (M_div - 1) * torch.log(M_div * rho[k] / (sigma ** 2))
        - M_div * rho[k] / (sigma ** 2)
        + np.log(M_div / (sigma ** 2)) - loggamma(M_div)
    )
    return res.unsqueeze(-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")

    length = 3
    sigma = 1.0
    num_samples = 1000

    delta = 1.
    eps = 1. + sigma ** 2
    rho_x = delta * torch.ones((length,), device=device).abs()
    rho = eps * torch.randn_like(rho_x).abs() 

    print(pwrspec_density(rho, rho_x, sigma, num_samples, device=device))

    # rho = torch.linspace(1e-4, 3*(rho_x.item()+sigma**2), 100, device=device)
    # arr = torch.zeros_like(rho)
    # for idx, val in enumerate(rho):
    #     arr[idx] += pwrspec_density(val.unsqueeze(0), rho_x=rho_x, sigma=sigma, num_samples=num_samples, device=device)
    # cumsum = torch.cumsum(arr, dim=0) * (rho[1] - rho[0])

    # # plt.plot(rho.cpu(), arr.cpu())
    # plt.plot(rho.cpu(), cumsum.cpu())
    # plt.show()