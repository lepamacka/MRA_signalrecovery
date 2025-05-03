import torch
import numpy as np
import functools
from pwrspec_score import pwrspec_score

def experimental_conditioner(
    x_t,
    t,
    marginal_prob_std_fn,
    rho_est,
    M,
    MRA_sigma,
    even,
    use_CLT,
    device,
):
    sigma_t = marginal_prob_std_fn(t)

    rho_t = rand_pwrspec_cond(
        rho_est=rho_est, 
        M=M, 
        sigma_t=sigma_t, 
        even=even, 
        device=device,
    )

    score = pwrspec_score(
        x=x_t, 
        rho=rho_t+(MRA_sigma**2), 
        M=M, 
        sigma=torch.sqrt(MRA_sigma**2 + sigma_t**2), 
        CLT=use_CLT, 
        device=device
    )
    return score

def rand_pwrspec_cond(
    rho_est, 
    M, 
    sigma_t, 
    even,
    device,
):
    rho_t = torch.zeros_like(rho_est)
    for k, rho_k in enumerate(rho_est):
        if k == 0 or (k == rho_est.shape[-1] and even):
            k_fac = 2.
        else:
            k_fac = 1.
        mu_k = rho_k + sigma_t**2
        sigma_k = torch.sqrt((k_fac * (sigma_t**2) / M) * (2 * rho_k + sigma_t**2))
        rho_t[k] = mu_k + sigma_k * torch.randn((1,), device=device)
    rho_t = torch.abs(rho_t)
    return rho_t
