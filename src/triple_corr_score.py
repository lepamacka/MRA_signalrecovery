import numpy as np
import torch
from torchtyping import TensorType
from scorematching.signalsamplers import circulant

def triple_corr_score_3(
        x: TensorType[..., 3], 
        triple_corr: TensorType[3, 3],
        M: int, 
        sigma: float,
        CLT=True,
        device='cpu',
) -> TensorType[..., "L"]:
    x_new = x.clone().to(device)    
    proj_op = torch.tensor(
        [
            [3., 1., 1., 1., 1., 1., 1., 0., 0.],
            [-2., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., -1., -1., -1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1.],
        ],
        device=device,
    )
    triple_corr_unfolded = unfold_3(triple_corr, device=device)
    triple_corr_proj = torch.einsum('ij, j -> i', proj_op, triple_corr_unfolded)

    covar = torch.tensor(
        [
            [15.,  3.,  3.,  3.,  3.,  3.,  3.,  0.,  0.],
            [3.,  3.,  3.,  3.,  1.,  1.,  1.,  0.,  0.],
            [3.,  3.,  3.,  3.,  1.,  1.,  1.,  0.,  0.],
            [3.,  3.,  3.,  3.,  1.,  1.,  1.,  0.,  0.],
            [3.,  1.,  1.,  1.,  3.,  3.,  3.,  0.,  0.],
            [3.,  1.,  1.,  1.,  3.,  3.,  3.,  0.,  0.],
            [3.,  1.,  1.,  1.,  3.,  3.,  3.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  3.],
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  3.],
        ],
        device=device,
    )
    covar *= (sigma ** 6) / (3 * M)
    covar_proj = torch.einsum('ik, kl -> il', proj_op, torch.einsum('ij, kj -> ik', covar, proj_op))
    covar_proj_inv = torch.inverse(covar_proj)
    
    mu_init = torch.zeros(size=(3, 3), device=device)
    mu_init[0, 0] += 1.
    mu_init[0, :] += 1.
    mu_init[:, 0] += 1.
    mu_init[1, 2] += 1.
    mu_init[2, 1] += 1.
    mu_init *= sigma ** 2
    mu_base = torch.einsum('..., ij -> ...ij', torch.sum(x_new, dim=-1), mu_init)
    mu_base += compute_triple_corr(x_new, average=False, device=device)
    mu_base_unfolded = unfold_3(mu_base, device=device)
    mu_base_proj = torch.einsum('ij, ...j -> ...i', proj_op, mu_base_unfolded)

    d_logdens_d_mu_proj = torch.einsum('ij, ...j -> ...i', covar_proj_inv, triple_corr_proj - mu_base_proj)

    score = torch.zeros_like(x_new)
    for k in range(3):
        if CLT:
            d_mu_d_xk = torch.zeros_like(mu_base)
            d_mu_d_xk += mu_init
            for dim_1 in range(3):
                for dim_2 in range(3):
                    d_mu_d_xk[..., dim_1, dim_2] += (
                        x_new[..., (k-dim_1)%3] * x_new[..., (k+dim_2)%3]
                        + x_new[..., (k+dim_1)%3] * x_new[..., (k+dim_1+dim_2)%3]
                        + x_new[..., (k-dim_2)%3] * x_new[..., (k-dim_1-dim_2)%3]
                    )
            d_mu_d_xk_unfolded = unfold_3(d_mu_d_xk, device=device)
            d_mu_d_xk_proj = torch.einsum('ij, ...j -> ...i', proj_op, d_mu_d_xk_unfolded)
            score[..., k] += torch.einsum('...i, ...i -> ...', d_logdens_d_mu_proj, d_mu_d_xk_proj)
        else:
            raise NotImplementedError
    return score

def unfold_3(input: TensorType[..., 3, 3], device):
    assert input.shape[-2] == 3 and input.shape[-1] == 3

    input_unfolded = torch.zeros(size=input.shape[:-2]+(9,), device=device)
    for n_l in range(3):
        for m_l in range(3):
            if (n_l + m_l) % 3 == 0 or (n_l == 0 or m_l == 0):
                if n_l == 0 and m_l == 0:
                    idx_l = 0
                elif n_l == 0:
                    idx_l = 3 * (m_l - 1) + 1
                elif m_l == 0:
                    idx_l = 3 * (3 - n_l - 1) + 2
                else:
                    idx_l = 3 * (n_l - 1) + 3
            elif n_l + m_l > 3:
                idx_l = 3 * 3 - 2 + (n_l - 1) * (3 - 2) + m_l - 2
            else: 
                idx_l = 3 * 3 - 2 + (n_l - 1) * (3 - 2) + m_l - 1
            input_unfolded[..., idx_l] = input[..., n_l, m_l]
    return input_unfolded

def compute_triple_corr(data, average=True, device='cpu'):
    length = data.shape[-1]
    data_circulant = circulant(data, dim=-1)
    if average:
        triple_corr = torch.zeros(size=(length, length), device=device)
    else:
        triple_corr = torch.zeros(size=data.shape[:-1]+(length, length), device=device)
    for dim_1 in range(length):
        for dim_2 in range(length):
            tmp = torch.mean(
                torch.einsum(
                    '...i, ...i -> ...i',
                    torch.einsum(
                        '...i, ...i -> ...i',
                        data_circulant[..., -dim_1, :],
                        data_circulant[..., dim_2, :],
                    ),
                    data_circulant[..., 0, :],
                ),
                dim=-1,
            )
            # print("here")
            # print(torch.einsum(
            #         '...i, ...i -> ...i',
            #         data_circulant[..., -dim_1, :],
            #         data_circulant[..., dim_2, :],
            #     ))
            # print(data_circulant[..., 0, :])
            # print(tmp)
            if average:
                triple_corr[dim_1, dim_2] = torch.mean(tmp)
            else:
                triple_corr[..., dim_1, dim_2] = tmp
    return triple_corr

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M = 100
    num_score_samples = 3
    sigma = 10.
    signal_true = torch.tensor([1., 0., 0.], device=device)
    data = signal_true + sigma * torch.randn(M, 3, device=device)

    # print(compute_triple_corr(signal_true, device=device))

    triple_corr = compute_triple_corr(data, device=device)
    print(triple_corr)
    input = signal_true + 0.1 * torch.randn(num_score_samples, 3, device=device)
    score = triple_corr_score_3(
        x=input,
        triple_corr=triple_corr,
        M=M,
        sigma=sigma,
    )
    print(score)

