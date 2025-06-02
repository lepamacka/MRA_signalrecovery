import functools
import matplotlib.pyplot as plt
import math
import torch
from torch.fft import fft
from pwrspec_score import pwrspec_score
from triple_corr_score import triple_corr_score_3, compute_triple_corr
from scorematching.signalsamplers import Gaussian, circulant

# Takes a (diffusion) score model and plots projected scores on a plane.
# The plane passes through plane_mag*[ones] and is normal to [ones].
# This projects the first 3 components.
@torch.no_grad()
def score_projector(
    t_diff, 
    scoremodel, 
    conditioner=None,
    plane_mag=0., 
    ax_bound=math.sqrt(2), 
    ax_pts=10,
    device='cpu',
):
    if scoremodel is None and conditioner is None:
        raise ValueError
    x_pts = torch.linspace(-ax_bound, ax_bound, ax_pts)
    y_pts = torch.linspace(-ax_bound, ax_bound, ax_pts)
    P = torch.tensor(
        [
            [1./math.sqrt(2), -1./math.sqrt(2), 0.], 
            [1./math.sqrt(6), 1./math.sqrt(6), -2./math.sqrt(6)],
        ], 
        device=device,
    )

    XY = torch.stack(
        torch.meshgrid(x_pts, y_pts, indexing='ij'), 
        dim=2
    ).to(device)

    XY_P = torch.einsum('kli, ij -> klj', XY, P) 
    XY_P += plane_mag * torch.ones((3), device=device) 

    XY_P_cat = XY_P.view((XY_P.shape[0]*XY_P.shape[1], 1, XY_P.shape[2]))
    
    if t_diff > 0:
        if len(scoremodel) > 3:
            proj_diag = torch.ones(
                (XY_P_cat.shape[0], 1, len(scoremodel)-3), 
                device=device,
            )
            XY_P_cat = torch.cat(
                (XY_P_cat, plane_mag * proj_diag), 
                dim=2,
            )
        t = t_diff*torch.ones((1,), device=device)
        if scoremodel != None:
            S_P_cat = torch.vmap(scoremodel)(XY_P_cat, t=t)
            if conditioner != None:
                S_P_cat += conditioner(XY_P_cat, t)
        else:
            S_P_cat = conditioner(XY_P_cat, t)
    else:
        if scoremodel != None:
            S_P_cat = torch.vmap(scoremodel)(XY_P_cat)
            if conditioner != None:
                S_P_cat += conditioner(XY_P_cat)
        else:
            S_P_cat = conditioner(XY_P_cat)
    
    S_P = S_P_cat[:, :, :3].view(XY_P.shape)
    S = torch.einsum('klj, ij -> kli', S_P, P)
    return S, XY, P

if __name__ == "__main__":
    ### Set pytorch parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Current device is \'{device}\'.")

    ### Set parameters for MRA
    M = 100000000
    sigma = 10.

    ### Set parameters for true signal
    length = 3
    signal_true = torch.zeros((length,), device=device, requires_grad=False)
    signal_true[0] = 1.

    ### Set parameters for conditioner
    conditioner_type = "triple correlation" # "power spectrum", "triple correlation"
    use_random_statistic = True
    use_CLT = True

    ### Set parameters for visualization of scores
    ax_bound = 1.
    ax_pts = 20
    plane_mag = signal_true.mean().item()

    ## Conditioner
    circulant_true = circulant(signal_true, dim=0)
    pwrspec_true = torch.abs(fft(signal_true, norm='ortho')).square()
    triple_corr_true = compute_triple_corr(signal_true, device=device)

    MRA_sampler = Gaussian(
        sigma=sigma, 
        signal=signal_true, 
        length=length, 
        device=device,
    )
    samples = MRA_sampler(num=M, do_random_shifts=True)
    if use_random_statistic:
        sample_pwrspec = torch.abs(fft(samples, norm='ortho')).square().mean(dim=0)
        sample_triple_corr = compute_triple_corr(samples, average=True, device=device)
    else:
        sample_pwrspec = pwrspec_true + sigma ** 2
        sample_triple_corr = torch.zeros(size=(3, 3), device=device)
        sample_triple_corr[0, 0] += 1.
        sample_triple_corr[0, :] += 1.
        sample_triple_corr[:, 0] += 1.
        sample_triple_corr[1, 2] += 1.
        sample_triple_corr[2, 1] += 1.
        sample_triple_corr *= sigma ** 2 * torch.mean(signal_true)
        sample_triple_corr += triple_corr_true

    if conditioner_type == "power spectrum":
        conditioner = functools.partial(
            pwrspec_score,
            rho=sample_pwrspec,
            M=M,
            sigma=sigma,
            device=device,
            CLT=use_CLT,
        )
    elif conditioner_type == "triple correlation":
        if length == 3:
            conditioner = functools.partial(
                triple_corr_score_3,
                triple_corr=sample_triple_corr,
                M=M,
                sigma=sigma,
                device=device,
                CLT=use_CLT,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    ## Plotting

    if signal_true.shape[0] == 3:
        S, XY, P = score_projector(
            t_diff=0,
            scoremodel=None, 
            conditioner=conditioner,
            plane_mag=plane_mag, 
            ax_bound=ax_bound,
            ax_pts=ax_pts,
            device=device,
        )
        XY = XY.to('cpu')
        S /= torch.norm(S, dim=-1, keepdim=True) ** (2. / 3.)
        S = S.to('cpu')

        fig, ax = plt.subplots()

        Q = ax.quiver(
            XY[:, :, 0], 
            XY[:, :, 1], 
            S[:, :, 0], 
            S[:, :, 1],
        )
        ax.set_aspect('equal', 'box')
        centers = plt.scatter(
            (P @ circulant(signal_true[:3], 0)).to('cpu')[0, :], 
            (P @ circulant(signal_true[:3], 0)).to('cpu')[1, :],
            c='red',
        ) 

        plt.title(f"Projection of conditional 3D-scores wrt {conditioner_type}")
        fig.tight_layout()
        plt.savefig('./../figs/scorematching/conditionalscores.png')

    plt.show()


