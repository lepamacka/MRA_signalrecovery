import torch
import math

# Takes a (diffusion) score model and plots projected scores on a plane.
# The plane passes through plane_mag*[ones] and is normal to [ones].
# This projects the first 3 components.
@torch.no_grad()
def score_projector(
    t_diff, 
    scoremodel, 
    plane_mag, 
    ax_bound=math.sqrt(2), 
    ax_pts=10, 
    conditioner=None,
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

    XY_P = torch.einsum('ij, kli -> klj', P, XY) 
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
    S = torch.einsum('ij, klj -> kli', P, S_P)
    return S, XY, P