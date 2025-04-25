import torch
import numpy as np
import math
import functools
import matplotlib.pyplot as plt
from matplotlib import animation
from smld.models.helpers import marginal_prob_std, diffusion_coeff
from smld.signalsamplers import circulant
from pwrspec_score import pwrspec_score
import smld.signalsamplers as samplers
from smld.models.convolutional import Convolutional
from smld.models.MLP import MLP
from smld.utils import Euler_Maruyama_sampler

# Takes a (diffusion) score model and plots projected scores on a plane.
# The plane passes through plane_mag*[ones] and is normal to [ones].
# This projects the first 3 components.
@torch.no_grad()
def score_projector(
    t_diff, 
    score_model, 
    plane_mag, 
    ax_bound=math.sqrt(2), 
    ax_pts=10, 
    conditioner=None,
    device='cpu',
):
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

    if len(score_model) > 3:
        proj_diag = torch.ones(
            (XY_P_cat.shape[0], 1, len(score_model)-3), 
            device=device
        )
        XY_P_cat = torch.cat(
            (XY_P_cat, plane_mag * proj_diag), 
            dim=2,
        )
    
    S_P_cat = torch.vmap(score_model)(
        XY_P_cat, 
        t=t_diff*torch.ones((1,), device=device),
    )
    print(S_P_cat.mean())
    if conditioner != None:
        S_P_cat += conditioner(XY_P_cat)
    print(S_P_cat.mean())
    S_P = S_P_cat[:, :, :3].view(XY_P.shape)
    S = torch.einsum('ij, klj -> kli', P, S_P)
    return S, XY, P

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 
    generator.seed()

    # Choose a model class to be loaded and corresponding base signal.
    
    # Set diffusion model parameters of trained model.
    hidden_layers = 5
    hidden_dim = 32
    embed_dim = 32
    model_sigma = 2.5

    # Set signal parameters, same as those used to train model.
    length = 3
    signal_sampler_str = "loop"
    signal_scale = .3

    # Set model type, signal sampler type.
    # Trained model must exist on PATH.
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, 
        sigma=model_sigma, 
        device=device,
    )
    score_model = Convolutional(
        marginal_prob_std_fn, 
        length, 
        hidden_dim, 
        hidden_layers, 
        embed_dim,
    ).to(device)

    model_path_name = "_".join((
        f"{str(score_model)}",
        f"len{length}",
        f"lay{hidden_layers}",
        f"hid{hidden_dim}",
        f"emb{embed_dim}",
        f"sigma{model_sigma}",
        f"{signal_sampler_str}{signal_scale}",
    ))
    PATH = "./../model_weights/smld/" + model_path_name + "/"
    print(PATH)
    signal = torch.load(PATH+"signal.pth", weights_only=True).to(device)
    signal_sampler = samplers.DegenerateLoop(
        signal_scale, 
        signal, 
        length, 
        generator=generator, 
        device=device,
    )
    score_model.load_state_dict(torch.load(PATH+"weights_dict.pth", weights_only=True))
    score_model.eval()

    # Set parameters for diffusion sampler.
    diffusion_steps = 1000
    diffusion_samples = 1000
    diffusion_epsilon = 1e-3
    model_sampler = Euler_Maruyama_sampler

    # Set parameters for conditional diffusion sampling.
    M = 10
    signal_true = circulant(signal_sampler(do_random_shifts=False), dim = 1).squeeze(0)
    signal_for_pwrspec = signal_true[0, :]
    signal_for_pwrspec += 1e-2*torch.randn(
        signal.shape,
        generator=generator,
        device=device,
    )
    sample_pwrspec = torch.abs(torch.fft.fft(signal_for_pwrspec))**2
    conditioner = functools.partial(
        pwrspec_score,
        z=sample_pwrspec,
        M=M,
        device=device,
    )
    conditioner = None

    # Set parameters for visualization of scores across diffusion time.
    num_steps = 200
    t_diff_init = 1e-3
    ax_bound = 2
    ax_pts = 20
    plane_mag = signal.mean().item()

    # Generate diffusion samples using the score model.
    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, 
        sigma=model_sigma, 
        device=device,
    )
    model_samples = model_sampler(
        score_model=score_model, 
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn, 
        length=length,
        batch_size=diffusion_samples,
        num_steps=diffusion_steps,
        eps=diffusion_epsilon,
        conditioner=conditioner,
        device=device,
    ).squeeze()

    # Generate signal samples
    signal_samples = signal_sampler(num=diffusion_samples, do_random_shifts=True)

    # Plot samples from diffusion sampler. ONLY FOR length = 3.
    if length == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            model_samples[:,0].to('cpu'), 
            model_samples[:,1].to('cpu'), 
            model_samples[:,2].to('cpu'), 
            c='b',
            label='model samples',
        )
        if conditioner != None:
            ax.scatter(
                signal_true[:,0].to('cpu'), 
                signal_true[:,1].to('cpu'), 
                signal_true[:,2].to('cpu'), 
                c='r',
                label='true signal',
            )
            ax.legend(['model samples', 'true_signal'])
        else:
            ax.legend(['model samples'])
        ax.view_init(elev=35, azim=45, roll=0)
        plt.savefig('./../figs/smld/model_samples_3d.png')
        ax.view_init(elev=90, azim=0, roll=0)
        plt.savefig('./../figs/smld/model_samples_top.png')

    # Plot samples from signal sampler. ONLY FOR length = 3.
    if length == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            signal_samples[:,0].to('cpu'), 
            signal_samples[:,1].to('cpu'), 
            signal_samples[:,2].to('cpu'), 
            c='g',
            label='signal samples',
        )
        if conditioner != None:
            ax.scatter(
                signal_true[:, 0].to('cpu'), 
                signal_true[:, 1].to('cpu'), 
                signal_true[:, 2].to('cpu'), 
                c='r',
                label='true signal',
            )
            ax.legend(['signal samples', 'true_signal'])
        else:
            ax.legend(['signal samples'])
        
        ax.view_init(elev=35, azim=45, roll=0)
        plt.savefig('./../figs/smld/signal_samples_3d.png')
        ax.view_init(elev=90, azim=0, roll=0)
        plt.savefig('./../figs/smld/signal_samples_top.png')

    # Plot samples from both samplers. FOR length = 3.
    if length == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            signal_samples[:,0].to('cpu'), 
            signal_samples[:,1].to('cpu'), 
            signal_samples[:,2].to('cpu'), 
            c='g', 
            label='signal samples',
        )
        ax.scatter(
            model_samples[:,0].to('cpu'), 
            model_samples[:,1].to('cpu'), 
            model_samples[:,2].to('cpu'), 
            c='b', 
            label='model samples',
        )
        if conditioner != None:
            ax.scatter(
                signal_true[:,0].to('cpu'), 
                signal_true[:,1].to('cpu'), 
                signal_true[:,2].to('cpu'), 
                c='r',
                label='true signal',
            )
            ax.legend(['signal samples', 'model samples', 'true_signal'])
        else:
            ax.legend(['signal samples', 'model samples'])
        ax.view_init(elev=35, azim=45, roll=0)
        plt.savefig('./../figs/smld/all_samples_3d.png')
        ax.view_init(elev=90, azim=0, roll=0)
        plt.savefig('./../figs/smld/all_samples_top.png')

    F1 = functools.partial(
        score_projector, 
        score_model=score_model, 
        conditioner=None,
        plane_mag=plane_mag, 
        ax_bound=ax_bound,
        ax_pts=ax_pts,
        device=device,
    )

    S1, XY1, P1 = F1(t_diff_init)
    XY1 = XY1.to('cpu')
    S1 = S1.to('cpu')

    fig, ax = plt.subplots()
    Q1 = ax.quiver(
        XY1[:, :, 0], 
        XY1[:, :, 1], 
        S1[:, :, 0], 
        S1[:, :, 1],
    )
    ax.set_aspect('equal', 'box')
    centers = plt.scatter(
        (P1 @ circulant(signal[:3], 0)).to('cpu')[0, :], 
        (P1 @ circulant(signal[:3], 0)).to('cpu')[1, :],
    ) 

    F2 = functools.partial(
        score_projector, 
        score_model=score_model, 
        conditioner=conditioner,
        plane_mag=plane_mag, 
        ax_bound=ax_bound,
        ax_pts=ax_pts,
        device=device,
    )

    plt.title(f"Projection of 3D-scores at diffusion time {t_diff_init}")
    fig.tight_layout()
    plt.savefig('./../figs/smld/projectedscores.png')

    S2, XY2, P2 = F2(t_diff_init)
    XY2 = XY2.to('cpu')
    S2 = S2.to('cpu')

    fig, ax = plt.subplots()

    Q2 = ax.quiver(
        XY2[:, :, 0], 
        XY2[:, :, 1], 
        S2[:, :, 0], 
        S2[:, :, 1],
    )
    ax.set_aspect('equal', 'box')
    centers = plt.scatter(
        (P2 @ circulant(signal[:3], 0)).to('cpu')[0, :], 
        (P2 @ circulant(signal[:3], 0)).to('cpu')[1, :],
    ) 

    plt.title(f"Projection of conditional 3D-scores at diffusion time {t_diff_init}")
    fig.tight_layout()
    plt.savefig('./../figs/smld/conditionalscores.png')

    plt.show()

    # def update_quiver(n, T, Q, X, Y, F):
    #     S_n, _, _ = F(T[n-1])
    #     U_n = S_n[:, :, 0].to('cpu')
    #     V_n = S_n[:, :, 1].to('cpu')
    #     Q.set_UVC(U_n,V_n)
    #     return Q

    # timesteps = torch.linspace(1., t_diff_init, num_steps)
    # anim = animation.FuncAnimation(
    #     fig, 
    #     update_quiver, 
    #     fargs=(timesteps, Q, XY[:, :, 0], XY[:, :, 1], F), 
    #     frames=num_steps, 
    #     interval=100,
    #     blit=False,
    # )

    # plt.title(f"Projection of 3D-scores for diffusion time")
    # anim.save('./../figs/smld/diffusion_scores.mp4')
    
