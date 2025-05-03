import os
import torch
from torch.fft import fft, ifft
import numpy as np
import functools
import matplotlib.pyplot as plt
from torchtyping import TensorType
from scorematching.signalsamplers import circulant
from pwrspec_score import pwrspec_score
from scorematching.utils import Langevin_sampler
from scorematching.models.scoremodels import ConvScoreModel
from scorematching.signalsamplers import Gaussian

# A simple score model, standard gaussian centered at 0.
class GaussianScoreModel(torch.nn.Module):
    def __init__(
        self, 
        mean: TensorType["L"],
        covar_mat: TensorType["L", "L"], 
    ):
        super().__init__()

        assert torch.allclose(covar_mat, covar_mat.T), "The covariance matrix must be symmetric."

        self.register_buffer(
            "prec_mat",
            torch.inverse(covar_mat),
        )
        self.register_buffer(
            "mean",
            mean,
        )
    
    def forward(
        self, 
        x: TensorType[..., "L"],
    ) -> TensorType[..., "L"]:
        return torch.einsum('ij, ...j -> ...i', self.prec_mat, self.mean - x)

if __name__ == "__main__":
    
    ## Set parameters

    ### Set pytorch parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Current device is \'{device}\'.")

    ### Set parameters for MRA
    M = 100
    sigma = 1.

    ### Set parameters for true signal
    length = 3
    signal_true = torch.zeros((length,), device=device, requires_grad=False)
    # signal_true[0] = np.sqrt(length)
    signal_true[0] = 2.

    # signal_true = 1. * torch.ones((length,), device=device, requires_grad=False)
    # signal_true += 0.5 * torch.randn_like(signal_true)

    ### Set parameters for conditioner
    use_random_pwrspec = False
    use_CLT = True
    use_none_cond = False

    ### Set parameters for score model
    scoremodel_type = "gaussian" # "gaussian", "learned", "none"
    
    ### Set parameters for gaussian score model
    mean = torch.zeros_like(signal_true)
    A = torch.eye(n=length, device=device)
    # mean = torch.tensor([0., 0., 0.], device=device)
    # A = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    # mean = torch.tensor([2., 0., 0.], device=device)
    # A = 0.5 * torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    # mean = torch.tensor([6., 0., 0.], device=device)
    # A = torch.tensor([[3., 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]], device=device)

    ### Set parameters for learned score model
    PATH = None # If None it will try to find a compatible model.
    hiddendim = 8

    ### Set parameters for Langevin sampling
    num_steps = 1000
    num_samples = 2 ** 9
    eps = 1e-3

    ### Set parameters for plotting
    plot_output_only = True
    plot_input = False
    plot_MRA_samples = False


    ## Conditioner
    circulant_true = circulant(signal_true, dim=0)
    pwrspec_true = torch.abs(fft(signal_true, norm='ortho')).square()

    MRA_sampler = Gaussian(
        sigma=sigma, 
        signal=signal_true, 
        length=length, 
        device=device,
    )
    samples = MRA_sampler(num=M, do_random_shifts=True)

    if use_random_pwrspec:
        sample_pwrspec = torch.abs(fft(samples, norm='ortho')).square().mean(dim=0)
    else:
        sample_pwrspec = pwrspec_true + sigma ** 2
    
    if use_none_cond:
        conditioner = None
    else:
        conditioner = functools.partial(
            pwrspec_score,
            rho=sample_pwrspec,
            M=M,
            sigma=sigma,
            device=device,
            CLT=use_CLT,
        )
    

    ## Score model
    if scoremodel_type == "gaussian":
        covar_mat = torch.einsum('ij, kj -> ik', A, A)
        scoremodel = GaussianScoreModel(mean, covar_mat).to(device)
    elif scoremodel_type == "learned":
        if PATH is None:
            PATH = f"./../model_weights/scorematching/MRA_convscoremodel_length{length}_hiddim{hiddendim}/"
            PATH = os.path.join(PATH, "weights_dict.pth")
        scoremodel = ConvScoreModel(length=length, hiddendim=hiddendim).to(device)
        state_dict = torch.load(PATH, weights_only=True, map_location=device)
        scoremodel.load_state_dict(state_dict)
        scoremodel.eval()
    elif scoremodel_type == "none":
        scoremodel = None
    else:
        raise ValueError("Unimplemented scoremodel_type")


    ## Generate input samples
    input = signal_true.std()*torch.randn(
        size=(num_samples, signal_true.shape[-1]), 
        device=device,
        requires_grad=False,
    )
    # input = torch.einsum('ij, ...j -> ...i', A, input)
    # input += mean
    

    ## Run Langevin sampling
    output = Langevin_sampler(
        input=input,
        scoremodel=scoremodel,
        conditioner=conditioner,
        num_steps=num_steps,
        eps=eps,
        device=device,
    )

    if torch.any(torch.isfinite(output.reshape((-1))).logical_not()):
        print("Some of the output is nan or inf.")

    ## Print comparison of input and output
    inputs_pwrspec_rmsd = (torch.abs(fft(input, norm='ortho')).square() - pwrspec_true).square().mean(dim=-1).sqrt()
    outputs_pwrspec_rmsd = (torch.abs(fft(output, norm='ortho')).square() - pwrspec_true).square().mean(dim=-1).sqrt()

    print(f"Average RMSD between input power spectra and the true power spectrum is: {inputs_pwrspec_rmsd.mean().item():.2f}")
    print(f"Average RMSD between output power spectra and the true power spectrum is: {outputs_pwrspec_rmsd.mean().item():.2f}")

    inputs_rmsd = (input.unsqueeze(1) - circulant_true).square().mean(dim=-1).sqrt().min(dim=-1)[0]
    outputs_rmsd = (output.unsqueeze(1) - circulant_true).square().mean(dim=-1).sqrt().min(dim=-1)[0]

    print(f"Average RMSD between input samples and the true signal is: {inputs_rmsd.mean().item():.2f}")
    print(f"Average RMSD between output samples and the true signal is: {outputs_rmsd.mean().item():.2f}")


    ## Plot results
    samples = samples.detach().to('cpu')
    input = input.detach().to('cpu')
    output = output.detach().to('cpu')
    signal_true = signal_true.detach().to('cpu')
    circulant_true = circulant_true.detach().to('cpu')

    if plot_MRA_samples and signal_true.shape[0] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            [0.],
            [0.],
            [0.],
            c='black',
            marker = 'x',
        )
        ax.scatter(
            circulant_true[0:3, 0], 
            circulant_true[0:3, 1], 
            circulant_true[0:3, 2], 
            c='red',
            marker='*',
            linewidth=4.,
        )
        ax.scatter(
            samples[:, 0], 
            samples[:, 1], 
            samples[:, 2], 
            c='cornflowerblue',
            marker='.',
        )
        ax.legend(["Origin", "True Signal", "MRA Samples"])  
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.view_init(elev=35, azim=-45, roll=0)

        # plt.savefig("./../figs/scorematching/moment_likelihood/MRA_samples.svg")
        # plt.savefig("./../figs/scorematching/moment_likelihood/MRA_samples.png")

        plt.show()
    elif signal_true.shape[0] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            [0.],
            [0.],
            [0.],
            c='black',
            marker = 'x',
        )
        ax.scatter(
            output[:, 0], 
            output[:, 1], 
            output[:, 2], 
            c='forestgreen',
            marker='.',
        )
        if use_none_cond:
            output_samples_tag = "Prior Samples"
        else:
            output_samples_tag = "Posterior Samples"
        if not plot_output_only:
            if plot_input:
                ax.scatter(
                    input[:, 0], 
                    input[:, 1], 
                    input[:, 2], 
                    c='cornflowerblue',
                    marker='.',
                )
            if signal_true.shape[0] == 3:
                ax.scatter(
                    circulant_true[0:3, 0], 
                    circulant_true[0:3, 1], 
                    circulant_true[0:3, 2], 
                    c='red',
                    marker='*',
                    linewidth=4.,
                )
                theta = np.linspace(0, 2 * np.pi, 100)
                radius = torch.norm(signal_true - signal_true.mean()).item()
                phi = -np.pi/4
                xyz = np.stack(
                    [
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.cos(theta),
                    ],
                    axis=0,
                ) # Start with a circle around the x-axis (theta), rotate to circle around [1, 1, 0] (phi).
                rotmat = np.array(
                    [
                        [0.9082704, -0.0917296, -0.4082040],
                        [-0.0917296, 0.9082704, -0.4082040],
                        [0.4082040, 0.4082040, 0.8165408],
                    ]
                ) # Rotation by 35.26 degrees to go from circle around [1, 1, 0] to circle around [1, 1, 1].
                xyz = rotmat @ xyz
                xyz1 = xyz + signal_true[0:3].mean().item()
                xyz2 = xyz - signal_true[0:3].mean().item()
                ax.plot(xyz1[0, ...], xyz1[1, ...], xyz1[2, ...], c='red', linestyle='dotted', linewidth=1.5)
                ax.plot(xyz2[0, ...], xyz2[1, ...], xyz2[2, ...], c='red', linestyle='dotted', linewidth=1.5)
                if plot_input:
                    ax.legend(["Origin", output_samples_tag, "Gaussian Samples", "True Signal", "Phase Manifold"])  
                else:
                    ax.legend(["Origin", output_samples_tag, "True Signal", "Phase Manifold"])  
            else:
                if plot_input:
                    ax.legend(["Origin", output_samples_tag, "Gaussian Samples"])
                else:
                    ax.legend(["Origin", output_samples_tag])
        else:
            ax.legend(["Origin", output_samples_tag])
        # Xs = torch.arange(-3, 3, 0.1)
        # Ys = torch.arange(-3, 3, 0.1)
        # X, Y = torch.meshgrid((Xs, Ys), indexing='xy')
        # Z = -(X + Y)
        # surf = ax.plot_surface(X, Y, Z, linewidth=0)

        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.view_init(elev=35, azim=-45, roll=0)

        # plt.savefig("./../figs/scorematching/moment_likelihood/fig.svg")

        plt.show()
    elif signal_true.shape[0] == 1:
        fig, ax = plt.subplots(1, 2, sharey=True, tight_layout=True)
        ax[0].hist(input)
        ax[1].hist(output)
        ax[0].plot(signal_true, 0*signal_true, 'd')
        ax[1].plot(signal_true, 0*signal_true, 'd')
        plt.show()



