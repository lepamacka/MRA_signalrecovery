import torch
from torch.fft import fft, ifft
import numpy as np
import functools
import matplotlib.pyplot as plt
from torchtyping import TensorType
from circulant import circulant
from pwrspec_score import pwrspec_score
from utils import Langevin_sampler
from scorematching.models.scoremodels import ConvScoreModel

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f"Current device is \'{device}\'.")
    
    # Set parameters for conditional sampling wrt power spectrum.
    M = 500
    signal_true = torch.tensor([4., 0., 0.], device=device)

    circulant_true = circulant(signal_true, dim=0)
    pwrspec_true = torch.abs(fft(signal_true, norm='ortho')).square()
    sample_pwrspec = M * pwrspec_true
    print(pwrspec_true)
    conditioner = functools.partial(
        pwrspec_score,
        z=sample_pwrspec,
        M=M,
        device=device,
        CLT=False,
    )

    # Set scoremodel parameters

    # # mean = torch.tensor([0., 0., 0.], device=device)
    # # A = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    # mean = torch.tensor([2., 0., 0.], device=device)
    # A = 0.5 * torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    # # mean = torch.tensor([6., 0., 0.], device=device)
    # # A = torch.tensor([[3., 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]], device=device)
    # covar_mat = torch.einsum('ij, kj -> ik', A, A)
    # scoremodel = GaussianScoreModel(mean, covar_mat).to(device)

    channels = 3
    hiddendim = 32
    PATH = f"scorematching/model_weights/MRA_convscoremodel_length{channels}_hiddim{hiddendim}/"
    scoremodel = ConvScoreModel(length=channels, hiddendim=hiddendim).to(device)
    scoremodel.load_state_dict(torch.load(PATH+"weights_dict.pth", weights_only=True))
    scoremodel.eval()
    # scoremodel = None

    # Set sampling parameters
    num_steps = 1000
    num_samples = 2 ** 8
    eps = 1e-3

    # Generate samples
    input = torch.randn(
        size=(num_samples, signal_true.shape[-1]), 
        device=device,
    )
    # input = torch.einsum('ij, ...j -> ...i', A, input)
    # input += mean
    
    output = Langevin_sampler(
        input=input,
        scoremodel=scoremodel,
        conditioner=conditioner,
        num_steps=num_steps,
        eps=eps,
        device=device,
    )

    # print(output.mean().item())

    # Plotting
    input = input.to('cpu')
    output = output.to('cpu')
    signal_true = signal_true.to('cpu')
    circulant_true = circulant_true.to('cpu')

    if signal_true.shape[0] >= 3:
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
            input[:, 0], 
            input[:, 1], 
            input[:, 2], 
            c='cornflowerblue',
            marker='.',
        )

        ax.scatter(
            output[:, 0], 
            output[:, 1], 
            output[:, 2], 
            c='forestgreen',
            marker='.',
        )

        ax.scatter(
            circulant_true[:, 0], 
            circulant_true[:, 1], 
            circulant_true[:, 2], 
            c='red',
            marker='*',
            linewidth=2.,
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
        xyz1 = xyz + signal_true.mean().item()
        xyz2 = xyz - signal_true.mean().item()
        ax.plot(xyz1[0, ...], xyz1[1, ...], xyz1[2, ...], c='red', linestyle='dotted', linewidth=0.5)
        ax.plot(xyz2[0, ...], xyz2[1, ...], xyz2[2, ...], c='red', linestyle='dotted', linewidth=0.5)

        ax.legend(["Origin", "Gaussian Samples", "Conditioned Samples", "True Signal", "Phase Manifold"])

        # Xs = torch.arange(-3, 3, 0.1)
        # Ys = torch.arange(-3, 3, 0.1)
        # X, Y = torch.meshgrid((Xs, Ys), indexing='xy')
        # Z = -(X + Y)
        # surf = ax.plot_surface(X, Y, Z, linewidth=0)

        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
    elif signal_true.shape[0] == 1:
        fig, ax = plt.subplots(1, 2, sharey=True, tight_layout=True)
        ax[0].hist(input)
        ax[1].hist(output)
        ax[0].plot(signal_true, 0*signal_true, 'd')
        ax[1].plot(signal_true, 0*signal_true, 'd')
        plt.show()



