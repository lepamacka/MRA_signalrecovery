import torch
from torch.fft import fft, ifft
import numpy as np
import functools
import matplotlib.pyplot as plt
from torchtyping import TensorType
from circulant import circulant
from pwrspec_score import pwrspec_score
from utils import Langevin_sampler

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
    M_true = 4000
    signal_true = torch.tensor([3., 0., 0.], device=device)
    circulant_true = circulant(signal_true, dim=0)
    pwrspec_true = torch.abs(fft(signal_true, norm='ortho')).square()
    sample_pwrspec = M_true * pwrspec_true
    print(pwrspec_true)
    conditioner = functools.partial(
        pwrspec_score,
        z=sample_pwrspec,
        M=M_true,
        device=device,
        CLT=True,
    )

    # Set scoremodel parameters
    mean = torch.tensor([0., 0., 0.], device=device)
    A = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    covar_mat = torch.einsum('ij, kj -> ik', A, A)
    scoremodel = GaussianScoreModel(mean, covar_mat).to(device)
    scoremodel.eval()
    scoremodel = None

    # Set sampling parameters
    num_steps = 1000
    num_samples = 2 ** 8
    eps = 1e-3

    # Generate samples
    input = torch.randn(
        size=(num_samples, signal_true.shape[-1]), 
        device=device,
    )
    input = torch.einsum('ij, ...j -> ...i', A, input + mean)
    
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

        # theta = np.linspace(0, 2 * np.pi, 100)
        # radius = torch.sqrt(pwrspec_true[0]).item()
        # y = radius * np.sin(theta)
        # z = radius * np.cos(theta)
        # phi = -np.pi/4
        # ax.plot(y*np.cos(phi), y*np.sin(phi), z)

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
        )
        ax.scatter(
            [0.],
            [0.],
            [0.],
            c='black',
            marker = 'x',
        )
        ax.legend(["Gaussian Samples", "Conditioned Samples", "True Signal", "Origin"])

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



