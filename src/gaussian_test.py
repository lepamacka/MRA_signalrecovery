import torch
from torch.fft import fft, ifft
import numpy as np
import functools
import matplotlib.pyplot as plt
from circulant import circulant
from pwrspec_score import pwrspec_score
from signalsamplers import Gaussian
from utils import Langevin_sampler

if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Current device is \'{device}\'.")
    
    # Set parameters for conditional sampling wrt power spectrum.
    M = 20
    signal_true = torch.tensor([1., 0., 0.], dtype=torch.complex64)
    circulant_true = circulant(signal_true, dim=0)
    sample_pwrspec = torch.abs(fft(signal_true, norm='ortho'))**2
    print(sample_pwrspec)
    conditioner = functools.partial(
        pwrspec_score,
        z=sample_pwrspec,
        M=M,
        device=device,
    )

    # A simple score model, standard gaussian centered at 0.
    class GaussianScoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return -x
    
    scoremodel = GaussianScoreModel()
            

    # Set sampling parameters
    num_steps = 1000
    num_samples = 50

    # Generate samples
    init_samples = torch.randn(size=(num_samples, 3))
    input = init_samples.clone()
    output = Langevin_sampler(
        input=input,
        scoremodel=scoremodel,
        conditioner=conditioner,
        n_steps=num_steps,
    )

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        init_samples[:, 0], 
        init_samples[:, 1], 
        init_samples[:, 2], 
        c='b',
    )
    ax.scatter(
        output[:, 0], 
        output[:, 1], 
        output[:, 2], 
        c='r',
    )
    Xs = torch.arange(-3, 3, 0.1)
    Ys = torch.arange(-3, 3, 0.1)
    X, Y = torch.meshgrid((Xs, Ys), indexing='xy')
    Z = -(X + Y)
    surf = ax.plot_surface(X, Y, Z, linewidth=0)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()



