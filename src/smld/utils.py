import torch
import numpy as np

def Euler_Maruyama_sampler(
    scoremodel, 
    marginal_prob_std,
    diffusion_coeff, 
    length,
    batch_size=64, 
    num_steps=500,
    eps=1e-3, 
    conditioner=None,
    cond_start_frac=0.,
    verbose=False,
    device='cuda', 
):
    """Generate samples from diffusion models with the Euler-Maruyama solver.

    Args:
    scoremodel: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samples to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, length, device=device) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for step, time_step in enumerate(time_steps):
            if verbose and step % int(num_steps / 100) == 0:
                print(f"Diffusion step: {step:>6}/{num_steps}")  
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None] * scoremodel(x, batch_time_step) * step_size
            if conditioner is not None and step / num_steps >= cond_start_frac:
                mean_x += (g**2)[:, None] * conditioner(x, time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)   
    return mean_x