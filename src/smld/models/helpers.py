import torch
import numpy as np

def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    
    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
    
    Returns:
    The standard deviation.
    """    
    t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.
    
    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
    
    Returns:
    The vector of diffusion coefficients.
    """
    return (sigma**t).to(device)