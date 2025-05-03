import torch
import torch.nn as nn
import numpy as np
import functools
from . import helpers

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(
        self, 
        embed_dim, 
        scale=30.,
    ):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class BatchConvolver1D(nn.Module):
    """Convolves batched input vectors [B, in_dim, length] with kernels from a 
    differentiable linear transform of a second input vector [B, kernel_indim].
    Output is [B, out_dim, length]."""
    def __init__(
        self, 
        length, 
        in_dim, 
        out_dim, 
        kernel_indim,
    ):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.length = length
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.circpad = nn.CircularPad1d((length-1)//2)
        self.kerneldense = nn.Linear(kernel_indim, out_dim*in_dim*length) # Might need more juice and/or nonlinearity.
        
    def forward(self, h, ht):
        kernels = self.kerneldense(ht).view(-1, self.out_dim, self.in_dim, self.length)
        h_pad = self.circpad(h).unsqueeze(1)
        output = torch.vmap(nn.functional.conv1d)(h_pad, kernels).squeeze(1)
        return output

# Diffusion model for multireference alignment, equivariant with respect to shifts.
class Convolutional(nn.Module):
    def __init__(
        self, 
        marginal_prob_std, 
        length, 
        hidden_dim, 
        hidden_layers=2, 
        embed_dim=32
    ):
        super().__init__()
        
        assert length%2 == 1, "Can only handle odd channel numbers due to padding peculiarities."
        assert length >= 3, "Signal length must be greater than or equal to 3."

        self.length = length
        self.marginal_prob_std = marginal_prob_std
        self.hidden_layers = hidden_layers
        
        kernel_indim = hidden_dim*length

        self.act = nn.ReLU()
        self.embed = GaussianFourierProjection(embed_dim)
        self.emb_dense1 = nn.Linear(embed_dim, embed_dim)
        self.emb_dense2 = nn.Linear(embed_dim, embed_dim)

        self.dense_alpha = nn.Linear(embed_dim, kernel_indim)
        self.batchconv_alpha = BatchConvolver1D(length, 1, hidden_dim, kernel_indim)
        
        self.dense_list = nn.ModuleList([])
        self.batchconv_list = nn.ModuleList([])
        for idx in range(self.hidden_layers):
            self.dense_list.append(nn.Linear(embed_dim, kernel_indim))
            self.batchconv_list.append(BatchConvolver1D(length, hidden_dim, hidden_dim, kernel_indim))

        self.dense_omega = nn.Linear(embed_dim, kernel_indim)
        self.batchconv_omega = BatchConvolver1D(length, hidden_dim, 1, kernel_indim)
    
    def forward(self, x, t):
        h = x.unsqueeze(1)

        emb_t = self.embed(t)
        emb_t = self.emb_dense1(emb_t)
        emb_t = self.act(emb_t)
        emb_t = self.emb_dense2(emb_t)
        emb_t = self.act(emb_t)
        
        ht = self.dense_alpha(emb_t)
        ht = self.act(ht)
        h = self.batchconv_alpha(h, ht)
        h = self.act(h)

        for idx in range(self.hidden_layers):
            ht = self.dense_list[idx](emb_t)
            ht = self.act(ht)
            h = self.batchconv_list[idx](h, ht)
            h = self.act(h)

        ht = self.dense_omega(emb_t)
        ht = self.act(ht)
        h = self.batchconv_omega(h, ht)

        score = h.squeeze(1) / self.marginal_prob_std(t)[:, None]
        return score

    def __str__(self):
        return 'conv'

    def __len__(self):
        return self.length

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 

    batch_size = 10
    length = 3
    hidden_dim = 16
    embed_dim = 16
    model_sigma = 5.0

    marginal_prob_std_fn = functools.partial(
        helpers.marginal_prob_std, 
        sigma=model_sigma, 
        device=device
    )
    model = Convolutional(
        marginal_prob_std=marginal_prob_std_fn,
        length=length, 
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    ).to(device)

    x = torch.randn(
        batch_size, 
        length, 
        device=device, 
        generator=generator
    )
    random_t = torch.rand(x.shape[0], device=device, generator=generator) * (1. - 1e-4) + 1e-4  
    z = torch.randn_like(x, device=device)
    std = marginal_prob_std_fn(random_t)
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, random_t)
    print(score)
