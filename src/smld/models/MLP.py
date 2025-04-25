import torch
import torch.nn as nn
import numpy as np
import functools
from helpers import marginal_prob_std

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(
        self, 
        embed_dim, 
        scale=30.,
    ):
        super().__init__()
        # Randomly sample weights during initialization. 
        # These are fixed during optimization, not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale, 
            requires_grad=False,
        )
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Multilayer perceptron diffusion model for multireference alignment.
class MLP(nn.Module):
    def __init__(
        self, 
        marginal_prob_std, 
        length, 
        hidden_dim, 
        hidden_layers=2, 
        embed_dim=None,
    ):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.hidden_layers = hidden_layers

        self.act = nn.ReLU()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.emb_dense_alpha = nn.Linear(embed_dim, embed_dim)
        self.dense_alpha = nn.Linear(length+embed_dim, hidden_dim)
        
        self.emb_dense_list = nn.ModuleList([])
        self.dense_list = nn.ModuleList([])
        for idx in range(self.hidden_layers):
            self.emb_dense_list.append(nn.Linear(embed_dim, embed_dim))
            self.dense_list.append(nn.Linear(hidden_dim+embed_dim, hidden_dim))
        
        self.emb_dense_omega = nn.Linear(embed_dim, embed_dim)
        self.dense_omega = nn.Linear(hidden_dim+embed_dim, length)
    
    def forward(self, x, t):  
        
        emb_t = self.embed(t)
        emb_t = self.act(emb_t)
        
        ht = self.emb_dense_alpha(emb_t)
        ht = self.act(ht)
        h = self.dense_alpha(torch.cat((x, ht), dim=1))
        h = self.act(h)

        for idx in range(self.hidden_layers):
            ht = self.emb_dense_list[idx](emb_t)
            ht = self.act(ht)
            h = self.dense_list[idx](torch.cat((h, ht), dim=1))
            h = self.act(h)
        
        ht = self.emb_dense_omega(emb_t)
        ht = self.act(ht)
        h = self.dense_omega(torch.cat((h, ht), dim=1))

        score = h / self.marginal_prob_std(t)[:, None]
        return score

    def __str__(self):
        return 'mlp'
    