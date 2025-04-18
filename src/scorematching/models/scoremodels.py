import torch
import torch.nn as nn
import numpy as np

# Score model for multireference alignment, equivariant with respect to shifts.
class ConvScoreModel(nn.Module):
    def __init__(self, length, hiddendim):
        super().__init__()
        assert length%2 == 1, "Can only handle odd channel numbers due to padding peculiarities."

        self.act = nn.Tanh()
        self.conv1 = BatchConvolver1D(length, 1, hiddendim, hiddendim*length)
        self.conv2 = BatchConvolver1D(length, hiddendim, hiddendim, hiddendim*length)
        self.conv3 = BatchConvolver1D(length, hiddendim, 1, hiddendim*length)
        # self.embed = nn.Sequential(
        #     nn.Linear(1, hiddendim),
        #     self.act,
        #     nn.Linear(hiddendim, hiddendim),
        #     self.act,
        #     nn.Linear(hiddendim, hiddendim*length),
        #     self.act
        # )
        self.register_parameter(
            "ht1",
            nn.Parameter(torch.randn((hiddendim*length,)))
        )
        self.register_parameter(
            "ht2",
            nn.Parameter(torch.randn((hiddendim*length,)))
        )
        self.register_parameter(
            "ht3",
            nn.Parameter(torch.randn((hiddendim*length,)))
        )

    def forward(self, x):
        h = x.unsqueeze(1)

        h = self.conv1(h, self.ht1)
        h = self.act(h)
        h = self.conv2(h, self.ht2)
        h = self.act(h)
        h = self.conv3(h, self.ht3)
        
        out = h.squeeze(1)
        return out

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
        kernels = kernels.expand(h.shape[0], -1, -1, -1)
        h_pad = self.circpad(h)
        output = torch.vmap(nn.functional.conv1d)(h_pad, kernels)
        return output
    