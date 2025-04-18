import torch
import numpy as np

# Helper function, makes circulants.
def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat(
        [
            tensor.flip((dim,)), 
            torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1,)
        ], 
        dim=dim,
    )
    return tmp.unfold(dim, S, 1).flip((-1,))

# Parent class, all signal distributions should inherit and override/implement this class.
class SignalSampler:
    def __init__(self, length):
        self.length = length

    def __call__(self, size, generator, device):
        pass

    def __len__(self):
        return self.length

# Choosing some vector mu, this signal is distributed around mu with scale sigma.
class GaussianSignal(SignalSampler):
    def __init__(self, length, mu, sigma):
        super().__init__(length)
        assert len(mu) == length
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size=[], generator=None, device='cpu'):
        size_out = list(size) + [self.length]
        return self.mu + self.sigma * torch.randn(size=size_out,
                                                  generator=generator,
                                                  device=device)

class DegenerateLoop(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
    ):
        super().__init__(length)     
        self.scale = scale
        self.signal = signal
        self.signal_circulant = circulant(self.signal, 0)

    def __call__(
        self, 
        size=[1,],
        generator=None, 
        device='cpu',
    ):
        do_random_shifts=False
        if do_random_shifts:
            shifts = torch.randint(
                low=0, 
                high=self.length, 
                size=size, 
                generator=generator, 
                device=device,
            )
        else:
            shifts = torch.zeros(
                size, 
                dtype=torch.long, 
                device=device,
            )
        samples = self.signal_circulant[shifts, :].to(device)
        rands = torch.rand(
            size, 
            device=device, 
            generator=generator,
        )
        samples += self.scale*self.circle_loop(rands, shifts, generator, device)
        return samples

    # The function loop_func should take a float between 0 and 1, a signal,
    # a shift, a scale, a device and return a scaled and shifted tensor 
    # on device with the shape of signal.
    def circle_loop(self, rands, shifts, generator, device):
        mat = torch.zeros(
            (shifts.shape[0], self.length), 
            device=device,
        )
        mat[:, 1] += torch.sin(2. * np.pi * rands)
        mat[:, 2] += torch.cos(2. * np.pi * rands)
        mat = mat.repeat(1, 2).unfold(1, self.length, 1)[:, :self.length, :]
        vecs = mat[torch.arange(shifts.shape[0]), -shifts, :]
        return vecs

# TO BE COMPLETED, SHOULD LOOK LIKE A SINE WAVE.
class SinSqrSignal(SignalSampler):
    def __init__(self, length, scale):
        super().__init__(length)
        assert len(mu) == length
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size=[], generator=None, device='cpu'):
        size_out = list(size) + [self.length]
        return self.mu + self.sigma * torch.randn(size=size_out,
                                                  generator=generator,
                                                  device=device)

    def _eval(self, scale=1.):
        assert scale >= 1.
        return torch.sine()
    