import torch

# Parent class, all signal distributions should inherit and override/implement this class.
class SignalSampler:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, size, generator, device):
        pass

# Choosing some vector mu, this signal is distributed around mu with scale sigma.
class GaussianSignal(SignalSampler):
    def __init__(self, channels, mu, sigma):
        super().__init__(channels)
        assert len(mu) == channels
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size=[], generator=None, device='cpu'):
        size_out = list(size) + [self.channels]
        return self.mu + self.sigma * torch.randn(size=size_out,
                                                  generator=generator,
                                                  device=device)

# TO BE COMPLETED, SHOULD LOOK LIKE A SINE WAVE.
class SinSqrSignal(SignalSampler):
    def __init__(self, channels, scale):
        super().__init__(channels)
        assert len(mu) == channels
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size=[], generator=None, device='cpu'):
        size_out = list(size) + [self.channels]
        return self.mu + self.sigma * torch.randn(size=size_out,
                                                  generator=generator,
                                                  device=device)

    def _eval(self, scale=1.):
        assert scale >= 1.
        return torch.sine()
    