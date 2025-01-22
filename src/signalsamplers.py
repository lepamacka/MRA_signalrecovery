import torch
import numpy as np
from circulant import circulant

# Parent class, all signal distributions should inherit and override/implement this class.
class SignalSampler:
    def __init__(
        self, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        assert len(signal) == length
        self.length = length
        self.generator = generator
        self.device = device
        self.signal = signal.to(device)
        self.signal_circulant = circulant(self.signal, 0)

    def __call__(
            self, 
            size, 
            do_random_shifts=False
        ):
        pass

    def __str__(self):
        return "undefined"

    def __len__(self):
        return self.length

# This sampler is distributed around the base signal with scale sigma.
class Gaussian(SignalSampler):
    def __init__(
        self, 
        sigma, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(signal, length, generator, device)
        self.sigma = sigma

    def __call__(
            self, 
            num=1, 
            do_random_shifts=False
        ):
        size_out = (num, self.length)
        if do_random_shifts:
            shifts = torch.randint(
                low=0, 
                high=self.length, 
                size=(num,), 
                generator=self.generator, 
                device=self.device,
            )
        else:
            shifts = torch.zeros(
                (num,), 
                dtype=torch.long, 
                device=self.device
            )
        samples = self.signal_circulant[shifts.tolist(), :]
        samples += self.sigma * torch.randn(
            size=size_out, 
            generator=self.generator, 
            device=self.device
        )
        return samples
        
    def __str__(self):
        return "gauss"

# This sampler is uniformly distributed along a loop centered at signal.
class DegenerateLoop(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(signal, length, generator, device)     
        self.scale = scale

    def __call__(
        self, 
        num=1, 
        do_random_shifts=False,
    ):
        if do_random_shifts:
            shifts = torch.randint(
                low=0, 
                high=self.length, 
                size=(num,), 
                generator=self.generator, 
                device=self.device,
            )
        else:
            shifts = torch.zeros(
                (num,), 
                dtype=torch.long, 
                device=self.device,
            )
        samples = self.signal_circulant[shifts, :]
        rands = torch.rand(
            (num,), 
            device=self.device, 
            generator=self.generator,
        )
        samples += self.scale*self.circle_loop(rands, shifts)
        return samples

    # The function loop_func should take a float between 0 and 1, a signal,
    # a shift, a scale, a device and return a scaled and shifted tensor 
    # on device with the shape of signal.
    def circle_loop(self, rands, shifts):
        mat = torch.zeros(
            (shifts.shape[0], self.length), 
            device=self.device,
        )
        mat[:, 1] += torch.sin(2. * np.pi * rands)
        mat[:, 2] += torch.cos(2. * np.pi * rands)
        mat = mat.repeat(1, 2).unfold(1, self.length, 1)[:, :self.length, :]
        vecs = mat[torch.arange(shifts.shape[0]), -shifts, :]
        return vecs
        
    def __str__(self):
        return "loop"


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 
    generator.seed()

    length = 5
    signal = torch.zeros((length,))
    signal[0] = 1.
    scale = 1.0
    num = 10
    
    loop_sampler = DegenerateLoop(scale, signal, length, generator, device)
    res = loop_sampler(num, True)
    print(res)