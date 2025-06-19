import torch
import numpy as np
import matplotlib.pyplot as plt

# Helper function, makes circulant matrix of input signal.
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

def align(input, base):        
    input_circulant = circulant(input, dim=-1)
    best_shift_idx = (base.unsqueeze(0) - input_circulant).square().sum(dim=-1).min(dim=-1)[1]
    # tmp = (base.unsqueeze(0) - input_circulant).square().sum(dim=-1)
    # best_shift_idx = tmp.min(dim=-1)[1]
    if input.ndim == 1:
        return input_circulant[best_shift_idx, :]
    elif input.ndim == 2:
        return input_circulant[torch.arange(input.shape[0]), best_shift_idx, :]
    else: 
        raise ValueError(f"{input.ndim = }, has to be 1 or 2.")

# Parent class, all signal distributions should inherit and override/implement this class.
class SignalSampler:
    def __init__(
        self, 
        scale,
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        if len(signal) != length:
            raise ValueError(f"Must choose 1D signal with {length = }")
        self.scale = scale
        self.length = length
        self.generator = generator
        self.device = device
        self.signal = signal.to(device)
        self.signal_circulant = circulant(self.signal, 0)

    def __call__(
            self, 
            num, 
            do_random_shifts=False
        ):
        shifts = self.generate_shifts(num, do_random_shifts)
        samples = self.signal_circulant[shifts.tolist(), :]
        return samples
    
    def generate_shifts(
            self,
            num,
            do_random_shifts,
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
                size=(num,), 
                dtype=torch.long, 
                device=self.device
            )
        return shifts

    def __str__(self):
        return "undefined"

    def __len__(self):
        return self.length

# This sampler is distributed around the base signal with scale sigma.
class GaussianSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, generator, device)

    def __call__(
            self, 
            num=1, 
            do_random_shifts=False
        ):
        shifts = self.generate_shifts(num, do_random_shifts)
        samples = self.signal_circulant[shifts, :]
        samples += self.scale * torch.randn(
            size=(num, self.length), 
            generator=self.generator, 
            device=self.device,
        )
        return samples
        
    def __str__(self):
        return "gauss"

# This sampler is uniformly distributed along a loop centered at signal.
class DegenerateLoopSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, generator, device)     

    def __call__(
        self, 
        num=1, 
        do_random_shifts=False,
    ):
        shifts = self.generate_shifts(num, do_random_shifts)
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

class PlanckSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, generator, device) 
        self.Y = torch.distributions.chi2.Chi2(df=4)
        self.Z = torch.distributions.uniform.Uniform(0, 1)
    
    def __call__(
        self, 
        num=1,
        do_random_shifts=False,
    ):
        shifts = self.generate_shifts(num, do_random_shifts)
        y = self.Y.sample((num,)).to(self.device) / 2.
        z = self.Z.sample((num,)).to(self.device)
        n = self.basel(z)
        freqs = y / (self.scale * n)
        signals = self.signal + torch.sin(
            torch.einsum(
                'n, l -> nl', 
                freqs, 
                torch.linspace(0, 2 * np.pi, self.length, device=self.device),
            ),
        )
        signals_circulants = circulant(signals, dim=-1)
        samples = signals_circulants[torch.arange(0, num), shifts, :]
        return samples
    
    def basel(self, rands):
        basel_nums = torch.zeros_like(rands)
        for idx, rand in enumerate(rands):
            thresh = (np.pi ** 2) * rand / 6.
            partial = 0.
            k = 0
            while partial < thresh:
                k += 1
                partial += 1. / (k ** 2)
            basel_nums[idx] += k
        return basel_nums

    def __str__(self):
        return "planck"
    
class HatSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, generator, device) 
        self.hats = torch.tril(torch.ones(size=(length, length), device=device))
    
    def __call__(
        self, 
        num=1,
        do_random_shifts=False,
    ):
        shifts = self.generate_shifts(num, do_random_shifts)
        step_idxs = torch.randint(
            low=0, 
            size=(num,),
            high=self.length, 
            generator=self.generator, 
            device=self.device,
        )
        signals = self.signal + self.hats[step_idxs, :]
        signals_circulants = circulant(signals, dim=-1)
        samples = signals_circulants[torch.arange(0, num), shifts, :]
        return samples

    def __str__(self):
        return "hat"

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 
    generator.seed()

    length = 41
    signal = torch.zeros((length,))
    # signal[0] = 1.
    scale = 1.0
    num = 1
    
    loop_sampler = PlanckSampler(scale, signal, length, generator, device)
    res = loop_sampler(num, False)

    print(torch.abs(torch.fft.fft(res, norm='ortho')).square().mean(dim=0))

    # plt.plot(res.to('cpu').T)
    # plt.show()