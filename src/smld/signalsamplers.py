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
        center,
        generator=None, 
        device='cpu',
    ):
        if len(signal) != length:
            raise ValueError(f"Must choose 1D signal with {length = }")
        self.scale = scale
        self.length = length
        self.generator = generator
        self.device = device
        self.center = center
        self.signal_mean = signal.mean().to(device)
        if center:
            self.signal = (signal - signal.mean()).to(device)
        else:
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
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device) 

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
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device)     

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
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device) 
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
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device) 
        self.hats = scale * torch.tril(
            signal.to(device) + torch.ones(size=(length, length), device=device)
        )
        if center:
            self.hats -= self.hats.mean(dim=1, keepdim=True)

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

class MultiSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device) 
        self.triang = torch.tril(
            torch.ones(
                size=(length, length), 
                device=device,
            )
        )

    def __call__(
        self, 
        num=1,
        do_random_shifts=False,
    ):
        shifts = self.generate_shifts(num, do_random_shifts)
        multi_scales = torch.rand(
            (num, self.length),
            device=self.device,
            generator=self.generator,
        )
        multi_idxs = self._batched_randperm(num)
        rand_ints = torch.randint(
            low=0, 
            high=self.length, 
            size=(num,),
            generator=self.generator, 
            device=self.device,
        )
        multi_scales *= self.triang[rand_ints, :]
        samples = torch.sum(
            multi_scales[:, :, None] * self.signal_circulant[multi_idxs, :],
            dim=1,
        )
        return samples

    def __str__(self):
        return "mlt"
    
    def _batched_randperm(self, batch_size):
        out = torch.randperm(
            batch_size * self.length, 
            dtype=torch.int64, 
            device=self.device, 
            generator=self.generator,
        )
        out = out.view(batch_size, self.length)
        out = out.argsort(dim=1)
        return out
        
class BellSampler(SignalSampler):
    def __init__(
        self, 
        scale, 
        signal, 
        length, 
        intensity=None,
        chi_df=None,
        center=True,
        generator=None, 
        device='cpu',
    ):
        super().__init__(scale, signal, length, center, generator, device) 
        if intensity is not None:
            self.intensity = intensity
        else:
            self.intensity = length/4
        if chi_df is not None:
            self.tau = torch.distributions.chi2.Chi2(df=chi_df)
        else:
            self.tau = torch.distributions.chi2.Chi2(df=length//4)
        # self.amp = torch.distributions.exponential.Exponential(scale)
        self.amp = torch.distributions.uniform.Uniform(0., scale)
        if length%2 == 1:
            self.squaredists = (torch.arange(length, device=device)-length//2)**2
        else:
            raise NotImplementedError
        self.triang = torch.tril(
            torch.ones(size=(length+1, length+1), device=device),
            diagonal=-1,
        )

    def __call__(
        self, 
        num=1,
        do_random_shifts=False,
    ):
        poissons = torch.poisson(
            input=self.intensity*torch.ones((num,), device=self.device),
            generator=self.generator,
        ).to(device=self.device, dtype=torch.int64)
        poissons[poissons>=self.length] = self.length * torch.ones_like(poissons[poissons>=self.length])
        poisson_max = poissons.max().item()
        rand_ints = torch.randint(
            low=0, 
            high=self.length, 
            size=(num, poisson_max),
            generator=self.generator, 
            device=self.device,
        )
        # rand_ints = self._batched_randperm(num)
        taus = self.tau.sample((num, poisson_max)).to(self.device)
        bells = self._make_bells(rand_ints, taus, num, poisson_max)
        amps = self.amp.sample((num, poisson_max)).to(self.device)
        amps = amps * self.triang[poissons, :poisson_max]
        signals = torch.sum(amps[:, :, None] * bells, dim=1)
        shifts = self.generate_shifts(num, do_random_shifts)
        signals_circulants = circulant(signals, dim=-1)
        samples = signals_circulants[torch.arange(0, num), shifts, :]
        if self.center:
            samples -= samples.mean(dim=-1, keepdim=True)
        return samples

    def __str__(self):
        return "bell"

    def _make_bells(self, rand_ints, taus, num, poisson_max):
        tmp = (
            torch.exp(-self.squaredists[None, None, :]/(2*taus[:, :, None]))
            / torch.sqrt(np.pi * taus[:, :, None]) 
        )
        tmp_circulant = circulant(tmp, dim=-1)
        return tmp_circulant[torch.arange(num)[:, None], torch.arange(poisson_max)[None, :], rand_ints, :]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 
    generator.seed()

    length = 41
    signal = torch.zeros((length,))
    # signal[0] = 1.
    scale = 3.0
    num = 512
    center = True
    
    sampler = BellSampler(
        scale=scale, 
        signal=signal,
        length=length,
        center=center,
        generator=generator,
        device=device,
    )
    res = sampler(num, False)
    print(res.norm(dim=1).mean())
    res /= res.norm(dim=1, keepdim=True)

    res_pwrspec = torch.abs(torch.fft.fft(res, norm='ortho')).square()

    diffs = res_pwrspec[:, None] - res_pwrspec[None, :]
    print(diffs.abs().sum(dim=2).mean(dim=1).max())

    diff_min_idx = diffs.abs().sum(dim=2).mean(dim=1).argmax()
    print(diff_min_idx)

    # torch.save(res[diff_min_idx, :].to('cpu'), "./../../../model_weights/bellsignal_example.pt")


    # loop_sampler = PlanckSampler(scale, signal, length, generator, device)
    # res = loop_sampler(num, False)

    # print(torch.abs(torch.fft.fft(res, norm='ortho')).square().mean(dim=0))


    # plt.plot(res.to('cpu').T)
    plt.plot(res[diff_min_idx, :].to('cpu'))
    plt.show()