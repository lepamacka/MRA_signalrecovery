import torch
import math
import signalsamplers as samplers

# Don't know if this even works. Supposed to enable reproducibility in a dataloader with the collate_fn.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Iterator for batches. Generates sample vectors for multireference alignment.
class ReferenceVectorIterator:
    def __init__(
        self, 
        signal_sampler, 
        length, 
        sigma, 
        epochsize, 
        generator=None,
        device='cpu'
    ):
        self.signals = signal_sampler(
            size=(epochsize,),
            generator=generator,
            device=device
        )
        self.shifts = torch.randint(
            low=0, 
            high=length, 
            size=(epochsize,), 
            generator=generator, 
            device=device
        )
        self.randvecs = sigma * torch.randn(
            size=(epochsize, length), 
            generator=generator, 
            device=device
        )
        self.epochsize = epochsize
        self.idx = 0

    def __next__(self):
        if self.idx >= self.epochsize:
            raise StopIteration()
        item = torch.roll(
            input=self.signals[self.idx, :], 
            shifts=self.shifts[self.idx].item(), 
            dims=-1
        ) 
        item += self.randvecs[self.idx, :]
        self.idx += 1
        return item

# Streams random samples for multireference alignment problem with isotropic gaussian noise.
# Set generator seed to enable reproducibility (doesn't seem to work?).
class ReferenceVectorSampler(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        signal_sampler, 
        length, 
        sigma, 
        epochsize, 
        generator=None,
        device='cpu'
    ):
        super().__init__()
        assert isinstance(signal_sampler, samplers.SignalSampler)
        assert len(signal_sampler) == length
        self.signal_sampler = signal_sampler
        self.length = length
        self.sigma = sigma
        self.epochsize = epochsize
        self.generator = generator
        self.device = device

    def __iter__(self):
        return ReferenceVectorIterator(
            signal_sampler=self.signal_sampler, 
            length=self.length, 
            sigma=self.sigma, 
            epochsize=self.epochsize, 
            generator=self.generator, 
            device=self.device
        )
