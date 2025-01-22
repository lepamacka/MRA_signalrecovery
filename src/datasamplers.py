import torch
import signalsamplers

# Don't know if this works. Supposed to enable reproducibility in a dataloader with the collate_fn.
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# Batch iterator for multireference alignment.
class BatchIteratorMRA:
    def __init__(
        self, 
        signal_sampler, 
        epoch_size,
    ):
        self.signals = signal_sampler(
            num=epoch_size, 
            do_random_shifts=True,
        )
        self.epoch_size = epoch_size
        self.idx = 0

    def __next__(self):
        if self.idx >= self.epoch_size:
            raise StopIteration()
        signal = self.signals[self.idx, :]
        self.idx += 1
        return signal

# Creates sample batch iterators for multireference alignment
class IterableDatasetMRA(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        signal_sampler, 
        epoch_size,
    ):
        super().__init__()
        assert isinstance(signal_sampler, signalsamplers.SignalSampler)
        self.signal_sampler = signal_sampler
        self.epoch_size = epoch_size

    def __iter__(self):
        return BatchIteratorMRA(
            signal_sampler=self.signal_sampler, 
            epoch_size=self.epoch_size,
        )