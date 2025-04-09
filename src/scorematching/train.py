import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import math
import numpy as np
import matplotlib.pyplot as plt
import samplers
from models.scoremodels import ConvScoreModel
from losses import loss_fn_score
from datasets import seed_worker, ReferenceVectorSampler
from time import perf_counter
import os

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")

    generator = torch.Generator(device=device) 
    generator_cpu = torch.Generator(device='cpu')
    generator.seed()
    generator_cpu.seed()
    # generator.manual_seed(1337)
    # generator_cpu.manual_seed(1337)

    channels = 3
    hiddendim = 32
    
    model = ConvScoreModel(length=channels, hiddendim=hiddendim).to(device)
    model_name = f"MRA_convscoremodel_length{channels}_hiddim{hiddendim}"
    model.train()
    
    batchsize = 2**6
    batchnum = 2**6
    n_epochs = 100
    digs = int(math.log10(n_epochs))+1
    epochsize = batchsize * batchnum

    signal = torch.zeros((channels))
    # signal[0, :channels//2] = torch.sin(2. * math.pi * torch.arange(0, channels//2)/channels)
    signal[0] = 1
    # signal = torch.randn((1,channels), generator=generator)
    signal_sigma = 0.2
    signal_sampler = samplers.GaussianSignal(channels=channels, 
                                            mu=signal, 
                                            sigma=signal_sigma)
    
    SNR = 9
    # sigma = torch.linalg.vector_norm(signal)/((channels**0.5)*(SNR**0.5))
    sigma = torch.tensor([0.2])
    print(f"Scale of noise is: {sigma.item():.3f}")

    dataset = ReferenceVectorSampler(signal_sampler=signal_sampler, 
                                     channels=channels, 
                                     sigma=sigma, 
                                     epochsize=epochsize, 
                                     generator=generator_cpu,
                                     device='cpu')
    
    # Have not found a way to ensure reproducibility with seed, probably due to dataloader peculiarities. 
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batchsize, 
                                             worker_init_fn=seed_worker, 
                                             generator=generator_cpu)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training samples: {epochsize*n_epochs}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = LinearLR(optimizer, 
                         start_factor=0.2, 
                         end_factor=1, 
                         total_iters=n_epochs//2)
    
    t_0 = perf_counter()
    for epoch in range(n_epochs):
        t_e0 = perf_counter()
        loss_avg = torch.tensor(0., requires_grad=False, device=device)
        t_b = [0]*4
        for inputs in dataloader:
            t_temp = [0]*5
            t_temp[0] = perf_counter()
            inputs = inputs.to(device)
            t_temp[1] = perf_counter()
            loss = loss_fn_score(model, inputs).mean()
            t_temp[2] = perf_counter()
            optimizer.zero_grad()
            loss.backward()
            t_temp[3] = perf_counter()
            optimizer.step()
            t_temp[4] = perf_counter()
            loss_avg += loss/batchnum
            t_b = [t_b[n] + t_temp[n+1] - t_temp[n] for n in range(4)]
        scheduler.step()
        t_e1 = perf_counter()
        print(f"Epoch: {epoch+1:{digs}d}/{n_epochs}" +
              f" | Loss: {loss_avg:9.2f}" +
              f" | Time: {t_e1-t_e0:8.4f}s" +
              f" | toGPU time: {t_b[0]:8.4f}s" +
              f" | Loss time: {t_b[1]:8.4f}s" +
              f" | Backward time: {t_b[2]:8.4f}s" +
              f" | Step time: {t_b[3]:8.4f}s")
    t_1 = perf_counter()
    
    print(f"\nTotal time elapsed: {t_1-t_0} secs.")
    
    PATH = "model_weights/" + model_name + "/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH+"weights_dict.pth")
    torch.save(signal, PATH+"signal.pth")