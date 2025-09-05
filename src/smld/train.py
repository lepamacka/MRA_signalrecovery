import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import functools
import math
import numpy as np
import matplotlib.pyplot as plt
import signalsamplers
from models.convolutional import Convolutional
from models.MLP import MLP
from losses import loss_fn
from datasamplers import IterableDatasetMRA
from models.helpers import marginal_prob_std
from time import perf_counter
import os

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'")
    generator = torch.Generator(device=device) 
    generator.seed()

    length = 41
    base_signal = torch.zeros((length,))
    # base_signal[0, :length//2] = torch.sin(2. * math.pi * torch.arange(0, length//2)/length)
    # base_signal[0] = 1.
    sampler_scale = 1.
    center = True

    # signal_sampler = signalsamplers.DegenerateLoopSampler(
    #     scale=signal_scale, 
    #     signal=base_signal, 
    #     length=length, 
    #     center=center,
    #     generator=generator, 
    #     device=device,
    # )
    # signal_sampler = signalsamplers.PlanckSampler(
    #     scale=sampler_scale, 
    #     signal=base_signal, 
    #     length=length, 
    #     center=center,
    #     generator=generator, 
    #     device=device,
    # )
    signal_sampler = signalsamplers.HatSampler(
        scale=sampler_scale, 
        signal=base_signal, 
        length=length, 
        center=center,
        generator=generator, 
        device=device,
    )

    hidden_layers = 8
    hidden_dim = 4
    embed_dim = 64
    model_sigma = 3.0

    batch_size = 2**8
    batch_num = 2**8
    n_epochs = 1000
    learning_rate = 1e-3
    scheduler_startfactor = 0.1
    digs = int(math.log10(n_epochs))+1
    epoch_size = batch_size * batch_num

    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, 
        sigma=model_sigma, 
        device=device,
    )
    model = Convolutional(
        marginal_prob_std_fn, 
        length, 
        hidden_dim, 
        hidden_layers, 
        embed_dim,
    ).to(device)
    # model = MLP(
    #     marginal_prob_std_fn, 
    #     length, 
    #     hidden_dim, 
    #     hidden_layers, 
    #     embed_dim,
    # ).to(device)
    model.train()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training samples: {epoch_size*n_epochs}")

    #SIGMA -> SCALE
    model_path_name = "_".join((
        f"{str(model)}",
        f"len{length}",
        f"lay{hidden_layers}",
        f"hid{hidden_dim}",
        f"emb{embed_dim}",
        f"sgm{model_sigma}",
        f"{str(signal_sampler)}{sampler_scale}",
        f"cnt{center}",
    ))
    
    PATH = "./../../../model_weights/smld/" # NOT LOCATION SAFE
    assert os.path.exists(PATH), "PATH must exist."

    PATH = PATH + model_path_name + "/" 
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    print(f"Model weights will be saved in \'{PATH}\'") 
    
    dataset = IterableDatasetMRA(
        signal_sampler=signal_sampler, 
        epoch_size=epoch_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
    ) # Maybe use: worker_init_fn=seed_worker, see datasamplers.py

    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )

    scheduler = LinearLR(
        optimizer, 
        start_factor=scheduler_startfactor, 
        end_factor=1, 
        total_iters=n_epochs//3,
    )
    
    t_0 = perf_counter()
    for epoch in range(n_epochs):
        t_e0 = perf_counter()
        loss_avg = torch.tensor(
            0., 
            requires_grad=False, 
            device=device,
        )
        specloss_avg = torch.tensor(
            [0., 0., 0.], 
            requires_grad=False, 
            device=device
        )
        for inputs in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            loss, specloss = loss_fn(
                model, 
                inputs, 
                marginal_prob_std_fn, 
                device=device,
            )
            loss_avg += loss
            specloss_avg += specloss
            loss.backward()
            optimizer.step()
        scheduler.step()
        t_e1 = perf_counter()
        loss_avg = loss_avg/batch_num
        specloss_avg = specloss_avg/batch_num
        print(
            f"Epoch: {epoch+1:{digs}d}/{n_epochs}",
            f" | Mean loss: {loss_avg:4.2f}",
            f" | Loss for t < 0.1: {specloss_avg[0]:4.2f}",
            f" | Loss for t < 0.3: {specloss_avg[1]:4.2f}",
            f" | Loss for t < 0.5: {specloss_avg[2]:4.2f}",
            f" | Epoch time: {t_e1-t_e0:5.2f}s",
        )
    t_1 = perf_counter()
    
    print(f"\nTotal time elapsed: {t_1-t_0} secs.")
    
    torch.save(model.state_dict(), PATH+"weights_dict.pth")
    torch.save(base_signal, PATH+"signal.pth")