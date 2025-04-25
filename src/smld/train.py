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

    length = 3
    signal = torch.zeros((length,))
    # signal[0, :length//2] = torch.sin(2. * math.pi * torch.arange(0, length//2)/length)
    signal[0] = 1.
    signal_scale = .3
    signal_sampler = signalsamplers.DegenerateLoop(
        signal_scale, 
        signal, 
        length, 
        generator=generator, 
        device=device,
    )

    hidden_layers = 5
    hidden_dim = 32
    embed_dim = 32
    model_sigma = 2.5

    batch_size = 2**8
    batch_num = 2**12
    n_epochs = 100
    learning_rate = 1e-4
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
        f"sigma{model_sigma}",
        f"{str(signal_sampler)}{signal_scale}",
    ))
    
    PATH = "./../model_weights/smld/" + model_path_name + "/" # Not location safe.
    print(f"Model will be saved in \'{PATH}\'") 
    
    dataset = IterableDatasetMRA(
        signal_sampler=signal_sampler, 
        epoch_size=epoch_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size
    ) # Maybe use: worker_init_fn=seed_worker, see datasamplers.py
    

    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate
    )

    scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
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
            inputs = inputs.to(device)
            loss, specloss = loss_fn(
                model, 
                inputs, 
                marginal_prob_std_fn, 
                device=device,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg += loss
            specloss_avg += specloss
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
    
    PATH = "./../model_weights/smld/" 
    if not os.path.exists(PATH):
        raise ValueError
    PATH = PATH + model_name + "/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH+"weights_dict.pth")
    torch.save(signal, PATH+"signal.pth")