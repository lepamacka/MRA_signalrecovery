import torch
import numpy as np 
from torch.fft import fft, ifft
from torchtyping import TensorType

# Expectation Maximization
def expecmax(
        X: TensorType["N", "L"], 
        sigma: float, 
        x0: TensorType["L"], 
        tol: float=1e-10, 
        batch_niter: int=3000, 
        full_niter: int=10000,
        device: torch.device='cpu',
        generator: torch.Generator=None,
    ) -> TensorType["L"]:

    # Preprocess initial guess
    if x0.shape[0] != X.shape[1]:
        raise ValueError("Length of x0 must be same as second dim of X.")
    x0fft = fft(x0)
    xfft = x0fft

    # Preprocess data X
    Xfft = fft(X)
    sum_sqr_X = torch.sum(torch.abs(X)**2, dim=1, keepdim=True)

    # Warm start
    if X.shape[0] >= 3000:
        batch_size = 1000
        for _ in range(batch_niter):
            sample = torch.randint(0, X.shape[0], (batch_size,), generator=generator, device=device)
            xfft_new = em_iteration(xfft, Xfft[sample, :], sum_sqr_X[sample, :], sigma)
            xfft = xfft_new

    # Iterate expectation maximization   
    for iter in range(full_niter):
        xfft_new = em_iteration(xfft, Xfft, sum_sqr_X, sigma)
        if torch.norm(ifft(xfft) - ifft(xfft_new))/torch.norm(ifft(xfft)) < tol:
            break
        xfft = xfft_new

    return ifft(xfft).real

def em_iteration(
        fftx: TensorType["L"], 
        fftX: TensorType["N", "L"], 
        sum_sqr_X: TensorType["L"], 
        sigma: float,
    ) -> TensorType["L"]:
    C = ifft(fftx.conj()[None, :] * fftX).real
    T = (2 * C - sum_sqr_X) / (2 * sigma**2)
    T = T - torch.max(T, dim=1, keepdim=True)[0]
    W = torch.exp(T)
    W = W / torch.sum(W, dim=1, keepdim=True)
    fftx_new = torch.mean(fft(W).conj() * fftX, dim=0)
    return fftx_new


if __name__ == "__main__":
    from smld.signalsamplers import GaussianSampler, circulant

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current device is \'{device}\'.")
    generator = torch.Generator(device=device) 
    generator.seed()

    length = 41
    signal = torch.zeros((length,), device=device)
    signal[0:length//2] = 1.
    sigma = 10.0
    center=False
    num = 100

    sampler = GaussianSampler(
        scale=sigma, 
        signal=signal,
        length=length,
        center=center,
        generator=generator,
        device=device,
    )
    samples = sampler(num, True)

    init_guess = torch.rand((length,), device=device, generator=generator)
    em_sol = expecmax(samples, sigma, init_guess, device=device, generator=generator)
    print(init_guess)
    print(em_sol)
    print((em_sol - circulant(signal, dim=0)).square().mean(dim=-1).sqrt().min(dim=-1)[0])