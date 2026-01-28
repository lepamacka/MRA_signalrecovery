import numpy as np
import torch
from torch.fft import fft
import scipy.fft
import matplotlib.pyplot as plt
import signalsamplers as samplers
from bispec_inversion.iterphasesync import bispec_inv_IPS_real

def phase_sync(vec_est, vec_true):
    
    min_diff = np.linalg.norm(vec_true - vec_est)/np.linalg.norm(vec_true)
    min_ind = 0
    for i in range(len(vec_true)):
        if min_diff > np.linalg.norm(vec_true - np.roll(vec_est, i))/np.linalg.norm(vec_true):
            min_diff = np.linalg.norm(vec_true - np.roll(vec_est, i))/np.linalg.norm(vec_true)
            min_ind = i
            
    vec = np.roll(vec_est, min_ind)
    
    return vec, min_diff

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Current device is \'{device}\'.")
generator = torch.Generator(device=device) 
generator.seed()

# Set signal parameters.
length = 41
base_signal = torch.zeros(length, device=device)
# signal_scale = 1/np.sqrt(length)
signal_scale = 1.

# Set MRA parameters.
num_samples = 100000
MRA_sigma = 10.
use_random_spectra = True
use_random_truesignal = False

# Set other parameters.
use_scipy = True
use_CLT = True

def condition_hatdistr_pwrspec(length, signal_scale, base_signal, MRA_sigma, num_samples, use_random_spectra, use_random_truesignal, use_scipy, use_CLT, generator, device):
    # Initialize signal sampler.
    signal_sampler = samplers.HatSampler(
        scale=signal_scale, 
        signal=base_signal, 
        length=length, 
        generator=generator, 
        device=device,
    )

    # Sample or set the true signal.
    if use_random_truesignal:
        signal_true = signal_sampler(do_random_shifts=False).squeeze(0)
    else:
        signal_true = signal_sampler.hats[length//2, :]#.roll(length//8)

    # Form signal circulant.
    # circulant_true = samplers.circulant(signal_true, dim=0)

    # Initialize MRA sampler.
    MRA_sampler = samplers.GaussianSampler(
        scale=MRA_sigma, 
        signal=signal_true, 
        length=length,
        generator=generator, 
        device=device,
    )

    # Generate MRA samples.
    MRA_samples = MRA_sampler(num=num_samples, do_random_shifts=True)

    # Compute or set spectra.
    if use_random_spectra:
        data_fft_1 = fft(MRA_samples, norm='ortho')
        power_spectrum_sample = torch.abs(data_fft_1).square().mean(dim=0)
    else:
        power_spectrum_true = torch.abs(fft(signal_true, norm='ortho')).square()
        power_spectrum_sample = power_spectrum_true + (MRA_sigma ** 2)
    # power_spectrum_est = power_spectrum_sample - (MRA_sigma ** 2)

    # Condition the signal sampler on power spectrum.
    signal_sampler.condition_distr(
        sample_pwrspec=power_spectrum_sample,
        num_samples=num_samples,
        sigma=MRA_sigma,
        use_CLT=use_CLT,
        use_scipy=use_scipy,
    )
    masses = torch.zeros_like(signal_sampler.distr)
    masses[0] += signal_sampler.distr[0]
    masses[1:] += signal_sampler.distr[1:] - signal_sampler.distr[:-1]
    return masses, signal_true

# conditioned_samples = signal_sampler(num=10, do_random_shifts=False)

# Get and invert bispectrum
# data_fft_2 = scipy.fft.fft((MRA_samples - MRA_samples.mean(dim=0)).numpy(force=True))
# circulant_data_fft_2 = samplers.circulant(torch.from_numpy(data_fft_2), dim=-1).numpy(force=True)
# bispectrum_sample = np.mean((np.einsum('mij, mij -> mij', np.einsum('mi, mj -> mij', data_fft_2, np.conj(data_fft_2)), circulant_data_fft_2)), axis=0)
# bispectrum_prior = np.zeros_like(bispectrum_sample)
# # for hat in signal_sampler.hats[]
# phases_est = bispec_inv_IPS_real(bispectrum_sample, maxiter=1000)
# power_spectrum_est_2 = np.mean(np.square(np.abs(data_fft_2)), axis=0) - length*(MRA_sigma ** 2)
# power_spectrum_est_2[power_spectrum_est_2 < 0.] = 0.
# fft_est = np.sqrt(power_spectrum_est_2) * phases_est
# signal_bispec = scipy.fft.ifft(fft_est) + MRA_samples.mean(dim=0).numpy(force=True)
# signal_bispec_aligned, min_diff = phase_sync(signal_bispec, signal_true.numpy(force=True))
# print(min_diff)

total_iters = 20
num_samples_vals = [int(10**(n/2)) for n in range(4, 13)] 

masses_list = torch.zeros((len(num_samples_vals), total_iters, length), device=device)
for idx_num, num_samples in enumerate(num_samples_vals):
    for idx_iter, iter in enumerate(range(total_iters)):
        print(idx_iter)
        masses, signal_true = condition_hatdistr_pwrspec(length, signal_scale, base_signal, MRA_sigma, num_samples, use_random_spectra, use_random_truesignal, use_scipy, use_CLT, generator, device)
        masses_list[idx_num, idx_iter, :] += masses

print(masses_list.shape)

true_idx = (torch.argmin(signal_true, dim=0)-1)%length

analytic_relative_errors = (masses_list.mean(dim=1)*(torch.arange(0, length, device=device)-20)).square().sum(dim=1).sqrt()/np.sqrt(20)
print(analytic_relative_errors)
print(analytic_relative_errors.shape)


fig1, ax1 = plt.subplots()
# ax1.plot(conditioned_samples.cpu().transpose(0, 1))
ax1.plot(signal_true.cpu(), 'r:', linewidth=2)
# ax1.plot(signal_bispec_aligned, 'g:', linewidth=2)

fig2, ax2 = plt.subplots()
ax2.bar(list(range(length)), masses.cpu())
ax2.vlines(true_idx.cpu(), [0.], [masses.max().cpu()], 'r')

fig3, ax3 = plt.subplots()
ax3.loglog(num_samples_vals, analytic_relative_errors.cpu())

torch.save(analytic_relative_errors.cpu(), "analytic_relative_errors.pt")

plt.show()