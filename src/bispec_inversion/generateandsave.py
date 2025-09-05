import os
import numpy as np
import matplotlib.pyplot as plt
from helperfuncs import fft_stats, gen_inv_seq_numpts, gen_inv_seq_sigma
from nonconvopt import bispec_inv_PM_real, bispec_inv_PM_complex
from iterphasesync import bispec_inv_IPS_real

def save_stats_to_file(true_sig, dft, pwrspec, bispec, bispec_adj, relerr, est_sig, times, name = None):
    
    assert os.path.exists(name) is False
    
    with open(name, 'wb') as f:
        np.save(f, true_sig)
        np.save(f, dft)
        np.save(f, pwrspec)
        np.save(f, bispec)
        np.save(f, bispec_adj)
        np.save(f, relerr)
        np.save(f, est_sig)
        np.save(f, times)
        
def load_stats_from_file(name = None):
    
    assert os.path.exists(name) is True
    
    with open(name, 'rb') as f:
        true_sig = np.load(f)
        dft = np.load(f)
        pwrspec = np.load(f)
        bispec = np.load(f)
        bispec_adj = np.load(f)
        relerr = np.load(f)
        est_sig = np.load(f)
        times = np.load(f)
        
    return true_sig, dft, pwrspec, bispec, bispec_adj, relerr, est_sig, times

if __name__ == '__main__':
    
    np.random.seed(2)
    
    N = 20
    N = N * 2 + 1
    batch_size = 10 ** 2
    batch_number = 10 ** 3
    # batch_seq = np.concatenate((np.array([10**1]), np.outer(np.array([10**0, 10**1, 10**2, 10**3]), np.array([25, 60])).ravel()))
    # batch_seq = np.round(10 ** np.linspace(0, 5, 11)).astype(np.int32)
    # batch_seq[1:] -= batch_seq[:-1]
    # sigma = 10 ** 0
    sigma_seq = 10 ** np.linspace(-1, 1, 9)
    signal_true = np.zeros(N, dtype=np.complex128)
    
    # signal_true[0:(N-1)//2] = 1.
    # signal_true[:] -= ((N-1)/2)/N + 0.01
    
    # signal_true[0:(N-1)//2] = np.random.normal(0, 1, size=(N-1)//2)
    # signal_true[0:(N-1)//2] = np.linspace(0, 3, num=(N-1)//2)
    # signal_true[0:(N-1)//2] = np.sqrt(2) * np.sin(np.arange((N-1)//2)*6*np.pi/N)
    # signal_true[0:(N-1)//2] = 1 + np.linspace(0, 4, num=(N-1)//2) * 1j
    # cmplx = np.any(np.iscomplex(signal_true))
    
    # dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_IPS = gen_inv_seq_numpts(signal_true, sigma, batch_size, batch_seq, bispec_inv_PM_real)
    # dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_PM = gen_inv_seq_numpts(signal_true, sigma, batch_size, batch_seq, bispec_inv_IPS_real)
    
    dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = gen_inv_seq_sigma(signal_true, sigma_seq, batch_size, batch_number, bispec_inv_PM_real)
    dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS = gen_inv_seq_sigma(signal_true, sigma_seq, batch_size, batch_number, bispec_inv_IPS_real)
    
    save_stats_to_file(signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM, 'PM_semisin_sigmaseq_neg1_1_100kpts.npy')
    save_stats_to_file(signal_true, dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS, 'IPS_semisin_sigmaseq_neg1_1_100kpts.npy')

    
    # signal_mean, signal_fft, signal_pwrspec, signal_bispec = fft_stats(signal_true)
    # _, __, ___, signal_adj_bispec = fft_stats(signal_true-signal_mean)
    
    # fig, ax = plt.subplots()
    # ax.loglog(batch_size*batch_seq, relerr_PM, 'go--')
    # ax.loglog(batch_size*batch_seq, relerr_IPS, 'rs--')