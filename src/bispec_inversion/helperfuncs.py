import time
import numpy as np
import matplotlib.pyplot as plt
import circulant
from scipy.fft import fft, ifft
from mradatagen import mra_batch_gen, mra_est
from nonconvopt import bispec_inv_PM_real, bispec_inv_PM_complex
from iterphasesync import bispec_inv_IPS_real

# Finds the rotation index of vec_est with smallest norm difference to vec_true.
def phase_sync(vec_est, vec_true):
    
    min_diff = np.linalg.norm(vec_true - vec_est)/np.linalg.norm(vec_true)
    min_ind = 0
    for i in range(len(vec_true)):
        if min_diff > np.linalg.norm(vec_true - np.roll(vec_est, i))/np.linalg.norm(vec_true):
            min_diff = np.linalg.norm(vec_true - np.roll(vec_est, i))/np.linalg.norm(vec_true)
            min_ind = i
            
    vec = np.roll(vec_est, min_ind)
    
    return vec, min_diff

# Computes the relevant stats for a given signal x.
def fft_stats(x):
    
    x_mean = np.mean(x)
    x_fft = fft(x)
    x_pwrspec = x_fft * np.conj(x_fft)
    x_bispec = np.outer(x_fft, np.conj(x_fft)) * circulant.circulant(x_fft)
    
    return x_mean, x_fft, x_pwrspec, x_bispec

# Reconstructs the dft of a signal given the mean, amplitudes and phases.
def reconstruct_dft(mean_estimate, pwrspec_estimate, phases_estimate):
    
    assert mean_estimate != 0.
    
    rotation = mean_estimate / np.abs(mean_estimate)
    reconstr_dft = np.sqrt(pwrspec_estimate) * phases_estimate * rotation
    reconstr_dft[0] = mean_estimate * len(pwrspec_estimate)
    
    return reconstr_dft

# Reconstructs signal given its mean, power spectrum and bispectrum, plus some inverter func.
def invert(mean, pwrspec, bispec_adj, bispec_inverter):
    
    phases = bispec_inverter(bispec_adj)
    dft = reconstruct_dft(mean, pwrspec, phases)
    res = ifft(dft)
    
    return res

# Generates data according to a sequence of batch numbers and computes relevant stats cumulatively along the sequence.
def gen_inv_seq_numpts(signal, sigma, batch, batch_num_seq, bispec_inverter, rand_shifts=True):
    
    seq_len = len(batch_num_seq)
    n = len(signal)
    cmplx = np.any(np.iscomplex(signal))
    
    fft_seq = pwrspec_seq = np.zeros(shape=(seq_len, n), dtype=np.complex128)
    pwrspec_seq = np.zeros(shape=(seq_len, n))
    bispec_seq = np.zeros(shape=(seq_len, n, n), dtype=np.complex128)
    bispec_adj_seq = np.zeros(shape=(seq_len, n, n), dtype=np.complex128)
    relerr_seq = np.zeros(seq_len)
    time_seq = np.zeros(seq_len)
    
    m = 0
    fft_tot = np.zeros(n, dtype=np.complex128)
    pwrspec_tot = np.zeros(n)
    bispec_tot = np.zeros((n, n), dtype=np.complex128)
    outer_tot = np.zeros((n, n), dtype=np.complex128)
    revmul_tot = np.zeros(n, dtype=np.complex128)
    
    t_start = time.perf_counter()
    
    for idx, batch_num in enumerate(batch_num_seq):
        
        m += batch * batch_num
        
        fft_temp, pwrspec_temp, bispec_temp, outer_temp, revmul_temp = mra_batch_gen(signal, batch, batch_num, sigma, rand_shifts)
        
        fft_tot += fft_temp
        pwrspec_tot += pwrspec_temp
        bispec_tot += bispec_temp
        outer_tot += outer_temp
        revmul_tot += revmul_temp
        
        sigma_est, mean_est, pwrspec_est, bispec_est, bispec_adj_est = mra_est(n, m, fft_tot, pwrspec_tot, bispec_tot, outer_tot, revmul_tot, cmplx)
        
        signal_est, relerr = phase_sync(invert(mean_est, pwrspec_est, bispec_adj_est, bispec_inverter), signal)
        
        fft_seq[idx, :] += fft_tot/m
        pwrspec_seq[idx, :] += pwrspec_est
        bispec_seq[idx, :, :] += bispec_est
        bispec_adj_seq[idx, :, :] += bispec_adj_est
        relerr_seq[idx] += relerr
        time_seq[idx] += time.perf_counter() - t_start
    
    return fft_seq, pwrspec_seq, bispec_seq, bispec_adj_seq, relerr_seq, signal_est, time_seq

# Generates data according to a sequence of std devs and computes relevant stats along the sequence.    
def gen_inv_seq_sigma(signal, sigma_seq, batch, batch_num, bispec_inverter, rand_shifts=True):
    
    seq_len = len(sigma_seq)
    n = len(signal)
    m = batch * batch_num
    cmplx = np.any(np.iscomplex(signal))
    
    fft_seq = pwrspec_seq = np.zeros(shape=(seq_len, n), dtype=np.complex128)
    pwrspec_seq = np.zeros(shape=(seq_len, n))
    bispec_seq = np.zeros(shape=(seq_len, n, n), dtype=np.complex128)
    bispec_adj_seq = np.zeros(shape=(seq_len, n, n), dtype=np.complex128)
    relerr_seq = np.zeros(seq_len)
    time_seq = np.zeros(seq_len)

    
    t_start = time.perf_counter()
    
    for idx, sigma in enumerate(sigma_seq):
        
        fft_tot, pwrspec_tot, bispec_tot, outer_tot, revmul_tot = mra_batch_gen(signal, batch, batch_num, sigma, rand_shifts)
        
        sigma_est, mean_est, pwrspec_est, bispec_est, bispec_adj_est = mra_est(n, m, fft_tot, pwrspec_tot, bispec_tot, outer_tot, revmul_tot, cmplx)
        
        signal_est, relerr = phase_sync(invert(mean_est, pwrspec_est, bispec_adj_est, bispec_inverter), signal)
        
        fft_seq[idx, :] += fft_tot/m
        pwrspec_seq[idx, :] += pwrspec_est
        bispec_seq[idx, :, :] += bispec_est
        bispec_adj_seq[idx, :, :] += bispec_adj_est
        relerr_seq[idx] += relerr
        time_seq[idx] += time.perf_counter() - t_start
    
    return fft_seq, pwrspec_seq, bispec_seq, bispec_adj_seq, relerr_seq, signal_est, time_seq


if __name__ == '__main__':
    
    np.random.seed(1)

    N = 20
    N = N * 2 + 1 
    batch_size = 10 ** 2
    batch_number = 10 ** 2
    # batch_seq = np.concatenate((np.array([10**1]), np.outer(np.array([10**0, 10**1, 10**2, 10**3]), np.array([25, 60])).ravel()))
    # batch_seq = np.round(10 ** np.linspace(0, 4, 9)).astype(np.int32)
    # M = batch_size * batch_number
    # sigma = 10 ** 0
    sigma_seq = 10 ** np.linspace(-1, 1, 9)
    signal_true = np.zeros(N, dtype=np.complex128)
    iters_IPS = 100
    
    # signal_true[0:(N-1)//2] = 1.
    # signal_true[:] -= ((N-1)/2)/N + 0.01
    
    # signal_true = np.random.normal(0.5, 2, size=N)
    # signal_true[0:(N-1)//2] = np.linspace(0, 3, num=(N-1)//2)
    signal_true[0:(N-1)//2] = 5 * np.sin(np.arange((N-1)//2)*6*np.pi/N)
    # signal_true[0:(N-1)//2] = 1 + np.linspace(0, 4, num=(N-1)//2) * 1j
    # cmplx = np.any(np.iscomplex(signal_true))
    
    # pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_IPS = gen_inv_seq_numpts(signal_true, sigma, batch_size, batch_seq, bispec_inv_PM_real)
    # pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_PM = gen_inv_seq_numpts(signal_true, sigma, batch_size, batch_seq, bispec_inv_IPS_real)
    
    fft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_IPS = gen_inv_seq_sigma(signal_true, sigma_seq, batch_size, batch_number, bispec_inv_PM_real)
    fft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_PM = gen_inv_seq_sigma(signal_true, sigma_seq, batch_size, batch_number, bispec_inv_IPS_real)
    
    
    print(time_IPS)
    print(time_PM)
    
    # t_stop = time.perf_counter()
    # print('Elapsed time, secs: ' + str(t_stop - t_start))
    
    signal_mean, signal_fft, signal_pwrspec, signal_bispec = fft_stats(signal_true)
    _, __, ___, signal_adj_bispec = fft_stats(signal_true-signal_mean)
    
    # fig, ax = plt.subplots()
    # ax.loglog(batch_size*batch_seq, relerr_PM, 'go--')
    # ax.loglog(batch_size*batch_seq, relerr_IPS, 'rs--')
    
    fig, ax = plt.subplots()
    ax.loglog(sigma_seq, relerr_PM, 'go--')
    ax.loglog(sigma_seq, relerr_IPS, 'rs--')
    
    
    # prior_str = 10 ** 4
    # prior_prec = 10 ** 1
    # prior_df = 3
    # prior = signal_true + np.random.normal(0, sigma/prior_prec, N)
    # prior = signal_true + sigma * np.random.standard_t(prior_df) / (np.sqrt(prior_df / (prior_df - 2)) * prior_prec)
    # fac_data = M / (M + prior_str)
    # fac_prior = prior_str / (M + prior_str)
    # prior_mean, prior_fft, prior_pwrspec, prior_bispec = fft_stats(prior)

    # fft_t, pwrspec_t, bispec_t, outer_t, revmul_t = mra_batch_gen(signal_true, batch_size, batch_number, sigma)
    # sigma_e, mean_e, pwrspec_e, bispec_e, bispec_adj_e = mra_est(N, M, fft_t, pwrspec_t, bispec_t, outer_t, revmul_t, cmplx)
    
    # mean_e = mean_e * fac_data + prior_mean * fac_prior
    # pwrspec_e = pwrspec_e * fac_data + prior_pwrspec * fac_prior
    # bispec_e = bispec_e * fac_data + prior_bispec * fac_prior
    
    # signal_IPS_e = invert(mean_e, pwrspec_e, bispec_adj_e, bispec_inv_IPS_real)
    # signal_PM_e = invert(mean_e, pwrspec_e, bispec_adj_e, bispec_inv_PM_real)
    
    # signal_IPS_e, relerr_IPS = phase_sync(signal_IPS_e, signal_true)
    # signal_PM_e, relerr_PM = phase_sync(signal_PM_e, signal_true)
    
    # print('IPS relerr = ' + str(np.linalg.norm(signal_true - signal_IPS_e)/np.linalg.norm(signal_true)))
    # print('PM relerr = ' + str(np.linalg.norm(signal_true - signal_PM_e)/np.linalg.norm(signal_true)))
    
    # print('Sigma relerr = ' + str(np.linalg.norm((sigma - sigma_e)/sigma)))
    # print('Mean relerr = ' + str(np.linalg.norm((signal_mean - mean_e)/signal_mean)))
    # print('Powerspec relerr = ' + str(np.linalg.norm(signal_pwrspec - pwrspec_e)/np.linalg.norm(signal_pwrspec)))
    # print('Bispec relerr = ' + str(np.linalg.norm(signal_bispec - bispec_e)/np.linalg.norm(signal_bispec)))
    # print('Bispec adj relerr = ' + str(np.linalg.norm(signal_adj_bispec - bispec_adj_e)/np.linalg.norm(signal_adj_bispec)))
    
    # x = np.arange(0, N)
    # y1 = signal_true
    # y2 = signal_IPS_e
    # y3 = signal_PM_e
    # fig, ax = plt.subplots()
    # ax.plot(x, y1)
    # ax.plot(x, y2)
    # ax.plot(x, y3)
