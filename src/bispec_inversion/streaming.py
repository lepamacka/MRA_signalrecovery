import numpy as np
from scipy.fft import fft
from scipy.linalg import circulant
#import matplotlib.pyplot as plt

np.random.seed(0)

N = 10
N = N*2 + 1
M = 10**5
sigma = 1

signal = np.zeros(N)
signal[0:N//2] = 1

signal_mean = np.mean(signal)
signal_fft = fft(signal)
signal_pwrspec = signal_fft * np.conj(signal_fft)
signal_bispec = np.outer(signal_fft, np.conj(signal_fft)) * np.transpose(circulant(signal_fft))

data_sum_tot = 0
data_sqr_tot = 0
data_pwrspec_tot = np.zeros(N, dtype=np.complex128)
data_bispec_tot = np.zeros(shape=(N, N), dtype=np.complex128)

signal_circ = np.concatenate([signal, signal])

for i in range(M):
    shift = np.random.randint(N)
    data = np.random.normal(0, sigma, N) + signal_circ[shift:shift+N]
    data_fft = fft(data)
    data_sum_tot += data_fft[0]
    data_sqr_tot += np.square(np.abs(data_fft[0]))
    data_pwrspec_tot += data_fft * np.conj(data_fft)
    data_bispec_tot += np.outer(data_fft, np.conj(data_fft)) * np.transpose(circulant(data_fft))


sigma_est = np.sqrt((data_sqr_tot/M - np.square(np.abs(data_sum_tot/M)))/N)
data_mean = data_sum_tot/(M*N)
data_pwrspec = data_pwrspec_tot/M - (sigma_est**2) * N
A = np.diag(np.ones(N))
A[0, :] += np.ones(N)
A[:, 0] += np.ones(N)
data_bispec = data_bispec_tot/M - (sigma_est**2) * (N**2) * data_mean * A

if __name__ == '__main__':
    print('Sigma difference = ' + str(sigma - sigma_est))
    print('Mean difference = ' + str(signal_mean - data_mean))
    print('Power spectrum mean difference = ' + str(np.mean(signal_pwrspec - data_pwrspec)))
    print('Bispectrum mean difference = ' + str(np.mean(signal_bispec - data_bispec)))
